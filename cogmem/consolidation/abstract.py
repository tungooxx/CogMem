"""Convert episodes to training data: SFT pairs and DPO preference pairs.

Two dataset types:
1. SFT dataset: high-Q episodes -> (instruction, response) pairs
2. DPO preference dataset: three sources of pairs that work from Cycle 0
"""

import json
from pathlib import Path


def episode_to_training_pair(episode: dict, include_failures: bool = True) -> dict | None:
    if not episode.get("script"):
        return None
    if not include_failures and not episode.get("success"):
        return None
    return {
        "instruction": episode["task_description"],
        "response": episode["script"],
        "weight": max(episode.get("q_value", 0.0), 0.01),
        "success": episode.get("success", False),
        "source_episode": episode["episode_id"],
    }


def q_weighted_duplicates(pairs: list[dict]) -> list[dict]:
    result = []
    for pair in pairs:
        copies = max(1, round(pair["weight"] * 3))
        result.extend([pair] * copies)
    return result


def prepare_training_dataset(
    episodes: list[dict],
    replay_buffer: list[dict] | None = None,
    include_failures: bool = True,
) -> list[dict]:
    pairs = []
    for ep in episodes:
        pair = episode_to_training_pair(ep, include_failures=include_failures)
        if pair is not None:
            pairs.append(pair)

    if replay_buffer:
        for example in replay_buffer:
            pairs.append({
                "instruction": example["instruction"],
                "response": example["response"],
                "weight": 1.0,
                "source_episode": "replay",
            })

    return pairs


def save_as_jsonl(pairs: list[dict], path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    weighted = q_weighted_duplicates(pairs)
    with open(path, "w") as f:
        for pair in weighted:
            obj = {
                "messages": [
                    {"role": "user", "content": pair["instruction"]},
                    {"role": "assistant", "content": pair["response"]},
                ]
            }
            f.write(json.dumps(obj) + "\n")
    return path


# -------------------------------------------------------------------------
# DPO preference pairs — works from Cycle 0
# -------------------------------------------------------------------------

def prepare_preference_dataset(
    all_episodes: list[dict],
    config,
) -> list[dict]:
    """Generate DPO preference pairs from real experience.

    Two sources — both from the model's actual task attempts:
      Source 1: Within-episode (passed attempt vs failed attempt, same task)
      Source 2: Cross-cycle (same task, different cycles with different outcomes)

    Cycle 0: pairs come from within-episode retries (if any).
    Cycle 1+: cross-cycle pairs appear as the model re-attempts tasks.
    """
    pairs = []

    # ═══ Source 1: Within-episode pairs ═══
    # Task had multiple attempts: final success vs earlier failure
    within_count = 0
    for ep in all_episodes:
        trajectory = ep.get("trajectory", [])
        if len(trajectory) < 2 or not ep.get("success"):
            continue

        chosen_code = ep.get("final_code") or ep.get("generated_code")
        if not chosen_code:
            continue

        # First attempt is usually the failure
        rejected_code = trajectory[0].get("code", "")

        if (rejected_code
                and len(rejected_code.strip()) > 20
                and chosen_code.strip() != rejected_code.strip()):
            pairs.append({
                "prompt": ep.get("task_description", ""),
                "chosen": chosen_code,
                "rejected": rejected_code,
            })
            within_count += 1

    # ═══ Source 2: Same-task across cycles (cycle 1+ only) ═══
    # Real experience: model attempted same task in different cycles
    tasks_by_id: dict[str, list[dict]] = {}
    for ep in all_episodes:
        task_id = ep.get("task_id")
        if not task_id:
            continue
        tasks_by_id.setdefault(task_id, []).append(ep)

    cross_cycle_count = 0
    for _, eps in tasks_by_id.items():
        if len(eps) < 2:
            continue

        eps.sort(key=lambda x: x.get("q_value", 0), reverse=True)
        best = eps[0]
        worst = eps[-1]

        if best.get("q_value", 0) - worst.get("q_value", 0) < config.min_q_gap:
            continue

        best_code = (best.get("final_code")
                     or best.get("generated_code")
                     or best.get("script"))
        worst_code = (worst.get("final_code")
                      or worst.get("generated_code")
                      or worst.get("script"))

        if (best_code and worst_code
                and len(best_code) > 20 and len(worst_code) > 20
                and best_code.strip() != worst_code.strip()):
            pairs.append({
                "prompt": best.get("task_description", ""),
                "chosen": best_code,
                "rejected": worst_code,
            })
            cross_cycle_count += 1

    print(f"  DPO pairs total: {len(pairs)}")
    print(f"    Within-episode:  {within_count}")
    print(f"    Cross-cycle:     {cross_cycle_count}")

    return pairs

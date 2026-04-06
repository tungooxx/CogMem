"""Convert episodes to training data: SFT pairs and DPO preference pairs.

Two dataset types:
1. SFT dataset: high-Q episodes -> (instruction, response) pairs
2. DPO preference dataset: three sources of pairs that work from Cycle 0
"""

import json
import random as _random
from pathlib import Path

import numpy as np


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
    """Generate DPO preference pairs. Works for ALL cycles.

    Three pairing strategies:
      Source 1: Within-episode (attempt that passed vs attempt that failed)
      Source 2: Same-domain (successful code vs failed code, same domain)
      Source 3: Same-task across cycles (only available cycle 1+)
    """
    rng = _random.Random(config.seed)
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
                and len(rejected_code) > 20
                and chosen_code != rejected_code):
            pairs.append({
                "prompt": ep.get("task_description", ""),
                "chosen": chosen_code,
                "rejected": rejected_code,
            })
            within_count += 1

    # ═══ Source 2: Success vs failure pairs (random sampling) ═══
    successes = []
    failures = []
    for ep in all_episodes:
        code = ep.get("final_code") or ep.get("generated_code") or ep.get("script")
        if not code or len(code.strip()) <= 20:
            continue
        if ep.get("success"):
            successes.append((ep, code))
        else:
            failures.append((ep, code))

    direct_count = 0
    for winner_ep, winner_code in successes:
        sampled = failures if len(failures) <= 3 else rng.sample(failures, 3)
        for loser_ep, loser_code in sampled:
            pairs.append({
                "prompt": winner_ep.get("task_description", ""),
                "chosen": winner_code,
                "rejected": loser_code,
            })
            direct_count += 1

    # ═══ Source 3: Same-task across cycles (cycle 1+ only) ═══
    tasks_by_id: dict[str, list[dict]] = {}
    for ep in all_episodes:
        task_id = ep.get("task_id", ep.get("episode_id", ""))
        tasks_by_id.setdefault(task_id, []).append(ep)

    cross_cycle_count = 0
    for task_id, eps in tasks_by_id.items():
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

        if best_code and worst_code and len(worst_code) > 20:
            pairs.append({
                "prompt": best.get("task_description", ""),
                "chosen": best_code,
                "rejected": worst_code,
            })
            cross_cycle_count += 1

    print(f"  DPO pairs total: {len(pairs)}")
    print(f"    Within-episode:  {within_count}")
    print(f"    Success-vs-fail: {direct_count}")
    print(f"    Cross-cycle:     {cross_cycle_count}")

    return pairs

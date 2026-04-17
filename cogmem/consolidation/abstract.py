"""Convert episodes or validated skill cards to training data.

Two dataset types:
1. SFT dataset: high-Q episodes -> (instruction, response) pairs
2. DPO preference dataset: three sources of pairs that work from Cycle 0
"""

import json
from pathlib import Path

from cogmem.consolidation.select import filter_manifest_eligible
from cogmem.memory.schema import get_episode_helpfulness


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def episode_to_training_pair(episode: dict, include_failures: bool = False) -> dict | None:
    response = episode.get("final_code") or episode.get("generated_code") or episode.get("script")
    if not response:
        return None
    if not include_failures and not episode.get("success"):
        return None
    return {
        "instruction": episode["task_description"],
        "response": response,
        "weight": max(get_episode_helpfulness(episode, 0.0), 0.01),
        "success": episode.get("success", False),
        "source_episode": episode["episode_id"],
    }


def _skill_card_manifest_eligible(card: dict, config) -> bool:
    allowed = set(getattr(config, "allowed_manifest_ids", []) or [])
    blocked = set(getattr(config, "blocked_manifest_ids", []) or [])
    require_manifest_match = bool(getattr(config, "require_manifest_match", False))
    manifest_ids = {
        str(manifest_id)
        for manifest_id in (card.get("manifest_ids", []) or [])
        if manifest_id
    }

    if blocked and manifest_ids & blocked:
        return False
    if allowed and not (manifest_ids & allowed):
        return False
    if require_manifest_match and not manifest_ids:
        return False
    return True


def _best_evidence_episode(
    card: dict,
    episodes_by_id: dict[str, dict],
) -> dict | None:
    evidence = [
        episodes_by_id[episode_id]
        for episode_id in (card.get("evidence_episode_ids", []) or [])
        if episode_id in episodes_by_id
    ]
    if not evidence:
        return None
    return max(evidence, key=lambda episode: get_episode_helpfulness(episode, 0.0))


def _build_skill_curriculum_instruction(
    card: dict,
    *,
    include_anti_patterns: bool = True,
) -> str:
    task_type = card.get("task_type") or "general"
    domain = card.get("domain") or "general"
    triggers = [str(item).strip() for item in (card.get("triggers", []) or []) if str(item).strip()]
    plan_steps = [str(item).strip() for item in (card.get("plan_steps", []) or []) if str(item).strip()]
    anti_patterns = [str(item).strip() for item in (card.get("anti_patterns", []) or []) if str(item).strip()]

    lines = [
        f"Write Python code for a {task_type} task in the {domain} domain.",
    ]
    if triggers:
        lines.append("This pattern is especially useful when the task involves: " + ", ".join(triggers[:5]) + ".")
    if plan_steps:
        lines.append("Follow this procedure:")
        for idx, step in enumerate(plan_steps[:5], start=1):
            lines.append(f"{idx}. {step}")
    if include_anti_patterns and anti_patterns:
        lines.append("Avoid these mistakes:")
        for pattern in anti_patterns[:4]:
            lines.append(f"- {pattern}")
    return "\n".join(lines)


def skill_card_to_curriculum_pair(
    card: dict,
    episodes_by_id: dict[str, dict],
    *,
    include_anti_patterns: bool = True,
) -> dict | None:
    """Create one generalized curriculum example from a promoted skill card."""
    best_episode = _best_evidence_episode(card, episodes_by_id)
    if best_episode is None:
        return None
    response = (
        best_episode.get("final_code")
        or best_episode.get("generated_code")
        or best_episode.get("script")
    )
    if not response:
        return None

    confidence = _clamp01(card.get("confidence", 0.0))
    transfer_gain = _clamp01(card.get("transfer_gain", 0.0))
    best_helpfulness = _clamp01(get_episode_helpfulness(best_episode, 0.0))
    return {
        "instruction": _build_skill_curriculum_instruction(
            card,
            include_anti_patterns=include_anti_patterns,
        ),
        "response": response,
        "weight": max(
            0.01,
            min(
                1.0,
                0.45 * confidence + 0.35 * transfer_gain + 0.20 * best_helpfulness,
            ),
        ),
        "success": True,
        "source_episode": best_episode["episode_id"],
        "source_skill_card": card["skill_id"],
        "source_kind": "skill_curriculum",
        "skill_confidence": confidence,
        "skill_transfer_gain": float(card.get("transfer_gain", 0.0) or 0.0),
    }


def skill_card_to_training_pairs(
    card: dict,
    episodes_by_id: dict[str, dict],
    *,
    include_failures: bool = False,
) -> list[dict]:
    """Build weighted SFT pairs from one promoted skill card and its evidence."""
    confidence = _clamp01(card.get("confidence", 0.0))
    transfer_gain = _clamp01(card.get("transfer_gain", 0.0))
    evidence_ids = card.get("evidence_episode_ids", []) or []

    pairs: list[dict] = []
    for episode_id in evidence_ids:
        episode = episodes_by_id.get(episode_id)
        if episode is None:
            continue
        pair = episode_to_training_pair(episode, include_failures=include_failures)
        if pair is None:
            continue
        base_weight = float(pair["weight"])
        pair["weight"] = max(
            0.01,
            min(
                1.0,
                0.50 * base_weight + 0.30 * confidence + 0.20 * transfer_gain,
            ),
        )
        pair["source_skill_card"] = card["skill_id"]
        pair["source_kind"] = "skill_evidence"
        pair["skill_confidence"] = confidence
        pair["skill_transfer_gain"] = float(card.get("transfer_gain", 0.0) or 0.0)
        pairs.append(pair)
    return pairs


def prepare_skill_training_dataset(
    skill_cards: list[dict],
    episodes: list[dict],
    replay_buffer: list[dict] | None = None,
    include_failures: bool = False,
    config=None,
) -> list[dict]:
    """Build SFT pairs from promoted skill cards, backed by evidence episodes."""
    eligible_episodes = filter_manifest_eligible(episodes, config) if config is not None else list(episodes)
    episodes_by_id = {ep["episode_id"]: ep for ep in eligible_episodes if ep.get("episode_id")}
    curriculum_examples_per_card = int(getattr(config, "skill_curriculum_examples_per_card", 1) or 0)
    include_anti_patterns = bool(getattr(config, "skill_curriculum_include_anti_patterns", True))

    pairs: list[dict] = []
    for card in skill_cards:
        if config is not None and not _skill_card_manifest_eligible(card, config):
            continue
        pairs.extend(
            skill_card_to_training_pairs(
                card,
                episodes_by_id,
                include_failures=include_failures,
            )
        )
        if curriculum_examples_per_card > 0:
            curriculum_pair = skill_card_to_curriculum_pair(
                card,
                episodes_by_id,
                include_anti_patterns=include_anti_patterns,
            )
            if curriculum_pair is not None:
                pairs.extend([curriculum_pair] * curriculum_examples_per_card)

    if replay_buffer:
        for example in replay_buffer:
            if config is not None and not filter_manifest_eligible([example], config):
                continue
            pairs.append({
                "instruction": example["instruction"],
                "response": example["response"],
                "weight": 1.0,
                "source_episode": "replay",
                "source_skill_card": "replay",
                "source_kind": "replay",
            })

    return pairs


def q_weighted_duplicates(pairs: list[dict]) -> list[dict]:
    result = []
    for pair in pairs:
        copies = max(1, round(pair["weight"] * 3))
        result.extend([pair] * copies)
    return result


def prepare_training_dataset(
    episodes: list[dict],
    replay_buffer: list[dict] | None = None,
    include_failures: bool = False,
    config=None,
) -> list[dict]:
    if config is not None:
        episodes = filter_manifest_eligible(episodes, config)
    pairs = []
    for ep in episodes:
        pair = episode_to_training_pair(ep, include_failures=include_failures)
        if pair is not None:
            pairs.append(pair)

    if replay_buffer:
        for example in replay_buffer:
            if config is not None and not filter_manifest_eligible([example], config):
                continue
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
    all_episodes = filter_manifest_eligible(all_episodes, config)
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

        # Find the best failed attempt (most recent with real code)
        rejected_code = ""
        for step in reversed(trajectory):
            if step.get("test_result") != "PASS" and step.get("code", "").strip():
                rejected_code = step["code"]
                break

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

        eps.sort(key=lambda x: get_episode_helpfulness(x, 0.0), reverse=True)
        best = eps[0]
        worst = eps[-1]

        if get_episode_helpfulness(best, 0.0) - get_episode_helpfulness(worst, 0.0) < config.min_q_gap:
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

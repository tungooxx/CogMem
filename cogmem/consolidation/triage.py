"""Q-value triage: split episodes into training zones.

The core novelty of CogMem's consolidation pipeline. Different Q-value
zones get different training treatments during the "sleep" phase:

    High Q  (>= threshold): SFT/DoRA — memorize proven patterns
    Middle Q (mid..high):   GRPO RL  — practice reasoning on hard problems
    Low Q   (< mid):        Keep as episodic memory — too hard to learn from

This mirrors memory consolidation in neuroscience: well-learned skills
become procedural (SFT), challenging experiences get rehearsed (GRPO),
and novel/confusing experiences stay as episodic memories.
"""

import random as _random


def triage_episodes(
    episodes: list[dict],
    config,
) -> dict[str, list[dict]]:
    """Split episodes into three Q-value zones.

    Args:
        episodes: List of episode dicts with 'q_value' key.
        config: ConsolidationConfig with q_high_threshold and q_mid_low.

    Returns:
        {"high": [...], "middle": [...], "low": [...]}
    """
    zones: dict[str, list[dict]] = {"high": [], "middle": [], "low": []}

    for ep in episodes:
        q = ep.get("q_value", 0.0)
        if q >= config.q_high_threshold:
            zones["high"].append(ep)
        elif q >= config.q_mid_low:
            zones["middle"].append(ep)
        else:
            zones["low"].append(ep)

    _log_triage(zones, config)
    return zones


def select_anchors(
    high_episodes: list[dict],
    config,
) -> list[dict]:
    """Select diverse anchor episodes from high-Q zone for GRPO stability.

    During GRPO training, high-Q anchors are mixed in to prevent the
    model from forgetting what it already knows (catastrophic forgetting).

    Selects top episodes per task_type for diversity.
    """
    if not high_episodes:
        return []

    target = min(config.grpo_anchor_count, len(high_episodes))

    # Group by task_type for diversity
    by_type: dict[str, list[dict]] = {}
    for ep in high_episodes:
        key = ep.get("task_type", "general")
        by_type.setdefault(key, []).append(ep)

    # Take top-Q episodes from each type
    anchors = []
    per_type = max(1, target // max(len(by_type), 1))

    for _type, eps in by_type.items():
        sorted_eps = sorted(eps, key=lambda x: x.get("q_value", 0), reverse=True)
        anchors.extend(sorted_eps[:per_type])

    # Trim to target, keeping highest Q
    anchors = sorted(anchors, key=lambda x: x.get("q_value", 0), reverse=True)
    anchors = anchors[:target]

    print(f"  Selected {len(anchors)} anchor episodes for GRPO stability")
    return anchors


def split_holdout(
    episodes: list[dict],
    fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split episodes into train and holdout sets.

    Returns:
        (train_episodes, holdout_episodes)
    """
    if not episodes:
        return [], []

    if len(episodes) == 1:
        return list(episodes), []

    rng = _random.Random(seed)
    shuffled = list(episodes)
    rng.shuffle(shuffled)

    n_holdout = min(len(shuffled) - 1, max(1, int(len(shuffled) * fraction)))
    holdout = shuffled[:n_holdout]
    train = shuffled[n_holdout:]
    return train, holdout


def _log_triage(zones: dict[str, list[dict]], config) -> None:
    total = sum(len(v) for v in zones.values())
    print(f"Q-value triage ({total} episodes):")
    print(
        f"  High   (Q >= {config.q_high_threshold}): "
        f"{len(zones['high'])} episodes -> SFT"
    )
    print(
        f"  Middle ({config.q_mid_low} <= Q < {config.q_high_threshold}): "
        f"{len(zones['middle'])} episodes -> GRPO"
    )
    print(
        f"  Low    (Q < {config.q_mid_low}): "
        f"{len(zones['low'])} episodes -> keep episodic"
    )
    for zone_name, eps in zones.items():
        if eps:
            q_vals = [ep.get("q_value", 0) for ep in eps]
            print(
                f"  {zone_name}: Q range [{min(q_vals):.2f}, {max(q_vals):.2f}], "
                f"mean={sum(q_vals) / len(q_vals):.2f}"
            )

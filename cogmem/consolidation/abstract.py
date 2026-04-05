"""Convert episodes to training data: SFT pairs and DPO preference pairs.

Two dataset types:
1. SFT dataset: high-Q episodes -> (instruction, response) pairs
2. DPO preference dataset: Q-value pairs -> (prompt, chosen, rejected)
"""

import json
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
# DPO preference pairs from Q-values
# -------------------------------------------------------------------------

def prepare_preference_dataset(
    all_episodes: list[dict],
    config,
) -> list[dict]:
    """Create DPO preference pairs from Q-value differences.

    For episodes about similar tasks with different Q-values:
    - High-Q code is "chosen"
    - Low-Q code is "rejected"
    - Q-value gap = preference strength

    This teaches the model to distinguish good code from bad code.
    """
    # Group episodes by task_id (exact match — same task, different attempts)
    by_task: dict[str, list[dict]] = {}
    for ep in all_episodes:
        key = ep.get("task_id", ep.get("episode_id", ""))
        by_task.setdefault(key, []).append(ep)

    # Also group by embedding similarity for cross-task pairs
    clusters = _cluster_by_embedding(all_episodes, config)

    preference_pairs = []

    # First: exact task matches (strongest signal)
    for task_id, eps in by_task.items():
        pairs = _make_pairs_from_group(eps, config)
        preference_pairs.extend(pairs)

    # Second: embedding-similar tasks (weaker but more pairs)
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        pairs = _make_pairs_from_group(cluster, config)
        # Avoid duplicates from exact matches
        existing = {(p["prompt"][:100], p["chosen"][:100]) for p in preference_pairs}
        for p in pairs:
            key = (p["prompt"][:100], p["chosen"][:100])
            if key not in existing:
                preference_pairs.append(p)
                existing.add(key)

    print(f"  Preference pairs: {len(preference_pairs)} "
          f"(from {len(all_episodes)} episodes)")
    return preference_pairs


def _make_pairs_from_group(
    episodes: list[dict], config,
) -> list[dict]:
    """Create chosen/rejected pairs from a group of episodes."""
    eps_with_code = []
    for ep in episodes:
        code = ep.get("final_code") or ep.get("generated_code") or ep.get("script")
        if code and len(code.strip()) > 10:
            eps_with_code.append((ep, code))

    if len(eps_with_code) < 2:
        return []

    # Sort by Q-value descending
    eps_with_code.sort(key=lambda x: x[0].get("q_value", 0), reverse=True)

    pairs = []
    for i, (winner, winner_code) in enumerate(eps_with_code):
        for loser, loser_code in eps_with_code[i + 1:]:
            q_gap = winner.get("q_value", 0) - loser.get("q_value", 0)
            if q_gap < config.min_q_gap:
                continue

            pairs.append({
                "prompt": winner.get("task_description", ""),
                "chosen": winner_code,
                "rejected": loser_code,
            })

    return pairs


def _cluster_by_embedding(
    episodes: list[dict], config, threshold: float = 0.75,
) -> list[list[dict]]:
    """Group episodes by embedding similarity."""
    # Only use episodes that have embeddings
    with_emb = [ep for ep in episodes if ep.get("intent_embedding")]
    if len(with_emb) < 2:
        return []

    try:
        from sklearn.cluster import AgglomerativeClustering

        embeddings = np.array([ep["intent_embedding"] for ep in with_emb])
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        clusters: dict[int, list[dict]] = {}
        for ep, label in zip(with_emb, labels):
            clusters.setdefault(label, []).append(ep)

        return list(clusters.values())

    except ImportError:
        # sklearn not available — skip embedding clustering
        return []

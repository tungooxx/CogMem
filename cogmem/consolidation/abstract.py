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
            pairs.append(
                {
                    "instruction": example["instruction"],
                    "response": example["response"],
                    "weight": 1.0,
                    "source_episode": "replay",
                }
            )

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

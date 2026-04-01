import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


class MemoryBank:
    def __init__(self, episodes: list[dict]):
        self._episodes = episodes
        self._index = {ep["episode_id"]: ep for ep in episodes}

    @classmethod
    def load(cls, path: str) -> "MemoryBank":
        with open(path) as f:
            data = json.load(f)
        return cls(data)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._episodes, f, indent=2)

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self):
        return iter(self._episodes)

    def get(self, episode_id: str) -> dict | None:
        return self._index.get(episode_id)

    def successful(self) -> list[dict]:
        return [ep for ep in self._episodes if ep["success"]]

    def by_task_type(self, task_type: str) -> list[dict]:
        return [ep for ep in self._episodes if ep["task_type"] == task_type]

    def task_types(self) -> set[str]:
        return {ep["task_type"] for ep in self._episodes}

    def stratified_holdout(
        self, n: int, seed: int = 42
    ) -> tuple[list[dict], list[dict]]:
        rng = random.Random(seed)
        by_type = defaultdict(list)
        for ep in self._episodes:
            by_type[ep["task_type"]].append(ep)

        holdout = []
        remaining_budget = n
        types = sorted(by_type.keys())
        per_type = max(1, n // len(types))

        for t in types:
            eps = list(by_type[t])
            rng.shuffle(eps)
            take = min(per_type, len(eps), remaining_budget)
            holdout.extend(eps[:take])
            remaining_budget -= take
            if remaining_budget <= 0:
                break

        # Fill remaining budget from any type
        if remaining_budget > 0:
            used_ids = {ep["episode_id"] for ep in holdout}
            pool = [ep for ep in self._episodes if ep["episode_id"] not in used_ids]
            rng.shuffle(pool)
            holdout.extend(pool[:remaining_budget])

        holdout_ids = {ep["episode_id"] for ep in holdout}
        available = [ep for ep in self._episodes if ep["episode_id"] not in holdout_ids]
        return holdout, available

    def summary_metrics(self) -> dict:
        q_values = [ep["q_value"] for ep in self._episodes]
        successes = [ep for ep in self._episodes if ep["success"]]

        by_type = defaultdict(lambda: {"success": 0, "total": 0})
        for ep in self._episodes:
            by_type[ep["task_type"]]["total"] += 1
            if ep["success"]:
                by_type[ep["task_type"]]["success"] += 1

        return {
            "total_episodes": len(self._episodes),
            "success_rate": len(successes) / len(self._episodes) if self._episodes else 0,
            "success_rate_by_type": {
                t: d["success"] / d["total"] for t, d in sorted(by_type.items())
            },
            "q_value_stats": {
                "mean": mean(q_values),
                "std": stdev(q_values) if len(q_values) > 1 else 0,
                "min": min(q_values),
                "max": max(q_values),
            },
            "high_q_episodes": sum(1 for q in q_values if q > 0.7),
            "low_q_episodes": sum(1 for q in q_values if q < 0.3),
        }

    def sha256(self) -> str:
        content = json.dumps(self._episodes, sort_keys=True).encode()
        return hashlib.sha256(content).hexdigest()

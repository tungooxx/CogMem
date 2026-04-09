"""CogMem Memory Bank — unified episodic memory with Q-value support.

Episodes can come from any collection method (runner.py, collect_sequential.py).
Q-values are tracked per episode and updated based on retrieval helpfulness.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


# Q-learning defaults
Q_INITIAL = 0.5
Q_ALPHA = 0.3


class MemoryBank:
    def __init__(self, episodes: list[dict]):
        self._episodes = episodes
        self._index = {ep["episode_id"]: ep for ep in episodes}

    @classmethod
    def load(cls, path: str) -> "MemoryBank":
        p = Path(path)
        if not p.exists():
            return cls([])
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

    @property
    def episodes(self) -> tuple[dict, ...]:
        return tuple(self._episodes)

    def add(self, episode: dict) -> None:
        """Add an episode to the bank."""
        if "episode_id" not in episode:
            raise ValueError("Episode must contain 'episode_id'")
        eid = episode["episode_id"]
        if eid in self._index:
            # Update existing episode in-place
            idx = next(i for i, ep in enumerate(self._episodes) if ep["episode_id"] == eid)
            self._episodes[idx] = episode
            self._index[eid] = episode
        else:
            self._episodes.append(episode)
            self._index[eid] = episode

    def get(self, episode_id: str) -> dict | None:
        return self._index.get(episode_id)

    def successful(self) -> list[dict]:
        return [ep for ep in self._episodes if ep.get("success")]

    def by_task_type(self, task_type: str) -> list[dict]:
        return [ep for ep in self._episodes if ep.get("task_type", "general") == task_type]

    def task_types(self) -> set[str]:
        return {ep.get("task_type", "general") for ep in self._episodes}

    def completed_task_ids(self) -> set[str]:
        """Get set of task_ids that have been attempted."""
        return {ep.get("task_id", "") for ep in self._episodes if ep.get("task_id")}

    # -----------------------------------------------------------------
    # Q-value updates
    # -----------------------------------------------------------------

    def update_q(self, episode_id: str, task_succeeded: bool) -> None:
        """Update Q-value of an episode based on retrieval outcome.

        Called when this episode is retrieved by another task.
        If that task succeeded → this memory was useful → Q goes UP.
        If that task failed → this memory didn't help → Q goes DOWN.
        """
        ep = self._index.get(episode_id)
        if ep is None:
            return

        reward = 1.0 if task_succeeded else 0.0
        old_q = ep.get("q_value", Q_INITIAL)
        ep["q_value"] = old_q + Q_ALPHA * (reward - old_q)
        ep["q_visits"] = ep.get("q_visits", 0) + 1
        if task_succeeded:
            ep["q_successes"] = ep.get("q_successes", 0) + 1
        else:
            ep["q_failures"] = ep.get("q_failures", 0) + 1

    # -----------------------------------------------------------------
    # Holdout splitting
    # -----------------------------------------------------------------

    def stratified_holdout(
        self, n: int, seed: int = 42
    ) -> tuple[list[dict], list[dict]]:
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        rng = random.Random(seed)
        by_type = defaultdict(list)
        for ep in self._episodes:
            by_type[ep.get("task_type", "general")].append(ep)

        holdout = []
        remaining_budget = n
        types = sorted(by_type.keys())
        per_type = max(1, n // max(len(types), 1))

        for t in types:
            eps = list(by_type[t])
            rng.shuffle(eps)
            take = min(per_type, len(eps), remaining_budget)
            holdout.extend(eps[:take])
            remaining_budget -= take
            if remaining_budget <= 0:
                break

        if remaining_budget > 0:
            used_ids = {ep["episode_id"] for ep in holdout}
            pool = [ep for ep in self._episodes if ep["episode_id"] not in used_ids]
            rng.shuffle(pool)
            holdout.extend(pool[:remaining_budget])

        holdout_ids = {ep["episode_id"] for ep in holdout}
        available = [ep for ep in self._episodes if ep["episode_id"] not in holdout_ids]
        return holdout, available

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def summary_metrics(self) -> dict:
        if not self._episodes:
            return {
                "total_episodes": 0,
                "success_rate": 0.0,
                "success_rate_by_type": {},
                "q_value_stats": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "high_q_episodes": 0,
                "mid_q_episodes": 0,
                "low_q_episodes": 0,
                "ever_retrieved": 0,
            }

        q_values = [ep.get("q_value", Q_INITIAL) for ep in self._episodes]
        successes = [ep for ep in self._episodes if ep.get("success")]
        visited = [ep for ep in self._episodes if ep.get("q_visits", 0) > 0]

        by_type = defaultdict(lambda: {"success": 0, "total": 0})
        for ep in self._episodes:
            by_type[ep.get("task_type", "general")]["total"] += 1
            if ep.get("success"):
                by_type[ep.get("task_type", "general")]["success"] += 1

        return {
            "total_episodes": len(self._episodes),
            "success_rate": len(successes) / len(self._episodes),
            "success_rate_by_type": {
                t: d["success"] / d["total"] for t, d in sorted(by_type.items())
            },
            "q_value_stats": {
                "mean": mean(q_values),
                "std": stdev(q_values) if len(q_values) > 1 else 0,
                "min": min(q_values),
                "max": max(q_values),
            },
            "high_q_episodes": sum(1 for q in q_values if q >= 0.7),
            "mid_q_episodes": sum(1 for q in q_values if 0.3 <= q < 0.7),
            "low_q_episodes": sum(1 for q in q_values if q < 0.3),
            "ever_retrieved": len(visited),
        }

    def sha256(self) -> str:
        content = json.dumps(self._episodes, sort_keys=True).encode()
        return hashlib.sha256(content).hexdigest()

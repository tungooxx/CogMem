"""Compatibility wrapper around the typed episodic store."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from statistics import mean, stdev

from cogmem.memory.episodic_store import EpisodicStore
from cogmem.memory.schema import DEFAULT_EPISODE_HELPFULNESS, get_episode_helpfulness, set_episode_helpfulness


Q_INITIAL = DEFAULT_EPISODE_HELPFULNESS
Q_ALPHA = 0.3


class MemoryBank(EpisodicStore):
    """Legacy episodic memory interface backed by the typed EpisodicStore."""

    def successful(self) -> list[dict]:
        return self.filter(success=True)

    def by_task_type(self, task_type: str) -> list[dict]:
        return self.filter(task_type=task_type)

    def task_types(self) -> set[str]:
        return {ep.get("task_type", "general") for ep in self._episodes}

    def completed_task_ids(self) -> set[str]:
        return {ep.get("task_id", "") for ep in self._episodes if ep.get("task_id")}

    def update_q(self, episode_id: str, task_succeeded: bool) -> None:
        episode = self._index.get(episode_id)
        if episode is None:
            return
        reward = 1.0 if task_succeeded else 0.0
        old_score = get_episode_helpfulness(episode, Q_INITIAL)
        new_score = old_score + Q_ALPHA * (reward - old_score)
        set_episode_helpfulness(episode, new_score, mirror_legacy_q_value=True)
        episode["q_visits"] = episode.get("q_visits", 0) + 1
        if task_succeeded:
            episode["q_successes"] = episode.get("q_successes", 0) + 1
        else:
            episode["q_failures"] = episode.get("q_failures", 0) + 1

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

        for task_type in types:
            eps = list(by_type[task_type])
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

    def summary_metrics(self) -> dict:
        if not self._episodes:
            empty_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            return {
                "total_episodes": 0,
                "success_rate": 0.0,
                "success_rate_by_type": {},
                "episode_helpfulness_stats": empty_stats,
                "q_value_stats": empty_stats,
                "high_q_episodes": 0,
                "mid_q_episodes": 0,
                "low_q_episodes": 0,
                "ever_retrieved": 0,
            }

        helpfulness_values = [get_episode_helpfulness(ep, Q_INITIAL) for ep in self._episodes]
        successes = [ep for ep in self._episodes if ep.get("success")]
        visited = [ep for ep in self._episodes if ep.get("q_visits", 0) > 0]

        by_type = defaultdict(lambda: {"success": 0, "total": 0})
        for ep in self._episodes:
            by_type[ep.get("task_type", "general")]["total"] += 1
            if ep.get("success"):
                by_type[ep.get("task_type", "general")]["success"] += 1

        stats = {
            "mean": mean(helpfulness_values),
            "std": stdev(helpfulness_values) if len(helpfulness_values) > 1 else 0,
            "min": min(helpfulness_values),
            "max": max(helpfulness_values),
        }
        return {
            "total_episodes": len(self._episodes),
            "success_rate": len(successes) / len(self._episodes),
            "success_rate_by_type": {
                task_type: values["success"] / values["total"]
                for task_type, values in sorted(by_type.items())
            },
            "episode_helpfulness_stats": stats,
            "q_value_stats": dict(stats),
            "high_q_episodes": sum(1 for score in helpfulness_values if score >= 0.7),
            "mid_q_episodes": sum(1 for score in helpfulness_values if 0.3 <= score < 0.7),
            "low_q_episodes": sum(1 for score in helpfulness_values if score < 0.3),
            "ever_retrieved": len(visited),
        }

    def sha256(self) -> str:
        content = json.dumps(self._episodes, sort_keys=True).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

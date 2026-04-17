"""Typed episodic storage helpers for CogMem."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from cogmem.memory.schema import normalize_episode_metrics


def infer_error_family(error: str | None) -> str | None:
    """Map a raw error string to a coarse family label."""
    if not error:
        return None
    text = str(error)
    match = re.search(r"([A-Za-z]+Error)", text)
    if match:
        return match.group(1)
    lowered = text.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "Timeout"
    if "assert" in lowered:
        return "AssertionError"
    if "syntax" in lowered:
        return "SyntaxError"
    return None


def _prompt_hash(task_description: str) -> str | None:
    if not task_description:
        return None
    return hashlib.sha256(task_description.encode("utf-8")).hexdigest()


def _episode_id(episode: dict) -> str:
    if episode.get("episode_id"):
        return str(episode["episode_id"])
    seed = json.dumps(
        {
            "task_id": episode.get("task_id"),
            "task_description": episode.get("task_description"),
            "timestamp": episode.get("timestamp"),
            "script": episode.get("script"),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return f"episode_{hashlib.sha256(seed).hexdigest()[:16]}"


def _default_validation_recipe(episode: dict) -> dict:
    if episode.get("source_benchmark") == "bigcodebench" or episode.get("task_type") == "bigcodebench":
        return {
            "kind": "bigcodebench_exec",
            "entry_point": episode.get("entry_point", ""),
            "task_id": episode.get("task_id"),
        }
    return {"kind": "episode_outcome"}


def normalize_episode_record(
    episode: dict,
    *,
    copy_episode: bool = False,
) -> dict:
    """Normalize an episode dict into the typed episodic schema."""
    target = deepcopy(episode) if copy_episode else episode
    target = normalize_episode_metrics(target, copy_episode=False)
    target["episode_id"] = _episode_id(target)
    task_description = str(target.get("task_description", "") or "")
    if not target.get("prompt_hash"):
        target["prompt_hash"] = _prompt_hash(task_description)
    if target.get("retrieved_ids") is None:
        target["retrieved_ids"] = list(target.get("retrieved_from", []) or [])
    else:
        target.setdefault("retrieved_ids", list(target.get("retrieved_from", []) or []))
    if target.get("adapter_ids") is None:
        target["adapter_ids"] = []
    else:
        target.setdefault("adapter_ids", [])
    if not target.get("error_family"):
        target["error_family"] = infer_error_family(target.get("error"))
    if not target.get("validation_recipe"):
        target["validation_recipe"] = _default_validation_recipe(target)
    if not target.get("source_benchmark"):
        target["source_benchmark"] = "unknown"
    return target


class EpisodicStore:
    """Typed episodic storage with filtering and lineage validation."""

    def __init__(self, episodes: list[dict] | None = None):
        self._episodes = [normalize_episode_record(ep, copy_episode=True) for ep in (episodes or [])]
        self._index = {ep["episode_id"]: ep for ep in self._episodes}

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self):
        return iter(self._episodes)

    @property
    def episodes(self) -> tuple[dict, ...]:
        return tuple(self._episodes)

    @classmethod
    def load(cls, path: str) -> "EpisodicStore":
        p = Path(path)
        if not p.exists():
            return cls([])
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def load_jsonl(cls, path: str) -> "EpisodicStore":
        p = Path(path)
        if not p.exists():
            return cls([])
        episodes = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return cls(episodes)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._episodes, f, indent=2, ensure_ascii=False)

    def save_jsonl(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for episode in self._episodes:
                f.write(json.dumps(episode, ensure_ascii=False) + "\n")

    @staticmethod
    def append_jsonl(path: str, episode: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        normalized = normalize_episode_record(episode, copy_episode=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    def append(self, episode: dict) -> dict:
        if "episode_id" not in episode:
            raise ValueError("Episode must contain 'episode_id'")
        normalized = normalize_episode_record(episode, copy_episode=True)
        self._episodes.append(normalized)
        self._index[normalized["episode_id"]] = normalized
        return normalized

    def update(self, episode_id: str, **fields) -> dict | None:
        episode = self._index.get(episode_id)
        if episode is None:
            return None
        episode.update(fields)
        normalized = normalize_episode_record(episode, copy_episode=False)
        self._index[episode_id] = normalized
        return normalized

    def get(self, episode_id: str) -> dict | None:
        return self._index.get(episode_id)

    def filter(
        self,
        *,
        split_name: str | None = None,
        task_type: str | None = None,
        success: bool | None = None,
        source_benchmark: str | None = None,
        manifest_id: str | None = None,
        library: str | None = None,
        error_family: str | None = None,
    ) -> list[dict]:
        results = list(self._episodes)
        if split_name is not None:
            results = [ep for ep in results if ep.get("split_name") == split_name]
        if task_type is not None:
            results = [ep for ep in results if ep.get("task_type") == task_type]
        if success is not None:
            results = [ep for ep in results if bool(ep.get("success")) is success]
        if source_benchmark is not None:
            results = [ep for ep in results if ep.get("source_benchmark") == source_benchmark]
        if manifest_id is not None:
            results = [ep for ep in results if ep.get("manifest_id") == manifest_id]
        if library is not None:
            results = [ep for ep in results if library in (ep.get("libs") or [])]
        if error_family is not None:
            results = [ep for ep in results if ep.get("error_family") == error_family]
        return results

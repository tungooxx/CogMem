"""Typed episodic storage helpers for CogMem."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from cogmem.memory.schema import get_episode_helpfulness, normalize_episode_metrics


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


def _dedupe_str_list(values) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalize_runtime_stats(stats: dict | None) -> dict:
    target = dict(stats or {})
    target["retrieved"] = int(target.get("retrieved", 0) or 0)
    target["helped"] = int(target.get("helped", 0) or 0)
    target["hurt"] = int(target.get("hurt", 0) or 0)
    target["preserved_success"] = int(target.get("preserved_success", 0) or 0)
    target["preserved_failure"] = int(target.get("preserved_failure", 0) or 0)
    target["passed"] = int(target.get("passed", 0) or 0)
    target["failed"] = int(target.get("failed", 0) or 0)
    target["domains"] = {str(k): int(v) for k, v in dict(target.get("domains", {}) or {}).items()}
    target["error_families"] = {str(k): int(v) for k, v in dict(target.get("error_families", {}) or {}).items()}
    target["task_ids"] = _dedupe_str_list(target.get("task_ids", []))[:20]
    target["route_breakdown"] = {
        str(route): _normalize_runtime_stats(route_stats)
        for route, route_stats in dict(target.get("route_breakdown", {}) or {}).items()
    }
    return target


def merge_runtime_stats(existing: dict | None, update: dict | None) -> dict:
    base = _normalize_runtime_stats(existing)
    incoming = _normalize_runtime_stats(update)
    merged = dict(base)
    for key in ["retrieved", "helped", "hurt", "preserved_success", "preserved_failure", "passed", "failed"]:
        merged[key] = int(base.get(key, 0)) + int(incoming.get(key, 0))
    domains = dict(base.get("domains", {}) or {})
    for key, value in dict(incoming.get("domains", {}) or {}).items():
        domains[str(key)] = int(domains.get(str(key), 0)) + int(value)
    merged["domains"] = domains
    error_families = dict(base.get("error_families", {}) or {})
    for key, value in dict(incoming.get("error_families", {}) or {}).items():
        error_families[str(key)] = int(error_families.get(str(key), 0)) + int(value)
    merged["error_families"] = error_families
    merged["task_ids"] = _dedupe_str_list(list(base.get("task_ids", [])) + list(incoming.get("task_ids", [])))[:20]
    route_breakdown = dict(base.get("route_breakdown", {}) or {})
    for route, route_stats in dict(incoming.get("route_breakdown", {}) or {}).items():
        route_breakdown[str(route)] = merge_runtime_stats(route_breakdown.get(str(route), {}), route_stats)
    merged["route_breakdown"] = route_breakdown
    return merged


def render_episode_summary_context(episode: dict, *, max_code_lines: int = 8) -> str:
    task_cue = str(episode.get("task_description", "") or "").strip()
    error_text = str(episode.get("error", "") or "").strip()
    libs = [str(lib).strip() for lib in episode.get("libs", []) or [] if str(lib).strip()]
    code = str(
        episode.get("final_code")
        or episode.get("generated_code")
        or episode.get("script")
        or ""
    ).strip()
    useful_lines = [
        line.rstrip()
        for line in code.splitlines()
        if line.strip() and not line.strip().lower().startswith("thought:")
    ][:max(max_code_lines, 1)]
    likely_fix = "Reuse the previously successful pattern before rewriting core logic."
    if error_text:
        likely_fix = f"Address the prior failure pattern around {infer_error_family(error_text) or 'the observed error'}."
    lines = [
        "Retrieved episodic memory:",
        f"Task cue: {task_cue[:220]}",
    ]
    if error_text:
        lines.append(f"Observed failure clue: {error_text[:220]}")
    lines.append(f"Likely fix pattern: {likely_fix}")
    if libs:
        lines.append(f"Key imports/APIs: {', '.join(libs[:5])}")
    if useful_lines:
        lines.append("Corrected code excerpt:")
        lines.append("```python")
        lines.extend(useful_lines)
        lines.append("```")
    return "\n".join(lines)


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
    target["runtime_stats"] = _normalize_runtime_stats(target.get("runtime_stats", {}))
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

    def apply_runtime_utility(self, utility_by_episode: dict[str, dict], *, route_name: str | None = None) -> int:
        updated = 0
        for episode_id, utility in dict(utility_by_episode or {}).items():
            episode = self._index.get(episode_id)
            if episode is None:
                continue
            utility_stats = _normalize_runtime_stats(utility)
            if route_name:
                route_stats = dict(utility_stats)
                route_stats["route_breakdown"] = {}
                utility_stats = dict(utility_stats)
                utility_stats["route_breakdown"] = {str(route_name): route_stats}
            merged = merge_runtime_stats(episode.get("runtime_stats", {}), utility_stats)
            self.update(episode_id, runtime_stats=merged)
            updated += 1
        return updated

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

"""Package-level helpers extracted from notebook experiment plumbing."""

from __future__ import annotations

import json
from pathlib import Path

from cogmem.benchmarks.bigcodebench.dataset import filter_by_split


def build_eval_cache_path(
    cache_dir: str,
    *,
    version: str,
    seen_tasks: int,
    unseen_tasks: int,
    top_k: int,
    scale: float,
) -> str:
    scale_token = str(scale).replace(".", "p")
    filename = (
        f"eval_cache_{version}_seen{seen_tasks}_unseen{unseen_tasks}_"
        f"top{top_k}_scale{scale_token}.json"
    )
    return str(Path(cache_dir) / filename)


def load_eval_cache(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_eval_cache(path: str, payload: dict) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def materialize_split_views(tasks: list[dict]) -> dict[str, list[dict]]:
    return {
        "train": filter_by_split(tasks, "train"),
        "dev": filter_by_split(tasks, "dev"),
        "test": filter_by_split(tasks, "test"),
    }

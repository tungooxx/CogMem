"""Deterministic split manifests and lineage helpers for BigCodeBench."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable


def infer_task_family(task: dict) -> str:
    """Infer a coarse task family from task metadata."""
    libs = [str(lib).lower() for lib in task.get("libs", []) if lib]
    if any("pandas" in lib or "dataframe" in lib for lib in libs):
        return "dataframe"
    if any(
        marker in lib
        for lib in libs
        for marker in ("matplotlib", "seaborn", "plotly", "bokeh")
    ):
        return "plotting"
    if any(
        marker in lib
        for lib in libs
        for marker in ("socket", "requests", "urllib", "http", "aiohttp")
    ):
        return "networking"
    if any(
        marker in lib
        for lib in libs
        for marker in ("os", "pathlib", "shutil", "glob", "csv", "json")
    ):
        return "file_io"
    prompt = str(task.get("instruct_prompt") or task.get("complete_prompt") or "").lower()
    if "dataframe" in prompt or "pandas" in prompt:
        return "dataframe"
    if any(token in prompt for token in ("plot", "chart", "histogram", "scatter")):
        return "plotting"
    if any(token in prompt for token in ("socket", "request", "url", "api")):
        return "networking"
    if any(token in prompt for token in ("file", "directory", "path", "csv", "json")):
        return "file_io"
    return "general_code"


def _prompt_bucket(task: dict) -> str:
    prompt = str(task.get("instruct_prompt") or task.get("complete_prompt") or "")
    length = len(prompt)
    if length < 300:
        return "short"
    if length < 900:
        return "medium"
    return "long"


def _bucket_key(task: dict) -> tuple[str, str]:
    return infer_task_family(task), _prompt_bucket(task)


def _canonical_manifest_payload(
    task_splits: dict[str, str],
    *,
    source_benchmark: str,
    dataset_version: str,
    label: str,
    seed: int,
    train_fraction: float,
    dev_fraction: float,
) -> dict:
    return {
        "builder": "cogmem.bigcodebench.splits.v1",
        "source_benchmark": source_benchmark,
        "dataset_version": dataset_version,
        "label": label,
        "seed": seed,
        "train_fraction": train_fraction,
        "dev_fraction": dev_fraction,
        "task_splits": dict(sorted(task_splits.items())),
    }


def _manifest_id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def build_split_manifest(
    tasks: list[dict],
    *,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
    seed: int = 42,
    label: str = "bigcodebench_cl",
    source_benchmark: str = "bigcodebench",
    dataset_version: str = "unknown",
) -> dict:
    """Build a deterministic train/dev/test manifest for BigCodeBench tasks."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    if train_fraction + dev_fraction >= 1.0:
        raise ValueError("train_fraction + dev_fraction must be < 1")

    task_splits: dict[str, str] = {}
    by_bucket: dict[tuple[str, str], list[dict]] = {}
    for task in tasks:
        task_id = task.get("task_id")
        if not task_id:
            raise ValueError("Every task must contain task_id")
        by_bucket.setdefault(_bucket_key(task), []).append(task)

    for bucket_key in sorted(by_bucket):
        bucket_tasks = sorted(by_bucket[bucket_key], key=lambda item: item["task_id"])
        bucket_rng = random.Random(f"{seed}:{bucket_key[0]}:{bucket_key[1]}")
        bucket_rng.shuffle(bucket_tasks)

        n_total = len(bucket_tasks)
        n_train = round(n_total * train_fraction)
        n_dev = round(n_total * dev_fraction)
        if n_total >= 3:
            n_train = max(1, min(n_train, n_total - 2))
            n_dev = max(1, min(n_dev, n_total - n_train - 1))
        elif n_total == 2:
            n_train = 1
            n_dev = 0

        for idx, task in enumerate(bucket_tasks):
            if idx < n_train:
                split_name = "train"
            elif idx < n_train + n_dev:
                split_name = "dev"
            else:
                split_name = "test"
            task_splits[task["task_id"]] = split_name

    payload = _canonical_manifest_payload(
        task_splits,
        source_benchmark=source_benchmark,
        dataset_version=dataset_version,
        label=label,
        seed=seed,
        train_fraction=train_fraction,
        dev_fraction=dev_fraction,
    )
    payload["manifest_id"] = _manifest_id(payload)
    return payload


def save_split_manifest(manifest: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_split_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def task_split_name(task_id: str, manifest: dict) -> str | None:
    return manifest.get("task_splits", {}).get(task_id)


def filter_task_ids_by_split(
    task_ids: Iterable[str],
    manifest: dict,
    split_name: str,
) -> list[str]:
    return [
        task_id
        for task_id in task_ids
        if task_split_name(task_id, manifest) == split_name
    ]


def annotate_tasks_with_manifest(
    tasks: list[dict],
    manifest: dict,
    *,
    split_name: str | None = None,
) -> list[dict]:
    """Attach manifest lineage metadata to task dicts."""
    annotated: list[dict] = []
    manifest_id = manifest.get("manifest_id")
    source_benchmark = manifest.get("source_benchmark", "bigcodebench")
    for task in tasks:
        task_id = task.get("task_id")
        current_split = task_split_name(task_id, manifest)
        if split_name is not None and current_split != split_name:
            continue
        enriched = dict(task)
        enriched["split_name"] = current_split
        enriched["manifest_id"] = manifest_id
        enriched["source_benchmark"] = source_benchmark
        annotated.append(enriched)
    return annotated

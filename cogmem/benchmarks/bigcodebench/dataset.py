"""Load and manage BigCodeBench tasks."""

import json
from pathlib import Path

from cogmem.benchmarks.bigcodebench.splits import annotate_tasks_with_manifest


def load_bigcodebench(
    version: str = "v0.1.4",
    hard_only: bool = False,
    split_manifest: dict | None = None,
    split_name: str | None = None,
) -> list[dict]:
    """Load BigCodeBench tasks from HuggingFace datasets.

    BigCodeBench uses the version string as the split name (not a config).

    Args:
        version: Dataset version, used as the split name (e.g., "v0.1.4").
        hard_only: If True, load only BigCodeBench-Hard (148 tasks).
        split_manifest: Optional split manifest to attach lineage metadata.
        split_name: Optional split to filter to when split_manifest is provided.

    Returns:
        List of task dicts with keys: task_id, instruct_prompt, complete_prompt,
        test, canonical_solution, entry_point.
    """
    from datasets import load_dataset

    repo = "bigcode/bigcodebench-hard" if hard_only else "bigcode/bigcodebench"
    ds = load_dataset(repo, split=version)

    tasks = []
    for item in ds:
        tasks.append({
            "task_id": item["task_id"],
            "instruct_prompt": item.get("instruct_prompt", ""),
            "complete_prompt": item.get("complete_prompt", ""),
            "test": item.get("test", ""),
            "canonical_solution": item.get("canonical_solution", ""),
            "entry_point": item.get("entry_point", ""),
            "libs": item.get("libs", []),
        })

    if split_manifest is not None:
        tasks = annotate_tasks_with_manifest(tasks, split_manifest, split_name=split_name)

    return tasks


def load_bigcodebench_from_jsonl(
    path: str,
    split_manifest: dict | None = None,
    split_name: str | None = None,
) -> list[dict]:
    """Load BigCodeBench tasks from a local JSONL file (for offline use)."""
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    if split_manifest is not None:
        tasks = annotate_tasks_with_manifest(tasks, split_manifest, split_name=split_name)
    return tasks


def save_tasks_jsonl(tasks: list[dict], path: str) -> None:
    """Save tasks to JSONL for offline use."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


def get_task_ids(tasks: list[dict]) -> list[str]:
    """Get sorted list of task IDs."""
    return sorted(t["task_id"] for t in tasks)


def filter_by_ids(tasks: list[dict], task_ids: set[str]) -> list[dict]:
    """Filter tasks to only those in task_ids set."""
    return [t for t in tasks if t["task_id"] in task_ids]


def filter_by_split(tasks: list[dict], split_name: str) -> list[dict]:
    """Filter tasks to a previously attached split label."""
    return [task for task in tasks if task.get("split_name") == split_name]

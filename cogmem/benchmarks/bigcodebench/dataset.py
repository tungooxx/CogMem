"""Load and manage BigCodeBench tasks."""

import json
from pathlib import Path


def load_bigcodebench(
    subset: str = "v0.1.2",
    split: str = "full",
    hard_only: bool = False,
) -> list[dict]:
    """Load BigCodeBench tasks from HuggingFace datasets.

    Args:
        subset: Dataset version/subset.
        split: Which split to load.
        hard_only: If True, load only BigCodeBench-Hard (148 tasks).

    Returns:
        List of task dicts with keys: task_id, instruct_prompt, complete_prompt,
        test, canonical_solution, entry_point.
    """
    from datasets import load_dataset

    if hard_only:
        ds = load_dataset("bigcode/bigcodebench-hard", subset, split=split)
    else:
        ds = load_dataset("bigcode/bigcodebench", subset, split=split)

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

    return tasks


def load_bigcodebench_from_jsonl(path: str) -> list[dict]:
    """Load BigCodeBench tasks from a local JSONL file (for offline use)."""
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
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

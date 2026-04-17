from cogmem.benchmarks.bigcodebench.splits import (
    annotate_tasks_with_manifest,
    build_split_manifest,
    filter_task_ids_by_split,
    infer_task_family,
)


def test_infer_task_family_uses_libs_first():
    task = {
        "task_id": "BigCodeBench/1",
        "instruct_prompt": "Do something with a dataframe",
        "libs": ["pandas"],
    }
    assert infer_task_family(task) == "dataframe"


def test_build_split_manifest_is_deterministic():
    tasks = [
        {"task_id": f"BigCodeBench/{idx}", "instruct_prompt": f"Task {idx}", "libs": []}
        for idx in range(12)
    ]

    manifest_a = build_split_manifest(tasks, seed=7)
    manifest_b = build_split_manifest(tasks, seed=7)

    assert manifest_a["manifest_id"] == manifest_b["manifest_id"]
    assert manifest_a["task_splits"] == manifest_b["task_splits"]


def test_annotate_tasks_with_manifest_filters_split():
    tasks = [
        {"task_id": "BigCodeBench/1", "instruct_prompt": "Plot a histogram", "libs": ["matplotlib"]},
        {"task_id": "BigCodeBench/2", "instruct_prompt": "Open a socket", "libs": ["socket"]},
        {"task_id": "BigCodeBench/3", "instruct_prompt": "Read a csv", "libs": ["csv"]},
        {"task_id": "BigCodeBench/4", "instruct_prompt": "Transform a dataframe", "libs": ["pandas"]},
        {"task_id": "BigCodeBench/5", "instruct_prompt": "General task", "libs": []},
    ]
    manifest = build_split_manifest(tasks, seed=11)

    train_tasks = annotate_tasks_with_manifest(tasks, manifest, split_name="train")

    assert train_tasks
    assert all(task["split_name"] == "train" for task in train_tasks)
    assert all(task["manifest_id"] == manifest["manifest_id"] for task in train_tasks)


def test_filter_task_ids_by_split():
    tasks = [
        {"task_id": f"BigCodeBench/{idx}", "instruct_prompt": f"Task {idx}", "libs": []}
        for idx in range(10)
    ]
    manifest = build_split_manifest(tasks, seed=3)

    train_ids = filter_task_ids_by_split([task["task_id"] for task in tasks], manifest, "train")

    assert train_ids
    assert all(manifest["task_splits"][task_id] == "train" for task_id in train_ids)

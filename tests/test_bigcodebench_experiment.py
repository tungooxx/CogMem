from cogmem.benchmarks.bigcodebench.experiment import (
    build_eval_cache_path,
    load_eval_cache,
    materialize_split_views,
    save_eval_cache,
)


def test_build_eval_cache_path_formats_scale_token(tmp_path):
    path = build_eval_cache_path(
        str(tmp_path),
        version="finaluse_v4",
        seen_tasks=500,
        unseen_tasks=50,
        top_k=1,
        scale=0.25,
    )

    assert path.endswith("eval_cache_finaluse_v4_seen500_unseen50_top1_scale0p25.json")


def test_save_and_load_eval_cache(tmp_path):
    path = tmp_path / "cache.json"
    payload = {"seen": {"pass_rate": 0.2}}
    save_eval_cache(str(path), payload)

    loaded = load_eval_cache(str(path))

    assert loaded == payload


def test_materialize_split_views():
    tasks = [
        {"task_id": "a", "split_name": "train"},
        {"task_id": "b", "split_name": "dev"},
        {"task_id": "c", "split_name": "test"},
    ]

    views = materialize_split_views(tasks)

    assert [task["task_id"] for task in views["train"]] == ["a"]
    assert [task["task_id"] for task in views["dev"]] == ["b"]
    assert [task["task_id"] for task in views["test"]] == ["c"]

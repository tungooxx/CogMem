import json

from cogmem.memory.episodic_store import (
    EpisodicStore,
    infer_error_family,
    normalize_episode_record,
)


def test_infer_error_family_handles_common_patterns():
    assert infer_error_family("ValueError: bad input") == "ValueError"
    assert infer_error_family("timed out after 30s") == "Timeout"
    assert infer_error_family("assert x == y") == "AssertionError"
    assert infer_error_family(None) is None


def test_normalize_episode_record_adds_typed_fields():
    episode = {
        "episode_id": "ep_001",
        "task_id": "BigCodeBench/1",
        "task_type": "bigcodebench",
        "task_description": "Write a parser",
        "success": False,
        "error": "ValueError: invalid literal",
        "retrieved_from": ["ep_000"],
        "entry_point": "task_func",
    }

    normalized = normalize_episode_record(episode, copy_episode=True)

    assert normalized["prompt_hash"] is not None
    assert len(normalized["prompt_hash"]) == 64
    assert normalized["retrieved_ids"] == ["ep_000"]
    assert normalized["adapter_ids"] == []
    assert normalized["error_family"] == "ValueError"
    assert normalized["validation_recipe"]["kind"] == "bigcodebench_exec"
    assert normalized["validation_recipe"]["entry_point"] == "task_func"
    assert normalized["source_benchmark"] == "unknown"
    assert normalized["episode_helpfulness"] == 0.0
    assert normalized["q_value"] == 0.0


def test_store_roundtrip_and_filtering(tmp_path):
    episodes = [
        normalize_episode_record(
            {
                "episode_id": "ep_train",
                "task_id": "t1",
                "task_type": "bigcodebench",
                "task_description": "Train task",
                "success": True,
                "split_name": "train",
                "manifest_id": "manifest_a",
                "libs": ["pandas"],
                "source_benchmark": "bigcodebench",
                "error": None,
            },
            copy_episode=True,
        ),
        normalize_episode_record(
            {
                "episode_id": "ep_dev",
                "task_id": "t2",
                "task_type": "bigcodebench",
                "task_description": "Dev task",
                "success": False,
                "split_name": "dev",
                "manifest_id": "manifest_b",
                "libs": ["numpy"],
                "source_benchmark": "bigcodebench",
                "error": "SyntaxError: invalid syntax",
            },
            copy_episode=True,
        ),
    ]

    json_path = tmp_path / "episodes.json"
    jsonl_path = tmp_path / "episodes.jsonl"

    store = EpisodicStore(episodes)
    store.save(str(json_path))
    store.save_jsonl(str(jsonl_path))
    EpisodicStore.append_jsonl(
        str(jsonl_path),
        {
            "episode_id": "ep_eval",
            "task_id": "t3",
            "task_type": "bigcodebench",
            "task_description": "Eval task",
            "success": True,
            "split_name": "test",
            "manifest_id": "manifest_c",
            "source_benchmark": "bigcodebench",
        },
    )

    loaded = EpisodicStore.load(str(json_path))
    loaded_jsonl = EpisodicStore.load_jsonl(str(jsonl_path))

    assert len(loaded) == 2
    assert len(loaded_jsonl) == 3
    assert loaded.filter(split_name="train")[0]["episode_id"] == "ep_train"
    assert loaded.filter(success=False)[0]["episode_id"] == "ep_dev"
    assert loaded.filter(manifest_id="manifest_a")[0]["episode_id"] == "ep_train"
    assert loaded.filter(source_benchmark="bigcodebench")[0]["episode_id"] == "ep_train"
    assert loaded.filter(library="pandas")[0]["episode_id"] == "ep_train"
    assert loaded.filter(error_family="SyntaxError")[0]["episode_id"] == "ep_dev"

    raw_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw_lines) == 3
    assert json.loads(raw_lines[-1])["episode_id"] == "ep_eval"

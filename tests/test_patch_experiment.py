import json
from types import SimpleNamespace

from cogmem.benchmarks.bigcodebench.experiment import save_eval_cache
from cogmem.patches.experiment import (
    PatchExperimentConfig,
    evaluate_patch_memory_bank,
    inspect_unseen_retrieval,
    prepare_patch_task_split,
    run_patch_episode_recording,
)


class DummyMemoryBank:
    def __init__(self):
        self.episodes = []
        self.memories = []
        self.artifact_bank = SimpleNamespace(patches=[])
        self.saved = 0
        self.sleep_calls = 0

    def load(self):
        return None

    def save(self):
        self.saved += 1

    def stats(self):
        return {"episodes": len(self.episodes), "memories": len(self.memories)}

    def record_episode(self, **kwargs):
        self.episodes.append(kwargs)

    def run_sleep_cycle(self, prune=True):
        self.sleep_calls += 1
        return self.stats()


def test_prepare_patch_task_split():
    tasks = [{"task_id": f"task_{idx}"} for idx in range(5)]

    train_tasks, eval_tasks = prepare_patch_task_split(tasks, 3)

    assert [task["task_id"] for task in train_tasks] == ["task_0", "task_1", "task_2"]
    assert [task["task_id"] for task in eval_tasks] == ["task_3", "task_4"]


def test_run_patch_episode_recording_resumes_from_progress(tmp_path, monkeypatch):
    config = PatchExperimentConfig(memory_dir=str(tmp_path / "cluster_memories"), n_candidates=2)
    bank = DummyMemoryBank()
    tasks = [
        {"task_id": "task_1", "instruct_prompt": "first task"},
        {"task_id": "task_2", "instruct_prompt": "second task"},
    ]

    progress_path = tmp_path / "cluster_memories" / config.episode_progress_filename
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps({
        "status": "running",
        "train_signature": {
            "count": 2,
            "first_task_id": "task_1",
            "last_task_id": "task_2",
        },
        "next_index": 1,
        "total_passed": 1,
        "episodes_before": 0,
        "episodes_now": 0,
    }))

    call_count = {"generate": 0}

    class DummyEmbedder:
        def encode(self, prompt):
            return [0.1, 0.2, len(prompt)]

    def fake_generate_many(*args, **kwargs):
        call_count["generate"] += 1
        return [
            "pass candidate with enough code length",
            "fail candidate with enough code length",
        ]

    monkeypatch.setattr("cogmem.patches.experiment._generate_many", fake_generate_many)
    monkeypatch.setattr(
        "cogmem.patches.experiment._best_contrast_pair",
        lambda passes, fails: ({"pass": passes[0], "fail": fails[0]}, 0.8),
    )
    monkeypatch.setattr("cogmem.patches.experiment.extract_code", lambda response: response)
    monkeypatch.setattr(
        "cogmem.patches.experiment.evaluate_solution",
        lambda task, code, timeout, mode: {"passed": code.startswith("pass")},
    )

    result = run_patch_episode_recording(
        tasks,
        base_model=object(),
        tokenizer=object(),
        embedder=DummyEmbedder(),
        config=config,
        memory_bank=bank,
    )

    assert call_count["generate"] == 1
    assert result["tasks_with_passes"] == 2
    assert result["new_episodes"] == 1
    assert len(bank.episodes) == 1
    saved = json.loads(progress_path.read_text())
    assert saved["status"] == "complete"
    assert saved["next_index"] == 2


def test_inspect_unseen_retrieval_summarizes_selection(monkeypatch):
    config = PatchExperimentConfig(inspect_unseen_size=2)
    bank = DummyMemoryBank()
    bank.memories = [
        SimpleNamespace(
            memory_id="memory_plot",
            family_label="plotting",
            retrievable=True,
            retrieval_threshold=0.5,
            reuse_count=0,
        ),
        SimpleNamespace(
            memory_id="memory_file",
            family_label="file_io",
            retrievable=True,
            retrieval_threshold=0.5,
            reuse_count=0,
        ),
    ]

    class DummyEmbedder:
        def encode(self, prompt):
            return [1.0, 0.0] if "plot" in prompt else [0.0, 1.0]

    def fake_applicability(memory, query, prompt):
        if memory.memory_id == "memory_plot":
            return 0.8 if "plot" in prompt else 0.2
        return 0.4

    monkeypatch.setattr("cogmem.patches.experiment._compute_applicability", fake_applicability)
    monkeypatch.setattr("cogmem.patches.experiment._score_memory_use", lambda *args, **kwargs: 0.3)
    monkeypatch.setattr("cogmem.patches.experiment._score_memory_final_use", lambda *args, **kwargs: 0.42)

    result = inspect_unseen_retrieval(
        [
            {"task_id": "task_plot", "instruct_prompt": "plot a chart"},
            {"task_id": "task_other", "instruct_prompt": "parse a file"},
        ],
        bank,
        DummyEmbedder(),
        config=config,
    )

    assert result["abstentions"] == 1
    assert result["memory_hits"] == {"memory_plot": 1}
    assert result["family_hits"] == {"plotting": 1}
    assert result["rows"][0]["selected"] is True
    assert result["rows"][1]["selected"] is False


def test_evaluate_patch_memory_bank_uses_cache(tmp_path, monkeypatch):
    config = PatchExperimentConfig(
        memory_dir=str(tmp_path / "cluster_memories"),
        eval_cache_version="runner_v1",
        eval_top_k=1,
        eval_scale=0.25,
        unseen_eval_size=1,
    )
    bank = DummyMemoryBank()
    cache_path = tmp_path / "cluster_memories" / "eval_cache_runner_v1_seen2_unseen1_top1_scale0p25.json"
    save_eval_cache(str(cache_path), {
        "seen_eval": {"label": "SEEN", "cold_passed": 1, "memory_passed": 1},
        "unseen_eval": {"label": "UNSEEN", "cold_passed": 0, "memory_passed": 0},
    })

    monkeypatch.setattr(
        "cogmem.patches.experiment._run_eval_split",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache should be used")),
    )

    result = evaluate_patch_memory_bank(
        [{"task_id": "train_1"}, {"task_id": "train_2"}],
        [{"task_id": "eval_1"}],
        base_model=object(),
        tokenizer=object(),
        memory_bank=bank,
        embedder=object(),
        config=config,
        force_rerun=False,
    )

    assert result["used_cache"] is True
    assert result["cache_path"].endswith("eval_cache_runner_v1_seen2_unseen1_top1_scale0p25.json")
    assert result["seen_eval"]["label"] == "SEEN"
    assert bank.sleep_calls == 1

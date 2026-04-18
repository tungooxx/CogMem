from types import SimpleNamespace

from cogmem.consolidation.experiment import compare_new_arch_base_vs_adapter


def test_compare_new_arch_base_vs_adapter_reports_improvements(monkeypatch):
    tasks = [
        {"task_id": "task_1"},
        {"task_id": "task_2"},
        {"task_id": "task_3"},
    ]

    def fake_load_new_arch_runtime(*, model_name, adapter_path=None):
        mode = "adapter" if adapter_path else "base"
        return object(), object(), SimpleNamespace(mode=mode)

    def fake_run_single_task(task, llm_client, **kwargs):
        successes = {
            "base": {"task_1", "task_2"},
            "adapter": {"task_2", "task_3"},
        }
        passed = task["task_id"] in successes[llm_client.mode]
        return {
            "task_id": task["task_id"],
            "success": passed,
            "num_attempts": 1,
            "error": None if passed else "failed",
        }

    monkeypatch.setattr(
        "cogmem.consolidation.experiment.load_new_arch_runtime",
        fake_load_new_arch_runtime,
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment.run_single_task",
        fake_run_single_task,
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment._release_new_arch_runtime",
        lambda *args, **kwargs: None,
    )

    result = compare_new_arch_base_vs_adapter(
        tasks,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        adapter_path="/tmp/adapter",
    )

    assert result["base"]["passed"] == 2
    assert result["adapter"]["passed"] == 2
    assert result["delta_passed"] == 0
    assert result["improved_task_ids"] == ["task_3"]
    assert result["regressed_task_ids"] == ["task_1"]

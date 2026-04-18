from types import SimpleNamespace

from cogmem.consolidation.experiment import (
    _run_task_with_skill_cards,
    compare_new_arch_base_vs_adapter,
    compare_new_arch_routes,
    persist_route_skill_utility,
    run_new_arch_qstar_cycle,
)
from cogmem.config import CogMemConfig
from cogmem.memory.skill_store import SkillStore


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


def test_compare_new_arch_routes_includes_skill_retrieval(monkeypatch):
    tasks = [
        {"task_id": "task_1", "instruct_prompt": "plot data", "libs": ["matplotlib"]},
        {"task_id": "task_2", "instruct_prompt": "plot lines", "libs": ["matplotlib"]},
    ]

    def fake_load_new_arch_runtime(*, model_name, adapter_path=None):
        mode = "adapter" if adapter_path else "base"
        return object(), object(), SimpleNamespace(mode=mode)

    def fake_run_single_task(task, llm_client, **kwargs):
        return {
            "task_id": task["task_id"],
            "success": llm_client.mode == "base",
            "num_attempts": 1,
            "error": None,
        }

    def fake_run_task_with_skill_cards(task, llm_client, *, retrieved_skill_cards, **kwargs):
        return {
            "task_id": task["task_id"],
            "success": True,
            "num_attempts": 1,
            "error": None,
            "retrieved_skill_ids": [card["skill_id"] for card in retrieved_skill_cards],
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
        "cogmem.consolidation.experiment._run_task_with_skill_cards",
        fake_run_task_with_skill_cards,
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment._release_new_arch_runtime",
        lambda *args, **kwargs: None,
    )

    class DummySkillStore:
        def __iter__(self):
            return iter([])

    monkeypatch.setattr(
        "cogmem.consolidation.experiment.SkillStore.load",
        lambda path: DummySkillStore(),
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment.rank_skill_cards_for_task",
        lambda store, task, **kwargs: [{"skill_id": "skill_plot", "status": "promoted"}],
    )

    result = compare_new_arch_routes(
        tasks,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        adapter_path="/tmp/adapter",
        skill_cards_path="/tmp/skills.json",
    )

    assert result["base"]["passed"] == 2
    assert result["base_plus_skill"]["passed"] == 2
    assert result["adapter"]["passed"] == 0
    assert result["adapter_plus_skill"]["passed"] == 2
    assert result["base_plus_skill"]["selected_skill_ids"] == {"skill_plot": 2}
    assert result["comparisons"]["base_plus_skill_vs_base"]["delta_passed"] == 0
    assert result["comparisons"]["adapter_plus_skill_vs_adapter"]["delta_passed"] == 2
    assert result["comparisons"]["base_plus_skill_vs_base"]["skill_utility"]["skill_plot"]["retrieved"] == 2


def test_run_new_arch_qstar_cycle_skips_when_base_plus_skill_route_loses(tmp_path, monkeypatch):
    bank_path = tmp_path / "memory_bank.json"
    bank_path.write_text("[]", encoding="utf-8")
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        '[{"skill_id":"skill_plot","task_type":"matplotlib:plot","domain":"matplotlib","status":"promoted"}]',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "cogmem.consolidation.experiment.MemoryBank.load",
        lambda path: [],
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment.filter_manifest_eligible",
        lambda episodes, config: [],
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment.prepare_skill_training_dataset",
        lambda *args, **kwargs: [{"instruction": "x", "output": "y"}],
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment.prepare_preference_dataset",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment.compare_new_arch_routes",
        lambda *args, **kwargs: {
            "comparisons": {
                "base_plus_skill_vs_base": {
                    "delta_passed": 0,
                    "regression_rate": 0.2,
                    "regressed_task_ids": ["task_1"],
                    "skill_utility": {},
                }
            }
        },
    )

    result = run_new_arch_qstar_cycle(
        str(bank_path),
        CogMemConfig(
            memory_bank_path=str(bank_path),
            active_model_hf="Qwen/Qwen2.5-3B-Instruct",
            skill_route_gate_min_delta_passed=1,
            skill_route_gate_max_regression_rate=0.10,
        ),
        cycle=0,
        skill_cards_path=str(skills_path),
        eval_tasks=[{"task_id": "task_1", "instruct_prompt": "plot data", "libs": ["matplotlib"]}],
    )

    assert result["status"] == "skipped_route_gate"
    assert result["route_gate"]["regressed_task_ids"] == ["task_1"]


def test_run_task_with_skill_cards_reranks_after_failure(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            return "```python\nprint('ok')\n```"

    task = {"task_id": "task_plot", "instruct_prompt": "plot data", "libs": ["matplotlib"]}
    attempts = iter(
        [
            {"passed": False, "error": "AssertionError: chart mismatch"},
            {"passed": True, "error": None},
        ]
    )

    monkeypatch.setattr(
        "cogmem.consolidation.experiment.format_messages",
        lambda task, use_instruct=True: [
            {"role": "system", "content": "system"},
            {"role": "user", "content": task["instruct_prompt"]},
        ],
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment.extract_code",
        lambda response, task: "print('ok')",
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment.evaluate_solution",
        lambda task, code, **kwargs: next(attempts),
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment.rank_skill_cards_for_record",
        lambda store, record, **kwargs: (
            [{"skill_id": "skill_fix_assertion", "status": "promoted"}]
            if "AssertionError" in str(record.get("error") or "")
            else [{"skill_id": "skill_plot", "status": "promoted"}]
        ),
    )

    result = _run_task_with_skill_cards(
        task,
        DummyClient(),
        retrieved_skill_cards=[{"skill_id": "skill_plot", "status": "promoted"}],
        skill_store=object(),
        skill_top_k=1,
        max_attempts=2,
    )

    assert result["success"] is True
    assert result["retrieved_skill_history"] == [["skill_plot"], ["skill_fix_assertion"]]
    assert result["retrieved_skill_ids"] == ["skill_plot", "skill_fix_assertion"]


def test_persist_route_skill_utility_updates_skill_store(tmp_path):
    skills_path = tmp_path / "skills.json"
    SkillStore(
        [
            {
                "skill_id": "skill_plot",
                "task_type": "matplotlib:plot",
                "domain": "matplotlib",
                "status": "promoted",
            }
        ]
    ).save(str(skills_path))

    persist_result = persist_route_skill_utility(
        str(skills_path),
        {
            "base_plus_skill_vs_base": {
                "skill_utility": {
                    "skill_plot": {
                        "retrieved": 2,
                        "helped": 1,
                        "hurt": 0,
                        "passed": 1,
                        "failed": 1,
                        "domains": {"matplotlib": 2},
                        "error_families": {"AssertionError": 1},
                        "task_ids": ["task_1", "task_2"],
                    }
                }
            }
        },
    )

    updated = SkillStore.load(str(skills_path)).get("skill_plot")
    assert persist_result["updated_routes"] == ["base_plus_skill_vs_base"]
    assert updated["runtime_stats"]["retrieved"] == 2
    assert updated["runtime_stats"]["helped"] == 1
    assert updated["runtime_stats"]["domains"]["matplotlib"] == 2
    assert updated["runtime_stats"]["route_breakdown"]["base_plus_skill_vs_base"]["retrieved"] == 2

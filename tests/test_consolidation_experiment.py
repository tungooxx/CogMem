import json
from types import SimpleNamespace

from cogmem.consolidation.experiment import (
    _run_task_with_memory_route,
    _run_task_with_skill_cards,
    build_new_arch_skill_cards,
    compare_new_arch_base_vs_adapter,
    compare_new_arch_routes,
    persist_route_memory_utility,
    persist_route_skill_utility,
    run_new_arch_qstar_cycle,
    select_runtime_route,
)
from cogmem.config import CogMemConfig
from cogmem.memory.memory_bank import MemoryBank
from cogmem.memory.skill_store import SkillStore


def test_compare_new_arch_base_vs_adapter_reports_improvements(monkeypatch):
    tasks = [{"task_id": "task_1"}, {"task_id": "task_2"}, {"task_id": "task_3"}]

    monkeypatch.setattr(
        "cogmem.consolidation.experiment.compare_new_arch_routes",
        lambda *args, **kwargs: {
            "task_count": 3,
            "base": {"passed": 2, "pass_rate": 2 / 3},
            "adapter": {"passed": 2, "pass_rate": 2 / 3},
            "improved_task_ids": ["task_3"],
            "regressed_task_ids": ["task_1"],
        },
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

    monkeypatch.setattr(
        "cogmem.consolidation.experiment.load_new_arch_runtime",
        lambda *, model_name, adapter_path=None: (
            object(),
            object(),
            SimpleNamespace(mode="adapter" if adapter_path else "base"),
        ),
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment._release_new_arch_runtime",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment._run_task_with_memory_route",
        lambda task, llm_client, *, route_mode="none", **kwargs: {
            "task_id": task["task_id"],
            "success": (
                llm_client.mode == "base"
                if route_mode == "none"
                else True
            ),
            "num_attempts": 1,
            "error": None,
            "selected_route": {
                "none": "none",
                "skill": "skill",
                "episode": "episode_summary",
                "router": "skill",
            }[route_mode],
            "abstained": False,
            "retrieved_skill_ids": ["skill_plot"] if route_mode in {"skill", "router"} else [],
            "retrieved_episode_id": "episode_plot" if route_mode in {"episode", "router"} else None,
            "retrieved_skill_history": [["skill_plot"]] if route_mode in {"skill", "router"} else [[]],
            "retrieved_route_history": [{"selected_route": route_mode}],
        },
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
        episode_memory_path="/tmp/memory_bank.json",
    )

    assert result["base"]["passed"] == 2
    assert result["base_plus_skill"]["passed"] == 2
    assert result["base_plus_episode"]["passed"] == 2
    assert result["base_plus_router"]["passed"] == 2
    assert result["adapter"]["passed"] == 0
    assert result["adapter_plus_router"]["passed"] == 2
    assert result["base_plus_skill"]["selected_skill_ids"] == {"skill_plot": 2}
    assert result["comparisons"]["base_plus_skill_vs_base"]["delta_passed"] == 0
    assert result["comparisons"]["base_plus_episode_vs_base"]["episode_utility"]["episode_plot"]["retrieved"] == 2
    assert result["comparisons"]["base_plus_router_vs_base"]["delta_passed"] == 0
    assert result["comparisons"]["adapter_plus_router_vs_adapter"]["delta_passed"] == 2
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
                "base_plus_router_vs_base": {
                    "delta_passed": 0,
                    "regression_rate": 0.2,
                    "regressed_task_ids": ["task_1"],
                    "skill_utility": {},
                    "episode_utility": {},
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
        "cogmem.consolidation.experiment.score_skill_cards_for_record",
        lambda store, record, **kwargs: (
            [(10.0, {"skill_id": "skill_fix_assertion", "status": "promoted"})]
            if "AssertionError" in str(record.get("error") or "")
            else [(10.0, {"skill_id": "skill_plot", "status": "promoted"})]
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


def test_router_chooses_episode_summary_on_retry_when_skill_is_weak(monkeypatch):
    class DummyClient:
        def chat(self, messages, **kwargs):
            return "```python\nprint('ok')\n```"

    task = {"task_id": "task_retry", "instruct_prompt": "debug a weird plotting failure", "libs": ["matplotlib"]}
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
        "cogmem.consolidation.experiment.score_skill_cards_for_record",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment._score_episode_summaries_for_record",
        lambda store, record, **kwargs: (
            [(9.0, {"episode_id": "episode_fix", "task_description": "debugged chart", "final_code": "print('ok')", "success": True})]
            if "AssertionError" in str(record.get("error") or "")
            else []
        ),
    )

    result = _run_task_with_memory_route(
        task,
        DummyClient(),
        route_mode="router",
        episode_store=MemoryBank(
            [
                {
                    "episode_id": "episode_fix",
                    "task_id": "old_task",
                    "task_description": "debugged chart",
                    "task_type": "bigcodebench",
                    "success": True,
                    "script": "print('ok')",
                    "generated_code": "print('ok')",
                    "final_code": "print('ok')",
                    "manifest_id": "manifest_a",
                    "source_benchmark": "bigcodebench",
                    "episode_helpfulness": 0.9,
                    "q_value": 0.9,
                }
            ]
        ),
        max_attempts=2,
    )

    assert result["success"] is True
    assert [item["selected_route"] for item in result["retrieved_route_history"]] == ["none", "episode_summary"]
    assert result["retrieved_episode_id"] == "episode_fix"


def test_router_blocks_episode_summary_on_first_attempt_by_default(monkeypatch):
    task = {"task_id": "task_first", "instruct_prompt": "debug a rare matplotlib failure", "libs": ["matplotlib"]}
    episode_store = MemoryBank(
        [
            {
                "episode_id": "episode_rare",
                "task_id": "old_task",
                "task_description": "debugged rare chart",
                "task_type": "bigcodebench",
                "success": True,
                "script": "print('ok')",
                "generated_code": "print('ok')",
                "final_code": "print('ok')",
                "manifest_id": "manifest_a",
                "source_benchmark": "bigcodebench",
                "episode_helpfulness": 1.0,
                "q_value": 1.0,
            }
        ]
    )

    monkeypatch.setattr(
        "cogmem.consolidation.experiment._score_episode_summaries_for_record",
        lambda *args, **kwargs: [(99.0, episode_store.get("episode_rare"))],
    )

    decision = select_runtime_route(
        task,
        route_mode="router",
        episode_store=episode_store,
        config=CogMemConfig(episode_retrieval_allow_first_attempt=False),
    )

    assert decision["selected_route"] == "none"


def test_persist_route_memory_utility_updates_skill_and_episode_stores(tmp_path):
    skills_path = tmp_path / "skills.json"
    memory_bank_path = tmp_path / "memory_bank.json"
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
    MemoryBank(
        [
            {
                "episode_id": "episode_plot",
                "task_id": "task_1",
                "task_description": "plot a chart",
                "task_type": "bigcodebench",
                "success": True,
                "script": "import matplotlib.pyplot as plt",
                "generated_code": "import matplotlib.pyplot as plt",
                "final_code": "import matplotlib.pyplot as plt",
                "manifest_id": "manifest_a",
                "source_benchmark": "bigcodebench",
                "episode_helpfulness": 1.0,
                "q_value": 1.0,
            }
        ]
    ).save(str(memory_bank_path))

    persist_result = persist_route_memory_utility(
        str(skills_path),
        str(memory_bank_path),
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
                },
                "episode_utility": {
                    "episode_plot": {
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
    updated_episode = MemoryBank.load(str(memory_bank_path)).get("episode_plot")
    assert persist_result["updated_routes"] == ["base_plus_skill_vs_base"]
    assert updated["runtime_stats"]["retrieved"] == 2
    assert updated["runtime_stats"]["helped"] == 1
    assert updated["runtime_stats"]["domains"]["matplotlib"] == 2
    assert updated["runtime_stats"]["route_breakdown"]["base_plus_skill_vs_base"]["retrieved"] == 2
    assert updated_episode["runtime_stats"]["retrieved"] == 2
    assert updated_episode["runtime_stats"]["helped"] == 1
    assert updated_episode["runtime_stats"]["route_breakdown"]["base_plus_skill_vs_base"]["retrieved"] == 2


def test_build_new_arch_skill_cards_requires_route_win_for_promotion(tmp_path, monkeypatch):
    bank_path = tmp_path / "memory_bank.json"
    MemoryBank(
        [
            {
                "episode_id": "ep_1",
                "task_id": "task_1",
                "task_description": "plot data with matplotlib",
                "task_type": "bigcodebench",
                "success": True,
                "script": "import matplotlib.pyplot as plt",
                "generated_code": "import matplotlib.pyplot as plt",
                "final_code": "import matplotlib.pyplot as plt",
                "manifest_id": "manifest_a",
                "source_benchmark": "bigcodebench",
                "episode_helpfulness": 0.9,
                "q_value": 0.9,
                "libs": ["matplotlib"],
            },
            {
                "episode_id": "ep_2",
                "task_id": "task_2",
                "task_description": "plot series with matplotlib",
                "task_type": "bigcodebench",
                "success": True,
                "script": "import matplotlib.pyplot as plt",
                "generated_code": "import matplotlib.pyplot as plt",
                "final_code": "import matplotlib.pyplot as plt",
                "manifest_id": "manifest_a",
                "source_benchmark": "bigcodebench",
                "episode_helpfulness": 0.85,
                "q_value": 0.85,
                "libs": ["matplotlib"],
            },
            {
                "episode_id": "ep_3",
                "task_id": "task_3",
                "task_description": "plot chart with matplotlib",
                "task_type": "bigcodebench",
                "success": True,
                "script": "import matplotlib.pyplot as plt",
                "generated_code": "import matplotlib.pyplot as plt",
                "final_code": "import matplotlib.pyplot as plt",
                "manifest_id": "manifest_a",
                "source_benchmark": "bigcodebench",
                "episode_helpfulness": 0.8,
                "q_value": 0.8,
                "libs": ["matplotlib"],
            },
        ]
    ).save(str(bank_path))

    monkeypatch.setattr(
        "cogmem.consolidation.experiment.build_skill_cards",
        lambda *args, **kwargs: SkillStore(
            [
                {
                    "skill_id": "skill_plot",
                    "task_type": "matplotlib:plot",
                    "domain": "matplotlib",
                    "status": "validated",
                    "validation": {"validated_task_ids": ["task_2", "task_3"]},
                    "manifest_ids": ["manifest_a"],
                    "evidence_episode_ids": ["ep_1"],
                    "evidence_task_ids": ["task_1"],
                    "distinct_task_count": 1,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "cogmem.consolidation.experiment.compare_new_arch_routes",
        lambda *args, **kwargs: {
            "comparisons": {
                "base_plus_skill_vs_base": {
                    "delta_passed": 1,
                    "delta_pass_rate": 0.5,
                    "regression_rate": 0.0,
                    "improved_task_ids": ["task_2"],
                    "regressed_task_ids": [],
                }
            }
        },
    )

    result = build_new_arch_skill_cards(
        str(bank_path),
        CogMemConfig(
            memory_bank_path=str(bank_path),
            min_holdout=2,
            skill_min_evidence=1,
            skill_validation_min_matches=1,
            skill_min_transfer_gain=0.0,
            skill_confidence_threshold=0.0,
            skill_route_promotion_min_tasks=1,
            skill_route_promotion_min_delta_passed=1,
            skill_route_promotion_max_regression_rate=0.1,
        ),
        skills_path=str(tmp_path / "skills.json"),
    )

    assert result["skill_summary"]["validated"] >= 0
    assert result["skill_summary"]["promoted"] == 1

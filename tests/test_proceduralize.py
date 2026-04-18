from cogmem.config import CogMemConfig
from cogmem.consolidation.proceduralize import (
    build_skill_cards,
    proceduralize_episodes,
    rank_skill_cards_for_task,
    validate_skill_card,
)
from cogmem.memory.memory_bank import MemoryBank


def _episode(
    episode_id: str,
    task_id: str,
    task_description: str,
    *,
    success: bool,
    script: str,
    q_value: float,
    task_type: str = "file_io",
    manifest_id: str = "manifest_a",
    error: str | None = None,
    entry_point: str = "",
    libs: list[str] | None = None,
) -> dict:
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "task_type": task_type,
        "task_description": task_description,
        "script": script,
        "generated_code": script,
        "final_code": script if success else None,
        "success": success,
        "q_value": q_value,
        "episode_helpfulness": q_value,
        "intent_embedding": [0.1, 0.2, 0.3],
        "validation_recipe": {"kind": "bigcodebench_exec", "task_id": task_id},
        "manifest_id": manifest_id,
        "source_benchmark": "bigcodebench",
        "error": error,
        "entry_point": entry_point,
        "libs": libs or [],
    }


def test_proceduralize_episodes_builds_candidate_cards():
    episodes = [
        _episode("ep_1", "t1", "Use glob to list files in a directory", success=True, script="1. import glob\n2. list files", q_value=0.9),
        _episode("ep_2", "t2", "Read all files from a directory with glob", success=True, script="1. import glob\n2. open files", q_value=0.8),
        _episode("ep_3", "t3", "Load matching files safely", success=True, script="1. check exists\n2. open file", q_value=0.7),
        _episode("ep_4", "t4", "Read file that does not exist", success=False, script="1. open missing file", q_value=0.1, error="FileNotFoundError: missing"),
    ]
    cfg = CogMemConfig(skill_min_evidence=3)

    cards = proceduralize_episodes(episodes, config=cfg)

    assert len(cards) == 1
    card = cards[0]
    assert card["task_type"] == "file_io"
    assert "glob" in card["triggers"]
    assert card["evidence_episode_ids"] == ["ep_1", "ep_2", "ep_3"]
    assert "avoid FileNotFoundError failure modes" in card["anti_patterns"]


def test_build_skill_cards_validates_and_promotes(tmp_path):
    train_episodes = [
        _episode("ep_1", "t1", "Use glob to list files in a directory", success=True, script="1. import glob\n2. list files", q_value=0.9),
        _episode("ep_2", "t2", "Read all files from a directory with glob", success=True, script="1. import glob\n2. open files", q_value=0.8),
        _episode("ep_3", "t3", "Load matching files safely", success=True, script="1. check exists\n2. open file", q_value=0.7),
    ]
    dev_episodes = [
        _episode("ep_4", "dev1", "Use glob to inspect files in directory", success=True, script="1. import glob\n2. inspect", q_value=0.95),
        _episode("ep_5", "dev2", "Open directory files safely", success=True, script="1. check exists\n2. open", q_value=0.85),
    ]
    cfg = CogMemConfig(
        skill_min_evidence=3,
        skill_validation_min_matches=1,
        skill_min_transfer_gain=0.0,
        skill_confidence_threshold=0.0,
    )
    path = tmp_path / "skills.json"

    store = build_skill_cards(train_episodes, dev_episodes, config=cfg, output_path=str(path))

    assert path.exists()
    assert len(store) == 1
    card = next(iter(store))
    assert card["validation"]["matched_episodes"] >= 1
    assert card["status"] == "promoted"
    assert card["distinct_task_count"] >= 3
    assert card["activation_conditions"]
    assert card["stop_conditions"]


def test_validate_skill_card_entrypoint(tmp_path):
    train_episodes = [
        _episode("ep_1", "t1", "Use glob to list files in a directory", success=True, script="1. import glob\n2. list files", q_value=0.9),
        _episode("ep_2", "t2", "Read all files from a directory with glob", success=True, script="1. import glob\n2. open files", q_value=0.8),
        _episode("ep_3", "t3", "Load matching files safely", success=True, script="1. check exists\n2. open file", q_value=0.7),
        _episode("ep_4", "dev1", "Use glob to inspect files in directory", success=True, script="1. import glob\n2. inspect", q_value=0.95),
    ]
    cfg = CogMemConfig(
        skill_min_evidence=3,
        skill_validation_min_matches=1,
        skill_min_transfer_gain=0.0,
        skill_confidence_threshold=0.0,
    )
    skill_store_path = tmp_path / "skills.json"
    bank_path = tmp_path / "memory_bank.json"

    store = build_skill_cards(train_episodes[:3], train_episodes[3:], config=cfg, output_path=str(skill_store_path))
    MemoryBank(train_episodes).save(str(bank_path))
    skill_id = next(iter(store))["skill_id"]

    validated = validate_skill_card(
        skill_id,
        ["dev1"],
        str(skill_store_path),
        str(bank_path),
        config=cfg,
    )

    assert validated["skill_id"] == skill_id
    assert validated["validation"]["matched_episodes"] == 1


def test_proceduralize_bigcodebench_splits_by_domain_and_filters_template_tokens():
    episodes = [
        _episode(
            "ep_p1",
            "t1",
            "You are starting task_func to group dataframe rows with pandas groupby",
            success=True,
            script="1. import pandas as pd\n2. df.groupby('city').sum()",
            q_value=0.9,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
        _episode(
            "ep_p2",
            "t2",
            "Use pandas to merge and group dataframe columns",
            success=True,
            script="1. import pandas as pd\n2. df.groupby('team').mean()",
            q_value=0.85,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
        _episode(
            "ep_p3",
            "t3",
            "Aggregate dataframe totals with pandas groupby",
            success=True,
            script="1. import pandas as pd\n2. df.groupby('dept').size()",
            q_value=0.8,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
        _episode(
            "ep_f1",
            "t4",
            "You are starting task_func to list files with glob and pathlib",
            success=True,
            script="1. import glob\n2. glob.glob('*.csv')",
            q_value=0.92,
            task_type="bigcodebench",
            libs=["glob", "pathlib"],
        ),
        _episode(
            "ep_f2",
            "t5",
            "Read matching files using pathlib and glob",
            success=True,
            script="1. from pathlib import Path\n2. list(Path('.').glob('*.txt'))",
            q_value=0.88,
            task_type="bigcodebench",
            libs=["pathlib"],
        ),
        _episode(
            "ep_f3",
            "t6",
            "Open files safely after glob matching",
            success=True,
            script="1. import glob\n2. files = glob.glob('*.json')",
            q_value=0.83,
            task_type="bigcodebench",
            libs=["glob"],
        ),
    ]
    cfg = CogMemConfig(skill_min_evidence=3)

    cards = proceduralize_episodes(episodes, config=cfg)

    assert len(cards) == 2
    task_types = {card["task_type"] for card in cards}
    assert "pandas:groupby" in task_types
    assert "file_io:glob" in task_types
    pandas_card = next(card for card in cards if card["task_type"] == "pandas:groupby")
    assert "task_func" not in pandas_card["triggers"]
    assert "starting" not in pandas_card["triggers"]
    assert "import" not in pandas_card["triggers"]
    assert "self" not in pandas_card["triggers"]
    assert "def" not in pandas_card["triggers"]


def test_validate_skill_cards_do_not_match_all_bigcodebench_tasks(tmp_path):
    train_episodes = [
        _episode(
            "ep_1",
            "t1",
            "Use pandas groupby to aggregate dataframe rows",
            success=True,
            script="1. import pandas as pd\n2. df.groupby('city').sum()",
            q_value=0.9,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
        _episode(
            "ep_2",
            "t2",
            "Aggregate dataframe columns with pandas groupby",
            success=True,
            script="1. import pandas as pd\n2. df.groupby('team').mean()",
            q_value=0.8,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
        _episode(
            "ep_3",
            "t3",
            "Summarize dataframe totals via groupby",
            success=True,
            script="1. import pandas as pd\n2. df.groupby('dept').size()",
            q_value=0.75,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
    ]
    dev_episodes = [
        _episode(
            "ep_4",
            "dev1",
            "Use pandas groupby to inspect dataframe totals",
            success=True,
            script="1. import pandas as pd\n2. df.groupby('name').count()",
            q_value=0.95,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
        _episode(
            "ep_5",
            "dev2",
            "List matching files with glob in a directory",
            success=True,
            script="1. import glob\n2. glob.glob('*.csv')",
            q_value=0.85,
            task_type="bigcodebench",
            libs=["glob"],
        ),
        _episode(
            "ep_6",
            "dev3",
            "Merge dataframe columns with pandas",
            success=True,
            script="1. import pandas as pd\n2. df.merge(other, on='id')",
            q_value=0.82,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
    ]
    cfg = CogMemConfig(
        skill_min_evidence=3,
        skill_validation_min_matches=1,
        skill_min_transfer_gain=0.0,
        skill_confidence_threshold=0.0,
    )
    path = tmp_path / "skills.json"

    store = build_skill_cards(train_episodes, dev_episodes, config=cfg, output_path=str(path))

    card = next(iter(store))
    assert card["task_type"] == "pandas:groupby"
    assert card["validation"]["matched_episodes"] == 1


def test_plan_steps_prefer_code_over_reasoning_text():
    episodes = [
        _episode(
            "ep_1",
            "t1",
            "Plot chart with matplotlib",
            success=True,
            script="Thought: To solve this task, I need to:\nimport matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot(xs, ys)",
            q_value=0.9,
            task_type="bigcodebench",
            libs=["matplotlib"],
        ),
        _episode(
            "ep_2",
            "t2",
            "Render matplotlib plot for the series",
            success=True,
            script="Thought: To solve this task, I need to:\nimport matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot(xs2, ys2)",
            q_value=0.85,
            task_type="bigcodebench",
            libs=["matplotlib"],
        ),
        _episode(
            "ep_3",
            "t3",
            "Build a matplotlib plot figure",
            success=True,
            script="Thought: To solve this task, I need to:\nimport matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot(xs3, ys3)",
            q_value=0.8,
            task_type="bigcodebench",
            libs=["matplotlib"],
        ),
    ]
    cfg = CogMemConfig(skill_min_evidence=3)

    cards = proceduralize_episodes(episodes, config=cfg)

    card = next(card for card in cards if card["task_type"] == "matplotlib:plot")
    assert all(not step.startswith("Thought:") for step in card["plan_steps"])
    assert all("import matplotlib" not in step.lower() for step in card["plan_steps"])
    assert any("figure and axes" in step.lower() for step in card["plan_steps"])


def test_validate_skill_card_requires_multiple_distinct_tasks_for_promotion(tmp_path):
    train_episodes = [
        _episode(
            "ep_1",
            "task_shared",
            "Validate dataframe columns before groupby",
            success=True,
            script="import pandas as pd\ndf.groupby('city').sum()",
            q_value=0.9,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
        _episode(
            "ep_2",
            "task_shared",
            "Validate dataframe columns before aggregation",
            success=True,
            script="import pandas as pd\ndf.groupby('team').mean()",
            q_value=0.85,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
        _episode(
            "ep_3",
            "task_shared",
            "Check dataframe columns before totals",
            success=True,
            script="import pandas as pd\ndf.groupby('dept').size()",
            q_value=0.82,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
    ]
    dev_episodes = [
        _episode(
            "ep_4",
            "dev1",
            "Use pandas groupby to inspect dataframe totals",
            success=True,
            script="import pandas as pd\ndf.groupby('name').count()",
            q_value=0.95,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
    ]
    cfg = CogMemConfig(
        skill_min_evidence=3,
        skill_min_distinct_tasks=2,
        skill_validation_min_matches=1,
        skill_min_transfer_gain=0.0,
        skill_confidence_threshold=0.0,
    )

    store = build_skill_cards(train_episodes, dev_episodes, config=cfg, output_path=str(tmp_path / "skills.json"))

    card = next(iter(store))
    assert card["distinct_task_count"] == 1
    assert card["status"] == "candidate"


def test_validate_skill_card_uses_unique_matched_tasks_for_promotion(tmp_path):
    train_episodes = [
        _episode(
            "ep_1",
            "task_a",
            "Use pandas groupby to aggregate dataframe rows",
            success=True,
            script="import pandas as pd\ndf.groupby('city').sum()",
            q_value=0.9,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
        _episode(
            "ep_2",
            "task_b",
            "Aggregate dataframe columns with pandas groupby",
            success=True,
            script="import pandas as pd\ndf.groupby('team').mean()",
            q_value=0.85,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
        _episode(
            "ep_3",
            "task_c",
            "Summarize dataframe totals via groupby",
            success=True,
            script="import pandas as pd\ndf.groupby('dept').size()",
            q_value=0.82,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
    ]
    dev_episodes = [
        _episode(
            "ep_4",
            "dev_shared",
            "Use pandas groupby to inspect dataframe totals",
            success=True,
            script="import pandas as pd\ndf.groupby('name').count()",
            q_value=0.95,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
        _episode(
            "ep_5",
            "dev_shared",
            "Use pandas groupby to inspect dataframe totals again",
            success=True,
            script="import pandas as pd\ndf.groupby('name').sum()",
            q_value=0.91,
            task_type="bigcodebench",
            libs=["pandas"],
        ),
    ]
    cfg = CogMemConfig(
        skill_min_evidence=3,
        skill_min_distinct_tasks=2,
        skill_validation_min_matches=2,
        skill_min_transfer_gain=0.0,
        skill_confidence_threshold=0.0,
    )

    store = build_skill_cards(train_episodes, dev_episodes, config=cfg, output_path=str(tmp_path / "skills.json"))

    card = next(iter(store))
    assert card["validation"]["matched_episodes"] == 2
    assert card["validation"]["matched_tasks"] == 1
    assert card["status"] == "candidate"


def test_rank_skill_cards_penalizes_negative_transfer_and_stop_risk():
    cards = [
        {
            "skill_id": "skill_good",
            "task_type": "matplotlib:plot",
            "domain": "matplotlib",
            "triggers": ["matplotlib", "plot", "axes"],
            "activation_conditions": ["the task prompt mentions matplotlib plot"],
            "stop_conditions": ["stop and reassess if the mismatch comes from task expectations rather than plot construction"],
            "transfer_gain": 0.6,
            "confidence": 0.9,
            "negative_transfer_rate": 0.0,
            "status": "promoted",
        },
        {
            "skill_id": "skill_bad",
            "task_type": "matplotlib:plot",
            "domain": "matplotlib",
            "triggers": ["matplotlib", "plot", "axes"],
            "activation_conditions": ["the task prompt mentions matplotlib plot"],
            "stop_conditions": ["stop and reassess if the failure mode or traceback points to AssertionError"],
            "transfer_gain": 0.1,
            "confidence": 0.4,
            "negative_transfer_rate": 0.8,
            "runtime_stats": {"retrieved": 5, "helped": 1, "hurt": 3},
            "status": "promoted",
        },
    ]

    ranked = rank_skill_cards_for_task(
        cards,
        {
            "task_id": "task_plot",
            "instruct_prompt": "Plot values with matplotlib and fix the AssertionError in the chart output",
            "libs": ["matplotlib"],
            "error": "AssertionError: chart mismatch",
        },
        limit=2,
        promoted_only=True,
    )

    assert [card["skill_id"] for card in ranked] == ["skill_good"]


def test_rank_skill_cards_abstains_when_low_coverage_skill_is_cross_domain():
    cards = [
        {
            "skill_id": "skill_plot",
            "task_type": "matplotlib:plot",
            "domain": "matplotlib",
            "triggers": ["matplotlib", "plot", "axes"],
            "activation_conditions": ["the task prompt mentions matplotlib plot"],
            "transfer_gain": 0.6,
            "confidence": 0.9,
            "negative_transfer_rate": 0.0,
            "status": "promoted",
        }
    ]
    cfg = CogMemConfig(
        skill_retrieval_min_score=4.5,
        skill_retrieval_strict_min_score=6.0,
        skill_retrieval_min_promoted_families_for_broad_match=3,
    )

    ranked = rank_skill_cards_for_task(
        cards,
        {
            "task_id": "task_file",
            "instruct_prompt": "Read all JSON files from a directory and aggregate counts",
            "libs": ["json", "pathlib"],
        },
        limit=1,
        promoted_only=True,
        config=cfg,
    )

    assert ranked == []


def test_rank_skill_cards_disables_harmful_skill_from_runtime_stats():
    cards = [
        {
            "skill_id": "skill_plot",
            "task_type": "matplotlib:plot",
            "domain": "matplotlib",
            "triggers": ["matplotlib", "plot", "axes"],
            "activation_conditions": ["the task prompt mentions matplotlib plot"],
            "transfer_gain": 0.6,
            "confidence": 0.9,
            "negative_transfer_rate": 0.0,
            "runtime_stats": {"retrieved": 12, "helped": 0, "hurt": 4},
            "status": "promoted",
        }
    ]
    cfg = CogMemConfig(
        skill_retrieval_min_promoted_families_for_broad_match=1,
        skill_runtime_disable_min_retrieved=8,
        skill_runtime_disable_min_hurt=3,
        skill_runtime_disable_hurt_rate=0.10,
    )

    ranked = rank_skill_cards_for_task(
        cards,
        {
            "task_id": "task_plot",
            "instruct_prompt": "Plot values with matplotlib",
            "libs": ["matplotlib"],
        },
        limit=1,
        promoted_only=True,
        config=cfg,
    )

    assert ranked == []

from cogmem.config import CogMemConfig
from cogmem.consolidation.proceduralize import (
    build_skill_cards,
    proceduralize_episodes,
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

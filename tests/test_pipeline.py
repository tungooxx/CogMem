from pathlib import Path

from cogmem.config import CogMemConfig
from cogmem.consolidation.pipeline import run_qstar_cycle
from cogmem.memory.memory_bank import MemoryBank
from cogmem.memory.skill_store import SkillStore


def _episode(
    episode_id: str,
    task_id: str,
    task_description: str,
    *,
    helpfulness: float,
) -> dict:
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "task_description": task_description,
        "task_type": "file_io",
        "script": "def solve(path):\n    return path\n",
        "generated_code": "def solve(path):\n    return path\n",
        "final_code": "def solve(path):\n    return path\n",
        "success": True,
        "q_value": helpfulness,
        "episode_helpfulness": helpfulness,
        "manifest_id": "manifest_a",
        "source_benchmark": "bigcodebench",
    }


def test_run_qstar_cycle_prefers_skill_card_sft_pairs(tmp_path, monkeypatch):
    bank_path = tmp_path / "memory_bank.json"
    MemoryBank(
        [
            _episode("ep_1", "task_1", "Read a file safely", helpfulness=0.95),
            _episode("ep_2", "task_2", "Load a file from disk", helpfulness=0.92),
            _episode("ep_3", "task_3", "Open a file with validation", helpfulness=0.91),
        ]
    ).save(str(bank_path))

    config = CogMemConfig(
        experiments_dir=str(tmp_path / "experiments"),
        adapters_dir=str(tmp_path / "adapters"),
        adapter_registry_path=str(tmp_path / "adapters" / "registry.json"),
        skills_dir=str(tmp_path / "skills"),
        logs_dir=str(tmp_path / "logs"),
        min_holdout=1,
        q_threshold=0.5,
        skill_min_evidence=1,
        skill_validation_min_matches=0,
        skill_min_transfer_gain=0.0,
        skill_confidence_threshold=0.0,
    )

    promoted_card = {
        "skill_id": "skill_file_io",
        "task_type": "file_io",
        "domain": "filesystem",
        "manifest_ids": ["manifest_a"],
        "evidence_episode_ids": ["ep_2"],
        "transfer_gain": 0.8,
        "confidence": 0.9,
        "status": "promoted",
    }

    monkeypatch.setattr(
        "cogmem.consolidation.pipeline.build_skill_cards",
        lambda *args, **kwargs: SkillStore([promoted_card]),
    )
    monkeypatch.setattr(
        "cogmem.consolidation.pipeline.prepare_preference_dataset",
        lambda *args, **kwargs: [],
    )

    captured: dict = {}

    def _fake_train_generator_full(selected, pref_dataset, config, cycle=0, **kwargs):
        captured["selected"] = selected
        captured["pref_dataset"] = pref_dataset
        captured["kwargs"] = kwargs
        adapter_dir = Path(config.adapters_dir) / "generator_v0"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        return str(adapter_dir)

    monkeypatch.setattr(
        "cogmem.consolidation.train_generator.train_generator_full",
        _fake_train_generator_full,
    )
    monkeypatch.setattr(
        "cogmem.consolidation.train_verifier.train_verifier",
        lambda *args, **kwargs: None,
    )

    results = run_qstar_cycle(str(bank_path), config, cycle=0, run_task_fn=None)

    assert results["training_source"] == "skill_cards"
    assert results["training_examples"] == 2
    assert captured["kwargs"]["source_skill_card_ids"] == ["skill_file_io"]
    assert len(captured["kwargs"]["sft_pairs"]) == 2
    assert {pair["source_kind"] for pair in captured["kwargs"]["sft_pairs"]} == {
        "skill_evidence",
        "skill_curriculum",
    }
    assert all(pair["source_skill_card"] == "skill_file_io" for pair in captured["kwargs"]["sft_pairs"])
    assert all(pair["source_episode"] == "ep_2" for pair in captured["kwargs"]["sft_pairs"])


def test_run_qstar_cycle_reuses_existing_skill_store(tmp_path, monkeypatch):
    bank_path = tmp_path / "memory_bank.json"
    MemoryBank(
        [
            _episode("ep_1", "task_1", "Read a file safely", helpfulness=0.95),
            _episode("ep_2", "task_2", "Load a file from disk", helpfulness=0.92),
            _episode("ep_3", "task_3", "Open a file with validation", helpfulness=0.91),
        ]
    ).save(str(bank_path))

    config = CogMemConfig(
        experiments_dir=str(tmp_path / "experiments"),
        adapters_dir=str(tmp_path / "adapters"),
        adapter_registry_path=str(tmp_path / "adapters" / "registry.json"),
        skills_dir=str(tmp_path / "skills"),
        logs_dir=str(tmp_path / "logs"),
        min_holdout=1,
        q_threshold=0.5,
        skill_min_evidence=1,
        skill_validation_min_matches=0,
        skill_min_transfer_gain=0.0,
        skill_confidence_threshold=0.0,
    )

    skills_path = tmp_path / "prebuilt_skills.json"
    SkillStore(
        [
            {
                "skill_id": "skill_prebuilt",
                "task_type": "file_io",
                "domain": "filesystem",
                "manifest_ids": ["manifest_a"],
                "evidence_episode_ids": ["ep_2"],
                "transfer_gain": 0.8,
                "confidence": 0.9,
                "status": "promoted",
            }
        ]
    ).save(str(skills_path))

    def _should_not_build(*args, **kwargs):
        raise AssertionError("build_skill_cards should not be called when an existing skill store is provided")

    monkeypatch.setattr(
        "cogmem.consolidation.pipeline.build_skill_cards",
        _should_not_build,
    )
    monkeypatch.setattr(
        "cogmem.consolidation.pipeline.prepare_preference_dataset",
        lambda *args, **kwargs: [],
    )

    captured: dict = {}

    def _fake_train_generator_full(selected, pref_dataset, config, cycle=0, **kwargs):
        captured["kwargs"] = kwargs
        adapter_dir = Path(config.adapters_dir) / "generator_v0"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        return str(adapter_dir)

    monkeypatch.setattr(
        "cogmem.consolidation.train_generator.train_generator_full",
        _fake_train_generator_full,
    )
    monkeypatch.setattr(
        "cogmem.consolidation.train_verifier.train_verifier",
        lambda *args, **kwargs: None,
    )

    results = run_qstar_cycle(
        str(bank_path),
        config,
        cycle=0,
        run_task_fn=None,
        existing_skill_cards_path=str(skills_path),
    )

    assert results["skill_cards_path"] == str(skills_path)
    assert results["training_source"] == "skill_cards"
    assert captured["kwargs"]["source_skill_card_ids"] == ["skill_prebuilt"]

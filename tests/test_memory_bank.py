import json
import pytest
from cogmem.memory.memory_bank import MemoryBank


class TestMemoryBankLoad:
    def test_load_from_json(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        assert len(bank) == 15

    def test_episode_access(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        ep = bank.get("ep_001")
        assert ep["task_type"] == "clean"
        assert ep["q_value"] == 0.92
        assert ep["episode_helpfulness"] == 0.92

    def test_get_missing_returns_none(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        assert bank.get("nonexistent") is None


class TestMemoryBankSave:
    def test_save_roundtrip(self, sample_memory_bank_path, tmp_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        out = str(tmp_path / "out.json")
        bank.save(out)
        bank2 = MemoryBank.load(out)
        assert len(bank2) == len(bank)


class TestMemoryBankQuery:
    def test_successful_episodes(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        successes = bank.successful()
        assert all(ep["success"] for ep in successes)
        assert len(successes) == 10

    def test_by_task_type(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        clean_eps = bank.by_task_type("clean")
        assert len(clean_eps) == 3
        assert all(ep["task_type"] == "clean" for ep in clean_eps)

    def test_task_types(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        types = bank.task_types()
        assert types == {"clean", "cool", "examine", "heat", "pick", "puttwo"}


class TestMemoryBankSplit:
    def test_holdout_split_size(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        holdout, available = bank.stratified_holdout(n=6, seed=42)
        assert len(holdout) == 6
        assert len(available) == 9
        assert len(holdout) + len(available) == 15

    def test_holdout_stratified(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        holdout, _ = bank.stratified_holdout(n=6, seed=42)
        holdout_types = {ep["task_type"] for ep in holdout}
        assert len(holdout_types) >= 3  # at least 3 task types represented

    def test_holdout_deterministic(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        h1, _ = bank.stratified_holdout(n=6, seed=42)
        h2, _ = bank.stratified_holdout(n=6, seed=42)
        ids1 = {ep["episode_id"] for ep in h1}
        ids2 = {ep["episode_id"] for ep in h2}
        assert ids1 == ids2


class TestMemoryBankMetrics:
    def test_summary_metrics(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        m = bank.summary_metrics()
        assert m["total_episodes"] == 15
        assert 0 < m["success_rate"] < 1
        assert "clean" in m["success_rate_by_type"]
        assert "mean" in m["q_value_stats"]
        assert "mean" in m["episode_helpfulness_stats"]
        assert "high_q_episodes" in m
        assert "low_q_episodes" in m

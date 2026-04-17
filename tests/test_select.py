import pytest
from cogmem.config import CogMemConfig
from cogmem.consolidation.select import (
    filter_manifest_eligible,
    select_q_top_k,
    select_recency,
    select_frequency,
    select_random,
    select_all,
)


@pytest.fixture
def config():
    return CogMemConfig(q_threshold=0.7)


class TestQTopK:
    def test_selects_above_threshold(self, sample_episodes, config):
        selected = select_q_top_k(sample_episodes, config)
        assert all(ep["q_value"] >= 0.7 for ep in selected)

    def test_sorted_by_q_descending(self, sample_episodes, config):
        selected = select_q_top_k(sample_episodes, config)
        q_values = [ep["q_value"] for ep in selected]
        assert q_values == sorted(q_values, reverse=True)

    def test_count(self, sample_episodes, config):
        selected = select_q_top_k(sample_episodes, config)
        expected = sum(1 for ep in sample_episodes if ep["q_value"] >= 0.7)
        assert len(selected) == expected


class TestRecency:
    def test_matches_q_top_k_count(self, sample_episodes, config):
        n_q = len(select_q_top_k(sample_episodes, config))
        selected = select_recency(sample_episodes, config)
        assert len(selected) == n_q

    def test_sorted_by_timestamp_descending(self, sample_episodes, config):
        selected = select_recency(sample_episodes, config)
        timestamps = [ep["timestamp"] for ep in selected]
        assert timestamps == sorted(timestamps, reverse=True)


class TestFrequency:
    def test_matches_q_top_k_count(self, sample_episodes, config):
        n_q = len(select_q_top_k(sample_episodes, config))
        selected = select_frequency(sample_episodes, config)
        assert len(selected) == n_q

    def test_sorted_by_visits_descending(self, sample_episodes, config):
        selected = select_frequency(sample_episodes, config)
        visits = [ep["q_visits"] for ep in selected]
        assert visits == sorted(visits, reverse=True)


class TestRandom:
    def test_matches_q_top_k_count(self, sample_episodes, config):
        n_q = len(select_q_top_k(sample_episodes, config))
        selected = select_random(sample_episodes, config)
        assert len(selected) == n_q

    def test_deterministic_with_seed(self, sample_episodes, config):
        s1 = select_random(sample_episodes, config, seed=42)
        s2 = select_random(sample_episodes, config, seed=42)
        ids1 = [ep["episode_id"] for ep in s1]
        ids2 = [ep["episode_id"] for ep in s2]
        assert ids1 == ids2

    def test_different_seed_different_result(self, sample_episodes, config):
        s1 = select_random(sample_episodes, config, seed=42)
        s2 = select_random(sample_episodes, config, seed=99)
        ids1 = {ep["episode_id"] for ep in s1}
        ids2 = {ep["episode_id"] for ep in s2}
        assert ids1 != ids2


class TestAll:
    def test_returns_everything(self, sample_episodes, config):
        selected = select_all(sample_episodes, config)
        assert len(selected) == len(sample_episodes)


class TestManifestFiltering:
    def test_filters_to_allowed_manifests(self, sample_episodes):
        episodes = []
        for idx, ep in enumerate(sample_episodes[:4]):
            clone = dict(ep)
            clone["manifest_id"] = "manifest_a" if idx % 2 == 0 else "manifest_b"
            episodes.append(clone)

        cfg = CogMemConfig(q_threshold=0.0, allowed_manifest_ids=["manifest_a"])
        filtered = filter_manifest_eligible(episodes, cfg)

        assert len(filtered) == 2
        assert {ep["manifest_id"] for ep in filtered} == {"manifest_a"}


class TestFairComparison:
    def test_all_policies_same_count_except_all(self, sample_episodes, config):
        n_q = len(select_q_top_k(sample_episodes, config))
        assert len(select_recency(sample_episodes, config)) == n_q
        assert len(select_frequency(sample_episodes, config)) == n_q
        assert len(select_random(sample_episodes, config)) == n_q
        # 'all' is intentionally different
        assert len(select_all(sample_episodes, config)) == len(sample_episodes)

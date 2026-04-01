import pytest
from cogmem.consolidation.verify import (
    binomial_ci,
    aggregate_seed_results,
    verification_passed,
)


class TestBinomialCI:
    def test_basic_ci(self):
        low, high = binomial_ci(successes=7, total=10, confidence=0.95)
        assert 0.3 < low < 0.7
        assert 0.8 < high <= 1.0

    def test_zero_successes(self):
        low, high = binomial_ci(successes=0, total=10, confidence=0.95)
        assert low == 0.0
        assert high > 0.0

    def test_all_successes(self):
        low, high = binomial_ci(successes=10, total=10, confidence=0.95)
        assert low > 0.0
        assert high == 1.0


class TestAggregateSeeds:
    def test_mean_and_std(self):
        results = [
            {"success_rate": 0.5, "successes": 5, "total": 10},
            {"success_rate": 0.6, "successes": 6, "total": 10},
            {"success_rate": 0.7, "successes": 7, "total": 10},
        ]
        agg = aggregate_seed_results(results)
        assert abs(agg["mean"] - 0.6) < 0.01
        assert agg["std"] > 0
        assert agg["n_seeds"] == 3


class TestVerificationPassed:
    def test_passes_when_better(self):
        assert verification_passed(
            consolidated_rate=0.6, baseline_rate=0.4, threshold=0.05
        )

    def test_passes_when_equal(self):
        assert verification_passed(
            consolidated_rate=0.4, baseline_rate=0.4, threshold=0.05
        )

    def test_fails_when_regressed(self):
        assert not verification_passed(
            consolidated_rate=0.3, baseline_rate=0.4, threshold=0.05
        )

    def test_passes_within_threshold(self):
        assert verification_passed(
            consolidated_rate=0.36, baseline_rate=0.4, threshold=0.05
        )

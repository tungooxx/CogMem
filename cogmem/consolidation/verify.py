from statistics import mean, stdev

from scipy import stats


def binomial_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval."""
    if total == 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    low = stats.beta.ppf(alpha / 2, successes, total - successes + 1) if successes > 0 else 0.0
    high = stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes) if successes < total else 1.0
    return low, high


def run_verification_single_seed(
    holdout_episodes: list[dict],
    run_task_fn,
    seed: int,
) -> dict:
    import random
    random.seed(seed)

    successes = 0
    for ep in holdout_episodes:
        result = run_task_fn(ep["task_description"])
        if result.get("success"):
            successes += 1

    rate = successes / len(holdout_episodes) if holdout_episodes else 0
    ci_low, ci_high = binomial_ci(successes, len(holdout_episodes))

    return {
        "success_rate": rate,
        "successes": successes,
        "total": len(holdout_episodes),
        "ci_95": (ci_low, ci_high),
        "seed": seed,
    }


def aggregate_seed_results(results: list[dict]) -> dict:
    rates = [r["success_rate"] for r in results]
    return {
        "mean": mean(rates),
        "std": stdev(rates) if len(rates) > 1 else 0.0,
        "min": min(rates),
        "max": max(rates),
        "n_seeds": len(results),
        "per_seed": results,
    }


def verification_passed(
    consolidated_rate: float, baseline_rate: float, threshold: float = 0.05
) -> bool:
    return consolidated_rate >= baseline_rate - threshold

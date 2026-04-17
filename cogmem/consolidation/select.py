import random as _random


def filter_manifest_eligible(episodes: list[dict], config) -> list[dict]:
    """Filter episodes to manifests explicitly allowed by config."""
    allowed = set(getattr(config, "allowed_manifest_ids", []) or [])
    blocked = set(getattr(config, "blocked_manifest_ids", []) or [])
    require_manifest_match = bool(getattr(config, "require_manifest_match", False))

    if require_manifest_match and not allowed:
        raise ValueError(
            "require_manifest_match=True but no allowed_manifest_ids were configured"
        )

    filtered = []
    for ep in episodes:
        manifest_id = ep.get("manifest_id")
        if blocked and manifest_id in blocked:
            continue
        if allowed and manifest_id not in allowed:
            continue
        if require_manifest_match and not manifest_id:
            continue
        filtered.append(ep)
    return filtered


def select_q_top_k(episodes: list[dict], config) -> list[dict]:
    eligible = filter_manifest_eligible(episodes, config)
    sorted_eps = sorted(eligible, key=lambda x: x["q_value"], reverse=True)
    if config.q_threshold is not None:
        selected = [ep for ep in sorted_eps if ep["q_value"] >= config.q_threshold]
        if selected:
            return selected
    # Fallback: take top 25% by Q-value
    n = max(1, len(sorted_eps) // 4)
    return sorted_eps[:n]


def _match_count(episodes: list[dict], config) -> int:
    return len(select_q_top_k(episodes, config))


def select_recency(episodes: list[dict], config, n: int | None = None) -> list[dict]:
    episodes = filter_manifest_eligible(episodes, config)
    if n is None:
        n = _match_count(episodes, config)
    sorted_eps = sorted(episodes, key=lambda x: x["timestamp"], reverse=True)
    return sorted_eps[:n]


def select_frequency(episodes: list[dict], config, n: int | None = None) -> list[dict]:
    episodes = filter_manifest_eligible(episodes, config)
    if n is None:
        n = _match_count(episodes, config)
    sorted_eps = sorted(episodes, key=lambda x: x.get("q_visits", 0), reverse=True)
    return sorted_eps[:n]


def select_random(
    episodes: list[dict], config, n: int | None = None, seed: int = 42
) -> list[dict]:
    episodes = filter_manifest_eligible(episodes, config)
    if n is None:
        n = _match_count(episodes, config)
    rng = _random.Random(seed)
    pool = list(episodes)
    rng.shuffle(pool)
    return pool[: min(n, len(pool))]


def select_all(episodes: list[dict], config) -> list[dict]:
    return list(filter_manifest_eligible(episodes, config))


POLICIES = {
    "q_top_k": select_q_top_k,
    "recency": select_recency,
    "frequency": select_frequency,
    "random": select_random,
    "all": select_all,
}

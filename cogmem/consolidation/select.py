import random as _random


def select_q_top_k(episodes: list[dict], config) -> list[dict]:
    selected = [ep for ep in episodes if ep["q_value"] >= config.q_threshold]
    selected.sort(key=lambda x: x["q_value"], reverse=True)
    return selected


def _match_count(episodes: list[dict], config) -> int:
    return len(select_q_top_k(episodes, config))


def select_recency(episodes: list[dict], config, n: int | None = None) -> list[dict]:
    if n is None:
        n = _match_count(episodes, config)
    sorted_eps = sorted(episodes, key=lambda x: x["timestamp"], reverse=True)
    return sorted_eps[:n]


def select_frequency(episodes: list[dict], config, n: int | None = None) -> list[dict]:
    if n is None:
        n = _match_count(episodes, config)
    sorted_eps = sorted(episodes, key=lambda x: x.get("q_visits", 0), reverse=True)
    return sorted_eps[:n]


def select_random(
    episodes: list[dict], config, n: int | None = None, seed: int = 42
) -> list[dict]:
    if n is None:
        n = _match_count(episodes, config)
    rng = _random.Random(seed)
    pool = list(episodes)
    rng.shuffle(pool)
    return pool[: min(n, len(pool))]


def select_all(episodes: list[dict], config) -> list[dict]:
    return list(episodes)


POLICIES = {
    "q_top_k": select_q_top_k,
    "recency": select_recency,
    "frequency": select_frequency,
    "random": select_random,
    "all": select_all,
}

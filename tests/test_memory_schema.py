from cogmem.memory.schema import (
    get_episode_helpfulness,
    normalize_episode_metrics,
    set_episode_helpfulness,
)


def test_normalize_episode_metrics_mirrors_legacy_q_value():
    episode = {"episode_id": "ep1", "q_value": 0.73}

    normalized = normalize_episode_metrics(episode, copy_episode=True)

    assert normalized["episode_helpfulness"] == 0.73
    assert normalized["q_value"] == 0.73


def test_set_episode_helpfulness_updates_legacy_q_value():
    episode = {"episode_id": "ep1"}

    set_episode_helpfulness(episode, 0.42)

    assert episode["episode_helpfulness"] == 0.42
    assert episode["q_value"] == 0.42


def test_get_episode_helpfulness_falls_back_to_success_flag():
    episode = {"episode_id": "ep1", "success": True}

    assert get_episode_helpfulness(episode) == 1.0

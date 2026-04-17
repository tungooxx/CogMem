"""Shared episodic memory field names and compatibility helpers."""

from __future__ import annotations

from copy import deepcopy


EPISODE_HELPFULNESS_KEY = "episode_helpfulness"
LEGACY_Q_VALUE_KEY = "q_value"
RETRIEVAL_CONFIDENCE_KEY = "retrieval_confidence"
NEGATIVE_TRANSFER_RATE_KEY = "negative_transfer_rate"
CARD_TRANSFER_GAIN_KEY = "card_transfer_gain"
ADAPTER_DEV_GAIN_KEY = "adapter_dev_gain"

DEFAULT_EPISODE_HELPFULNESS = 0.5


def get_episode_helpfulness(episode: dict, default: float = DEFAULT_EPISODE_HELPFULNESS) -> float:
    """Read episodic helpfulness with legacy q_value fallback."""
    if EPISODE_HELPFULNESS_KEY in episode:
        return float(episode[EPISODE_HELPFULNESS_KEY])
    if LEGACY_Q_VALUE_KEY in episode:
        return float(episode[LEGACY_Q_VALUE_KEY])
    if "success" in episode:
        return 1.0 if episode.get("success") else 0.0
    return float(default)


def set_episode_helpfulness(
    episode: dict,
    value: float,
    *,
    mirror_legacy_q_value: bool = True,
) -> dict:
    """Write explicit episodic helpfulness and optionally mirror to q_value."""
    episode[EPISODE_HELPFULNESS_KEY] = float(value)
    if mirror_legacy_q_value:
        episode[LEGACY_Q_VALUE_KEY] = float(value)
    return episode


def normalize_episode_metrics(
    episode: dict,
    *,
    default: float = DEFAULT_EPISODE_HELPFULNESS,
    copy_episode: bool = False,
) -> dict:
    """Ensure explicit episodic helpfulness fields exist on an episode dict."""
    target = deepcopy(episode) if copy_episode else episode
    helpfulness = get_episode_helpfulness(target, default=default)
    set_episode_helpfulness(target, helpfulness, mirror_legacy_q_value=True)
    return target

import pytest
from cogmem.config import CogMemConfig
from cogmem.consolidation.router import ConsolidatedDomain, route_task


class RouterConfig:
    """Minimal config for router tests."""
    consolidation_match_threshold: float = 0.75
    retrieval_min_q: float = 0.3

@pytest.fixture
def config():
    return RouterConfig()


@pytest.fixture
def clean_domain():
    return ConsolidatedDomain(
        name="clean",
        centroid=[0.1] * 384,
        adapter_path="adapters/clean",
    )


class TestRouteTask:
    def test_routes_to_consolidated_when_similar(self, config, clean_domain):
        task_embedding = [0.1] * 384  # identical to centroid
        result = route_task(
            task_embedding=task_embedding,
            consolidated_domains=[clean_domain],
            memory_bank_episodes=[],
            config=config,
        )
        assert result[0] == "consolidated"
        assert result[1] == "adapters/clean"

    def test_routes_to_episodic_when_not_consolidated(self, config):
        task_embedding = [0.9] * 384
        episodes = [
            {
                "intent_embedding": [0.85] * 384,
                "q_value": 0.8,
                "episode_helpfulness": 0.8,
                "episode_id": "ep_1",
            }
        ]
        result = route_task(
            task_embedding=task_embedding,
            consolidated_domains=[],
            memory_bank_episodes=episodes,
            config=config,
        )
        assert result[0] == "episodic"

    def test_routes_to_cold_when_nothing_matches(self, config):
        task_embedding = [0.99] * 384
        result = route_task(
            task_embedding=task_embedding,
            consolidated_domains=[],
            memory_bank_episodes=[],
            config=config,
        )
        assert result[0] == "cold"
        assert result[1] is None

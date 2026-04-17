import pytest
from cogmem.consolidation.adapter_registry import AdapterRegistry
from cogmem.config import CogMemConfig
from cogmem.consolidation.router import ConsolidatedDomain, RoutedAdapters, record_routing_decision, route_task


class RouterConfig:
    """Minimal config for router tests."""
    consolidation_match_threshold: float = 0.75
    retrieval_min_q: float = 0.3
    adapter_route_min_dev_gain: float = 0.0
    adapter_route_allow_global: bool = True
    adapter_route_allow_family: bool = True

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
    def test_routes_to_adapter_registry_before_consolidated(self, config, clean_domain):
        registry = AdapterRegistry([
            {
                "adapter_id": "adapter_global",
                "adapter_path": "adapters/global",
                "base_model": "Qwen/Qwen2.5-3B-Instruct",
                "adapter_role": "global",
                "dev_gain": 0.15,
                "compatible_families": ["general"],
            },
            {
                "adapter_id": "adapter_file_io",
                "adapter_path": "adapters/file_io",
                "base_model": "Qwen/Qwen2.5-3B-Instruct",
                "adapter_role": "family",
                "dev_gain": 0.25,
                "compatible_families": ["clean"],
            },
        ])
        result = route_task(
            task_embedding=[0.1] * 384,
            consolidated_domains=[clean_domain],
            memory_bank_episodes=[],
            config=config,
            adapter_registry=registry,
            task_family="clean",
        )
        assert result[0] == "adapter"
        assert result[1].global_adapter_path == "adapters/global"
        assert result[1].family_adapter_path == "adapters/file_io"

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

    def test_record_routing_decision_stamps_adapter_ids(self):
        episode = {"episode_id": "ep_1"}
        route_kind, payload = "adapter", RoutedAdapters(
            global_adapter_id="adapter_global",
            family_adapter_id="adapter_family",
            adapter_ids=["adapter_global", "adapter_family"],
        )

        stamped = record_routing_decision(episode, route_kind, payload)

        assert stamped["route_kind"] == "adapter"
        assert stamped["adapter_ids"] == ["adapter_global", "adapter_family"]
        assert stamped["selected_global_adapter"] == "adapter_global"
        assert stamped["selected_family_adapter"] == "adapter_family"

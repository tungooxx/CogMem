from dataclasses import dataclass, field

import numpy as np

from cogmem.consolidation.adapter_registry import AdapterRegistry, AdapterArtifact
from cogmem.memory.schema import get_episode_helpfulness


@dataclass
class ConsolidatedDomain:
    name: str
    centroid: list[float]
    adapter_path: str


@dataclass
class RoutedAdapters:
    global_adapter_path: str | None = None
    family_adapter_path: str | None = None
    global_adapter_id: str | None = None
    family_adapter_id: str | None = None
    adapter_ids: list[str] = field(default_factory=list)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm)


def _route_registry_adapters(
    adapter_registry: AdapterRegistry,
    task_family: str | None,
    config,
) -> RoutedAdapters | None:
    selected = adapter_registry.select_routed_adapters(
        task_family=task_family,
        min_dev_gain=getattr(config, "adapter_route_min_dev_gain", 0.0),
        allow_global=getattr(config, "adapter_route_allow_global", True),
        allow_family=getattr(config, "adapter_route_allow_family", True),
    )
    global_adapter: AdapterArtifact | None = selected["global"]
    family_adapter: AdapterArtifact | None = selected["family"]
    if global_adapter is None and family_adapter is None:
        return None

    adapter_ids = []
    if global_adapter is not None:
        adapter_ids.append(global_adapter.adapter_id)
    if family_adapter is not None and family_adapter.adapter_id not in adapter_ids:
        adapter_ids.append(family_adapter.adapter_id)
    return RoutedAdapters(
        global_adapter_path=global_adapter.adapter_path if global_adapter else None,
        family_adapter_path=family_adapter.adapter_path if family_adapter else None,
        global_adapter_id=global_adapter.adapter_id if global_adapter else None,
        family_adapter_id=family_adapter.adapter_id if family_adapter else None,
        adapter_ids=adapter_ids,
    )


def record_routing_decision(
    episode: dict,
    route_kind: str,
    route_payload,
) -> dict:
    """Stamp adapter routing metadata onto an episode record."""
    episode["route_kind"] = route_kind
    if route_kind == "adapter" and isinstance(route_payload, RoutedAdapters):
        episode["adapter_ids"] = list(route_payload.adapter_ids)
        episode["selected_global_adapter"] = route_payload.global_adapter_id
        episode["selected_family_adapter"] = route_payload.family_adapter_id
    else:
        episode.setdefault("adapter_ids", [])
    return episode


def route_task(
    task_embedding: list[float],
    consolidated_domains: list[ConsolidatedDomain],
    memory_bank_episodes: list[dict],
    config,
    *,
    adapter_registry: AdapterRegistry | None = None,
    task_family: str | None = None,
) -> tuple[str, object]:
    if adapter_registry is not None:
        routed = _route_registry_adapters(adapter_registry, task_family, config)
        if routed is not None:
            return ("adapter", routed)

    # Check consolidated domains first (legacy adapter path)
    for domain in consolidated_domains:
        sim = _cosine_sim(task_embedding, domain.centroid)
        if sim > config.consolidation_match_threshold:
            return ("consolidated", domain.adapter_path)

    # Fallback to episodic retrieval
    if memory_bank_episodes:
        scored = []
        for ep in memory_bank_episodes:
            emb = ep.get("intent_embedding")
            if emb:
                sim = _cosine_sim(task_embedding, emb)
                scored.append((sim, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and get_episode_helpfulness(scored[0][1], 0.0) > config.retrieval_min_q:
            top_episodes = [ep for _, ep in scored[:3]]
            return ("episodic", top_episodes)

    return ("cold", None)

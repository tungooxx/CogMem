from dataclasses import dataclass

import numpy as np


@dataclass
class ConsolidatedDomain:
    name: str
    centroid: list[float]
    adapter_path: str


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm)


def route_task(
    task_embedding: list[float],
    consolidated_domains: list[ConsolidatedDomain],
    memory_bank_episodes: list[dict],
    config,
) -> tuple[str, object]:
    # Check consolidated domains first
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
        if scored and scored[0][1].get("q_value", 0) > config.retrieval_min_q:
            top_episodes = [ep for _, ep in scored[:3]]
            return ("episodic", top_episodes)

    return ("cold", None)

"""Episode-first cluster memory bank for cognitive patch retrieval.

Episodes are the primary persisted unit. Cluster memories are built from
similar episodes and may distill reusable LoRA patch artifacts that are
composed at inference time.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cogmem.patches.bank import PatchBank
from cogmem.patches.compose import PatchedModel
from cogmem.patches.create import (
    DEFAULT_PATCH_LR,
    DEFAULT_PATCH_RANK,
    DEFAULT_PATCH_TRAIN_STEPS,
    _tokenize_training_messages,
    create_patch_from_cluster,
)
from cogmem.patches.patch import CognitivePatch

DEFAULT_CLUSTER_MIN_SUPPORT = 3
DEFAULT_CLUSTER_SIMILARITY = 0.52
DEFAULT_LAYER_WINDOW = 4
DEFAULT_TOKEN_WINDOW = 64
DEFAULT_TOP_DIRECTIONS = 3
DEFAULT_CONTROL_EPISODES = 3
DEFAULT_PATCH_SCALE = 0.25


@dataclass
class EpisodeRecord:
    episode_id: str
    task_id: str
    task_embedding: list[float]
    prompt: str
    family_label: str
    failed_code: str
    passed_code: str
    pass_fail_similarity: float
    candidate_patch_id: str = ""
    success: bool = True
    created_at: float = field(default_factory=time.time)
    last_used_at: float = 0.0
    reuse_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EpisodeRecord":
        return cls(**raw)


@dataclass
class ClusterMemory:
    memory_id: str
    family_label: str
    centroid_embedding: list[float]
    member_episode_ids: list[str]
    support_count: int
    layer_window: list[int]
    token_window: int
    top_contrast_directions: list[list[float]] = field(default_factory=list)
    explained_variance: list[float] = field(default_factory=list)
    held_out_steering_gain: float = 0.0
    negative_steering_penalty: float = 0.0
    transfer_rate: float = 0.0
    distillation_success: float = 0.0
    reuse_count: int = 0
    recency_score: float = 0.0
    utility_regression: float = 0.0
    redundancy_penalty: float = 0.0
    q_value: float = 0.0
    distilled_patch_ids: list[str] = field(default_factory=list)
    retrievable: bool = False
    created_at: float = field(default_factory=time.time)
    last_used_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ClusterMemory":
        return cls(**raw)


class ClusterMemoryBank:
    """Default retrieval surface: episodes -> cluster memories -> patch artifacts."""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.episodes: list[EpisodeRecord] = []
        self.memories: list[ClusterMemory] = []
        self._episode_index: dict[str, int] = {}
        self._memory_index: dict[str, int] = {}
        self._patch_bank = PatchBank(str(self.save_dir / "patch_artifacts"))

    def load(self) -> None:
        self._patch_bank.load()
        self.episodes = self._load_records("episodes.json", EpisodeRecord)
        self.memories = self._load_records("memories.json", ClusterMemory)
        self._episode_index = {ep.episode_id: i for i, ep in enumerate(self.episodes)}
        self._memory_index = {mem.memory_id: i for i, mem in enumerate(self.memories)}

    def save(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._patch_bank.save()
        self._save_records("episodes.json", self.episodes)
        self._save_records("memories.json", self.memories)

    def record_episode(
        self,
        task_id: str,
        prompt: str,
        task_embedding: list[float],
        failed_code: str,
        passed_code: str,
        pass_fail_similarity: float,
        success: bool = True,
        family_label: str | None = None,
        episode_id: str | None = None,
    ) -> EpisodeRecord:
        label = family_label or derive_family_label(prompt)
        eid = episode_id or _make_episode_id(task_id, prompt, failed_code, passed_code)
        episode = EpisodeRecord(
            episode_id=eid,
            task_id=task_id,
            task_embedding=list(task_embedding),
            prompt=prompt,
            family_label=label,
            failed_code=failed_code,
            passed_code=passed_code,
            pass_fail_similarity=float(pass_fail_similarity),
            success=bool(success),
        )
        idx = self._episode_index.get(eid)
        if idx is None:
            self._episode_index[eid] = len(self.episodes)
            self.episodes.append(episode)
        else:
            self.episodes[idx] = episode
        return episode

    def build_memories(
        self,
        base_model,
        tokenizer,
        min_support: int = DEFAULT_CLUSTER_MIN_SUPPORT,
        similarity_threshold: float = DEFAULT_CLUSTER_SIMILARITY,
        layer_window_size: int = DEFAULT_LAYER_WINDOW,
        token_window: int = DEFAULT_TOKEN_WINDOW,
        top_directions: int = DEFAULT_TOP_DIRECTIONS,
        control_episodes: int = DEFAULT_CONTROL_EPISODES,
        distill_rank: int = DEFAULT_PATCH_RANK,
        distill_steps: int = DEFAULT_PATCH_TRAIN_STEPS,
        distill_lr: float = DEFAULT_PATCH_LR,
    ) -> dict[str, Any]:
        eligible = [
            ep for ep in self.episodes
            if ep.success and ep.passed_code.strip() and ep.failed_code.strip()
        ]
        groups = _cluster_episodes(eligible, similarity_threshold, min_support)

        memories: list[ClusterMemory] = []
        for group in groups:
            layer_window = _resolve_layer_window(base_model, layer_window_size)
            deltas = [
                _compute_episode_delta(
                    base_model,
                    tokenizer,
                    ep.failed_code,
                    ep.passed_code,
                    layer_window=layer_window,
                    token_window=token_window,
                )
                for ep in group
            ]
            directions, explained = compute_top_contrast_directions(
                np.stack(deltas, axis=0),
                top_k=top_directions,
            )
            memory = ClusterMemory(
                memory_id=_make_memory_id(group),
                family_label=group[0].family_label,
                centroid_embedding=np.mean(
                    np.asarray([ep.task_embedding for ep in group], dtype=np.float32),
                    axis=0,
                ).tolist(),
                member_episode_ids=[ep.episode_id for ep in group],
                support_count=len(group),
                layer_window=list(layer_window),
                token_window=token_window,
                top_contrast_directions=[direction.tolist() for direction in directions],
                explained_variance=explained,
                created_at=max(ep.created_at for ep in group),
            )
            _distill_and_score_memory(
                memory,
                group,
                self.episodes,
                self._patch_bank,
                base_model,
                tokenizer,
                distill_rank=distill_rank,
                distill_steps=distill_steps,
                distill_lr=distill_lr,
                control_episodes=control_episodes,
            )
            memories.append(memory)

        _apply_recency_scores(memories)
        _apply_redundancy_penalties(memories)
        _recompute_q_values(memories)

        self.memories = memories
        self._memory_index = {mem.memory_id: i for i, mem in enumerate(self.memories)}
        self.save()
        return self.stats()

    def get_active_memories(
        self,
        query_embedding: list[float],
        task_prompt: str,
        top_k: int = 5,
    ) -> list[ClusterMemory]:
        if not self.memories:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        family_label = derive_family_label(task_prompt)
        scored: list[tuple[float, ClusterMemory]] = []

        for memory in self.memories:
            if not memory.retrievable or not memory.distilled_patch_ids:
                continue
            centroid = np.asarray(memory.centroid_embedding, dtype=np.float32)
            similarity = cosine_similarity(query, centroid)
            family_match = 1.0 if memory.family_label == family_label else 0.0
            score = similarity * 0.50 + family_match * 0.20 + memory.q_value * 0.30
            scored.append((score, memory))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in scored[:top_k]]

    def get_active_patches(
        self,
        query_embedding: list[float],
        task_prompt: str,
        top_k: int = 5,
    ) -> list[CognitivePatch]:
        memories = self.get_active_memories(query_embedding, task_prompt, top_k=top_k)
        return self.load_patches_for_memories(memories)

    def load_patches_for_memories(
        self,
        memories: list[ClusterMemory],
    ) -> list[CognitivePatch]:
        active: list[CognitivePatch] = []
        seen: set[str] = set()
        for memory in memories:
            for patch_id in memory.distilled_patch_ids:
                if patch_id in seen:
                    continue
                patch = self._patch_bank.get_patch(patch_id)
                if patch is None:
                    continue
                self._patch_bank.load_weights(patch)
                patch.q_value = memory.q_value
                active.append(patch)
                seen.add(patch_id)
        return active

    def update_memory_utility(
        self,
        memory_id: str,
        task_succeeded: bool,
        cold_succeeded: bool | None = None,
    ) -> None:
        idx = self._memory_index.get(memory_id)
        if idx is None:
            return

        memory = self.memories[idx]
        memory.reuse_count += 1
        memory.last_used_at = time.time()

        if cold_succeeded is True and not task_succeeded:
            memory.utility_regression = min(1.0, memory.utility_regression + 0.1)
        elif cold_succeeded is False and task_succeeded:
            memory.utility_regression = max(0.0, memory.utility_regression - 0.05)

        _apply_recency_scores(self.memories)
        _recompute_q_values(self.memories)
        self.save()

    def stats(self) -> dict[str, Any]:
        family_counts: dict[str, int] = {}
        for memory in self.memories:
            family_counts[memory.family_label] = family_counts.get(memory.family_label, 0) + 1
        return {
            "episodes": len(self.episodes),
            "memories": len(self.memories),
            "retrievable_memories": sum(1 for memory in self.memories if memory.retrievable),
            "artifact_patches": len(self._patch_bank.patches),
            "mean_q": float(np.mean([memory.q_value for memory in self.memories])) if self.memories else 0.0,
            "families": family_counts,
        }

    @property
    def artifact_bank(self) -> PatchBank:
        return self._patch_bank

    def _save_records(self, filename: str, records: list[Any]) -> None:
        path = self.save_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([record.to_dict() for record in records], handle, indent=2)

    def _load_records(self, filename: str, record_type):
        path = self.save_dir / filename
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        return [record_type.from_dict(item) for item in raw]


def derive_family_label(prompt: str) -> str:
    text = prompt.lower()
    keyword_groups = [
        ("plotting", ["matplotlib", "plot", "histogram", "scatter", "heatmap", "pairplot", "chart", "seaborn"]),
        ("dataframe", ["dataframe", "pandas", "column", "df"]),
        ("sklearn", ["sklearn", "pca", "regression", "classifier", "scaler", "encoder"]),
        ("file_io", ["csv", "json", "file", "directory", "path"]),
        ("crypto_bytes", ["sha", "hash", "bytes", "hex", "encrypt"]),
        ("random_numeric", ["random", "seed", "simulate", "distribution", "sample"]),
        ("datetime", ["date", "time", "timestamp", "ordinal"]),
        ("networking", ["ip", "port", "socket", "http", "url"]),
    ]
    scores = []
    for label, words in keyword_groups:
        score = sum(1 for word in words if word in text)
        if score:
            scores.append((score, label))
    if scores:
        scores.sort(reverse=True)
        return scores[0][1]
    return "general_code"


def compute_top_contrast_directions(
    deltas: np.ndarray,
    top_k: int = DEFAULT_TOP_DIRECTIONS,
) -> tuple[list[np.ndarray], list[float]]:
    if deltas.ndim != 2:
        raise ValueError(f"expected 2D deltas, got shape {deltas.shape}")
    centered = deltas - deltas.mean(axis=0, keepdims=True)
    if not np.any(centered):
        centered = deltas
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    total = float(np.sum(singular_values ** 2)) + 1e-8
    directions = [vh[i].astype(np.float32) for i in range(min(top_k, vh.shape[0]))]
    explained = [
        float((singular_values[i] ** 2) / total)
        for i in range(min(top_k, singular_values.shape[0]))
    ]
    return directions, explained


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _cluster_episodes(
    episodes: list[EpisodeRecord],
    similarity_threshold: float,
    min_support: int,
) -> list[list[EpisodeRecord]]:
    by_family: dict[str, list[EpisodeRecord]] = {}
    for episode in episodes:
        by_family.setdefault(episode.family_label, []).append(episode)

    groups: list[list[EpisodeRecord]] = []
    for family_episodes in by_family.values():
        if len(family_episodes) < min_support:
            continue
        embs = np.asarray([episode.task_embedding for episode in family_episodes], dtype=np.float32)
        visited = [False] * len(family_episodes)
        for i in range(len(family_episodes)):
            if visited[i]:
                continue
            queue = [i]
            component: list[int] = []
            while queue:
                current = queue.pop()
                if visited[current]:
                    continue
                visited[current] = True
                component.append(current)
                for nxt in range(len(family_episodes)):
                    if visited[nxt]:
                        continue
                    sim = cosine_similarity(embs[current], embs[nxt])
                    if sim >= similarity_threshold:
                        queue.append(nxt)
            if len(component) >= min_support:
                groups.append([family_episodes[idx] for idx in component])
    return groups


def _resolve_layer_window(base_model, layer_window_size: int) -> list[int]:
    total_layers = getattr(getattr(base_model, "config", None), "num_hidden_layers", layer_window_size)
    start = max(total_layers - layer_window_size, 0)
    return list(range(start, total_layers))


def _compute_episode_delta(
    base_model,
    tokenizer,
    failed_code: str,
    passed_code: str,
    layer_window: list[int],
    token_window: int,
) -> np.ndarray:
    failed_vec = _pool_hidden_states(base_model, tokenizer, failed_code, layer_window, token_window)
    passed_vec = _pool_hidden_states(base_model, tokenizer, passed_code, layer_window, token_window)
    return passed_vec - failed_vec


def _pool_hidden_states(
    base_model,
    tokenizer,
    text: str,
    layer_window: list[int],
    token_window: int,
) -> np.ndarray:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max(token_window, 8),
        add_special_tokens=False,
    )
    device = _resolve_model_device(base_model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = base_model(**encoded, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    pooled = []
    for layer_idx in layer_window:
        state = hidden_states[layer_idx + 1]
        window = min(token_window, state.shape[1])
        pooled.append(state[:, -window:, :].mean(dim=1).squeeze(0).float().cpu().numpy())
    return np.concatenate(pooled, axis=0).astype(np.float32)


def _distill_and_score_memory(
    memory: ClusterMemory,
    group: list[EpisodeRecord],
    all_episodes: list[EpisodeRecord],
    patch_bank: PatchBank,
    base_model,
    tokenizer,
    distill_rank: int,
    distill_steps: int,
    distill_lr: float,
    control_episodes: int,
) -> None:
    train_eps, holdout_eps = _split_cluster_examples(group)
    if len(train_eps) < 2 or not holdout_eps:
        return

    examples = [{"prompt": ep.prompt, "code": ep.passed_code} for ep in train_eps]
    patch_id = f"cluster_patch_{memory.memory_id}"
    patch, _ = create_patch_from_cluster(
        base_model,
        tokenizer,
        examples,
        patch_id=patch_id,
        rank=distill_rank,
        n_steps=distill_steps,
        lr=distill_lr,
        return_stats=True,
    )
    patch.embedding = list(memory.centroid_embedding)
    patch.source_type = "cluster_memory"
    patch.description = f"Distilled from {memory.memory_id}"
    patch_bank.add(patch)

    for episode in group:
        episode.candidate_patch_id = patch.patch_id

    holdout_gains = [
        _measure_patch_teacher_forcing_gain(
            base_model,
            tokenizer,
            patch,
            episode.prompt,
            episode.passed_code,
        )
        for episode in holdout_eps
    ]
    control_gains = [
        _measure_patch_teacher_forcing_gain(
            base_model,
            tokenizer,
            patch,
            episode.prompt,
            episode.passed_code,
        )
        for episode in _select_control_episodes(
            all_episodes,
            memory.family_label,
            control_episodes,
        )
    ]

    memory.held_out_steering_gain = float(np.mean(holdout_gains)) if holdout_gains else 0.0
    memory.transfer_rate = float(np.mean([gain > 0 for gain in holdout_gains])) if holdout_gains else 0.0
    harms = [max(-gain, 0.0) for gain in control_gains]
    memory.negative_steering_penalty = float(np.mean(harms)) if harms else 0.0
    memory.distillation_success = 1.0 if (
        memory.held_out_steering_gain > 0 and memory.transfer_rate >= 0.5
    ) else 0.0
    if memory.distillation_success > 0:
        memory.distilled_patch_ids = [patch.patch_id]
        memory.retrievable = True
    else:
        memory.retrievable = False


def _split_cluster_examples(
    group: list[EpisodeRecord],
) -> tuple[list[EpisodeRecord], list[EpisodeRecord]]:
    ordered = sorted(group, key=lambda episode: episode.created_at)
    holdout_count = max(1, len(ordered) // 3)
    holdout = ordered[-holdout_count:]
    train = ordered[:-holdout_count]
    if len(train) < 2:
        train = ordered[:-1]
        holdout = ordered[-1:]
    return train, holdout


def _select_control_episodes(
    episodes: list[EpisodeRecord],
    family_label: str,
    limit: int,
) -> list[EpisodeRecord]:
    controls = [
        episode for episode in episodes
        if episode.family_label != family_label and episode.success and episode.passed_code.strip()
    ]
    controls.sort(key=lambda episode: episode.created_at, reverse=True)
    return controls[:limit]


def _measure_patch_teacher_forcing_gain(
    base_model,
    tokenizer,
    patch: CognitivePatch,
    prompt: str,
    passed_code: str,
) -> float:
    from cogmem.benchmarks.bigcodebench.prompts import SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": passed_code},
    ]
    tok = _tokenize_training_messages(tokenizer, messages, max_length=1024)
    device = _resolve_model_device(base_model)
    model_inputs = {
        "input_ids": torch.tensor([tok["input_ids"]], device=device),
        "attention_mask": torch.tensor([tok["attention_mask"]], device=device),
        "labels": torch.tensor([tok["labels"]], device=device),
    }
    with torch.no_grad():
        cold_loss = float(base_model(**model_inputs).loss.item())
        with PatchedModel(base_model, [patch], scaling_factor=DEFAULT_PATCH_SCALE):
            patched_loss = float(base_model(**model_inputs).loss.item())
    return cold_loss - patched_loss


def _apply_recency_scores(memories: list[ClusterMemory]) -> None:
    if not memories:
        return
    created = [memory.created_at for memory in memories]
    lo = min(created)
    hi = max(created)
    span = max(hi - lo, 1e-8)
    for memory in memories:
        memory.recency_score = float((memory.created_at - lo) / span)


def _apply_redundancy_penalties(memories: list[ClusterMemory]) -> None:
    by_family: dict[str, list[ClusterMemory]] = {}
    for memory in memories:
        by_family.setdefault(memory.family_label, []).append(memory)

    for family_memories in by_family.values():
        for memory in family_memories:
            if len(family_memories) == 1:
                memory.redundancy_penalty = 0.0
                continue
            centroid = np.asarray(memory.centroid_embedding, dtype=np.float32)
            sims = [
                cosine_similarity(centroid, np.asarray(other.centroid_embedding, dtype=np.float32))
                for other in family_memories
                if other.memory_id != memory.memory_id
            ]
            max_sim = max(sims) if sims else 0.0
            memory.redundancy_penalty = float(np.clip((max_sim - 0.85) / 0.15, 0.0, 1.0))


def _recompute_q_values(memories: list[ClusterMemory]) -> None:
    max_reuse = max((memory.reuse_count for memory in memories), default=0)
    norm_denom = max(max_reuse, 1)
    for memory in memories:
        normalized_reuse = memory.reuse_count / norm_denom
        q = (
            0.30 * memory.held_out_steering_gain
            + 0.25 * memory.transfer_rate
            + 0.15 * normalized_reuse
            + 0.05 * memory.distillation_success
            + 0.05 * memory.recency_score
            - 0.10 * memory.utility_regression
            - 0.10 * memory.redundancy_penalty
        )
        memory.q_value = float(np.clip(q, 0.0, 1.0))
        if not memory.distilled_patch_ids or memory.distillation_success <= 0:
            memory.retrievable = False


def _make_episode_id(task_id: str, prompt: str, failed_code: str, passed_code: str) -> str:
    digest = hashlib.sha256(
        f"{task_id}::{prompt}::{failed_code}::{passed_code}".encode("utf-8")
    ).hexdigest()[:12]
    return f"episode_{task_id.replace('/', '_')}_{digest}"


def _make_memory_id(group: list[EpisodeRecord]) -> str:
    family = group[0].family_label
    digest = hashlib.sha256(
        "||".join(sorted(episode.episode_id for episode in group)).encode("utf-8")
    ).hexdigest()[:10]
    return f"memory_{family}_{digest}"


def _resolve_model_device(base_model):
    device = getattr(base_model, "device", None)
    if device is not None:
        return device
    return next(base_model.parameters()).device

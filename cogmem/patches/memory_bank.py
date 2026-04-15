"""Episode-first cluster memory bank for cognitive patch retrieval.

Episodes are the primary persisted unit. Cluster memories are built from
similar episodes and may distill reusable LoRA patch artifacts that are
composed at inference time.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
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
DEFAULT_APPLICABILITY_POS_WEIGHT = 0.60
DEFAULT_APPLICABILITY_NEG_WEIGHT = 0.25
DEFAULT_APPLICABILITY_STRUCT_WEIGHT = 0.15
DEFAULT_USE_TRANSFER_WEIGHT = 0.45
DEFAULT_USE_RECENT_SUCCESS_WEIGHT = 0.20
DEFAULT_USE_REUSE_WEIGHT = 0.15
DEFAULT_USE_ONLINE_HURT_WEIGHT = 0.20
DEFAULT_FINAL_USE_PROMOTE_WEIGHT = 0.15
DEFAULT_MULTI_MEMORY_CONFIDENCE = 0.80
DEFAULT_MULTI_MEMORY_MARGIN = 0.05
DEFAULT_SLEEP_PROMOTE_MIN_SCORE = 0.40
DEFAULT_SLEEP_PROMOTE_MIN_SUPPORT = 3
DEFAULT_SLEEP_PRUNE_MAX_SCORE = 0.10
DEFAULT_RETRIEVE_NEGATIVE_POOL = 8
DEFAULT_RETRIEVE_MARKERS = 8
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "code", "column", "data",
    "def", "display", "draw", "each", "file", "for", "from", "function", "if",
    "in", "input", "list", "make", "number", "of", "on", "or", "output", "path",
    "plot", "return", "save", "self", "set", "should", "specified", "that", "the",
    "then", "this", "to", "using", "value", "with", "write", "you", "your",
}


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
    negative_episode_ids: list[str] = field(default_factory=list)
    positive_prototype: list[float] = field(default_factory=list)
    negative_prototype: list[float] = field(default_factory=list)
    structural_markers: list[str] = field(default_factory=list)
    top_contrast_directions: list[list[float]] = field(default_factory=list)
    explained_variance: list[float] = field(default_factory=list)
    local_support_gain: float = 0.0
    held_out_steering_gain: float = 0.0
    negative_steering_penalty: float = 0.0
    transfer_rate: float = 0.0
    transfer_gain: float = 0.0
    distillation_success: float = 0.0
    reuse_count: int = 0
    seen_help_count: int = 0
    seen_hurt_count: int = 0
    unseen_help_count: int = 0
    unseen_hurt_count: int = 0
    recency_score: float = 0.0
    utility_regression: float = 0.0
    redundancy_penalty: float = 0.0
    promotion_score: float = 0.0
    recent_success_rate: float = 0.0
    online_hurt_rate: float = 0.0
    q_value: float = 0.0
    retrieval_threshold: float = 0.0
    distilled_patch_ids: list[str] = field(default_factory=list)
    retrievable: bool = False
    created_at: float = field(default_factory=time.time)
    last_used_at: float = 0.0

    def __post_init__(self) -> None:
        # Keep legacy q_value metadata readable after the promotion/use split.
        if self.promotion_score <= 0.0 and self.q_value > 0.0:
            self.promotion_score = self.q_value
        elif self.q_value <= 0.0 and self.promotion_score > 0.0:
            self.q_value = self.promotion_score

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ClusterMemory":
        return cls(**raw)

    def retrievable_payload(self) -> dict[str, Any]:
        """Return the transfer-facing memory record used at retrieval time."""
        return {
            "cluster_metadata": {
                "memory_id": self.memory_id,
                "family_label": self.family_label,
                "support_count": self.support_count,
                "layer_window": list(self.layer_window),
                "token_window": self.token_window,
                "centroid_embedding": list(self.centroid_embedding),
                "positive_prototype": list(self.positive_prototype),
                "negative_prototype": list(self.negative_prototype),
                "retrieval_threshold": self.retrieval_threshold,
                "promotion_score": self.promotion_score,
                "q_value": self.q_value,
                "retrievable": self.retrievable,
            },
            "evidence": {
                "member_episode_ids": list(self.member_episode_ids),
                "negative_episode_ids": list(self.negative_episode_ids),
                "structural_markers": list(self.structural_markers),
                "top_contrast_directions": list(self.top_contrast_directions),
                "explained_variance": list(self.explained_variance),
            },
            "transfer_stats": {
                "local_support_gain": self.local_support_gain,
                "held_out_gain": self.held_out_steering_gain,
                "transfer_rate": self.transfer_rate,
                "transfer_gain": self.transfer_gain,
                "negative_steering_penalty": self.negative_steering_penalty,
                "seen_help_count": self.seen_help_count,
                "seen_hurt_count": self.seen_hurt_count,
                "unseen_help_count": self.unseen_help_count,
                "unseen_hurt_count": self.unseen_hurt_count,
                "recent_success_rate": self.recent_success_rate,
                "online_hurt_rate": self.online_hurt_rate,
                "redundancy_penalty": self.redundancy_penalty,
                "utility_regression": self.utility_regression,
            },
            "patch_ids": list(self.distilled_patch_ids),
        }


class ClusterMemoryBank:
    """Default retrieval surface: episodes -> cluster memories -> patch artifacts."""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.episodes: list[EpisodeRecord] = []
        self.memories: list[ClusterMemory] = []
        self._episode_index: dict[str, int] = {}
        self._memory_index: dict[str, int] = {}
        self._patch_bank = PatchBank(str(self.save_dir / "patch_artifacts"))
        self._build_executor: ThreadPoolExecutor | None = None
        self._build_future: Future | None = None
        self._build_lock = threading.RLock()

    def load(self) -> None:
        with self._build_lock:
            self._patch_bank.load()
            self.episodes = self._load_records("episodes.json", EpisodeRecord)
            self.memories = self._load_records("memories.json", ClusterMemory)
            self._episode_index = {ep.episode_id: i for i, ep in enumerate(self.episodes)}
            self._memory_index = {mem.memory_id: i for i, mem in enumerate(self.memories)}

    def save(self) -> None:
        with self._build_lock:
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
        with self._build_lock:
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
        with self._build_lock:
            eligible = [
                ep for ep in self.episodes
                if ep.success and ep.passed_code.strip() and ep.failed_code.strip()
            ]
            if not eligible:
                return self.stats()
            eligible_episode_ids = {episode.episode_id for episode in eligible}
            groups = _cluster_episodes(eligible, similarity_threshold, min_support)

            memories: list[ClusterMemory] = []
            for group in groups:
                layer_window = _resolve_layer_window(base_model, layer_window_size)
                centroid = np.mean(
                    np.asarray([ep.task_embedding for ep in group], dtype=np.float32),
                    axis=0,
                ).astype(np.float32)
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
                    centroid_embedding=centroid.tolist(),
                    member_episode_ids=[ep.episode_id for ep in group],
                    support_count=len(group),
                    layer_window=list(layer_window),
                    token_window=token_window,
                    negative_episode_ids=[],
                    positive_prototype=centroid.tolist(),
                    negative_prototype=[],
                    structural_markers=_extract_structural_markers(
                        [ep.prompt for ep in group],
                    ),
                    top_contrast_directions=[direction.tolist() for direction in directions],
                    explained_variance=explained,
                    created_at=max(ep.created_at for ep in group),
                )
                negative_episode_ids, negative_prototype = _compute_negative_prototype(
                        centroid,
                        group,
                        eligible,
                        limit=max(control_episodes, 1),
                    )
                memory.negative_episode_ids = negative_episode_ids
                memory.negative_prototype = negative_prototype
                _distill_and_score_memory(
                    memory,
                    group,
                    eligible,
                    self._patch_bank,
                    base_model,
                    tokenizer,
                    distill_rank=distill_rank,
                    distill_steps=distill_steps,
                    distill_lr=distill_lr,
                    control_episodes=control_episodes,
                )
                memories.append(memory)

            memories = self._merge_memories(memories, eligible_episode_ids)
            self.memories = memories
            self._memory_index = {mem.memory_id: i for i, mem in enumerate(self.memories)}
            return self.run_sleep_cycle(prune=True)

    def get_active_memories(
        self,
        query_embedding: list[float],
        task_prompt: str,
        top_k: int = 5,
    ) -> list[ClusterMemory]:
        if not self.memories:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        max_reuse = max((memory.reuse_count for memory in self.memories), default=0)
        scored: list[tuple[float, float, float, ClusterMemory]] = []

        for memory in self.memories:
            if not memory.retrievable or not memory.distilled_patch_ids:
                continue
            applicability = compute_applicability(
                memory,
                query,
                task_prompt,
            )
            if applicability <= memory.retrieval_threshold:
                continue
            use_score = score_memory_use(
                memory,
                query,
                task_prompt,
                max_reuse=max_reuse,
            )
            final_use = score_memory_final_use(
                memory,
                query,
                task_prompt,
                max_reuse=max_reuse,
            )
            scored.append((final_use, use_score, applicability, memory))

        scored.sort(key=lambda item: item[0], reverse=True)
        return _select_top_memories(scored, top_k=top_k)

    def get_active_patches(
        self,
        query_embedding: list[float],
        task_prompt: str,
        top_k: int = 5,
        return_memories: bool = False,
    ) -> list[CognitivePatch] | tuple[list[ClusterMemory], list[CognitivePatch]]:
        memories = self.get_active_memories(query_embedding, task_prompt, top_k=top_k)
        patches = self.load_patches_for_memories(
            memories,
            query_embedding=query_embedding,
            task_prompt=task_prompt,
        )
        if return_memories:
            return memories, patches
        return patches

    def load_patches_for_memories(
        self,
        memories: list[ClusterMemory],
        query_embedding: list[float] | None = None,
        task_prompt: str = "",
    ) -> list[CognitivePatch]:
        active: list[CognitivePatch] = []
        seen: set[str] = set()
        query = (
            np.asarray(query_embedding, dtype=np.float32)
            if query_embedding is not None
            else None
        )
        max_reuse = max((memory.reuse_count for memory in self.memories), default=0)
        for memory in memories:
            for patch_id in memory.distilled_patch_ids:
                if patch_id in seen:
                    continue
                patch = self._patch_bank.get_patch(patch_id)
                if patch is None:
                    continue
                self._patch_bank.load_weights(patch)
                if query is not None:
                    patch.q_value = score_memory_final_use(
                        memory,
                        query,
                        task_prompt,
                        max_reuse=max_reuse,
                    )
                else:
                    patch.q_value = memory.promotion_score
                active.append(patch)
                seen.add(patch_id)
        return active

    def update_memory_utility(
        self,
        memory_id: str,
        task_succeeded: bool,
        cold_succeeded: bool | None = None,
        eval_split: str = "",
        persist: bool = True,
    ) -> None:
        with self._build_lock:
            idx = self._memory_index.get(memory_id)
            if idx is None:
                return

            memory = self.memories[idx]
            memory.reuse_count += 1
            memory.last_used_at = time.time()
            observation = 1.0 if task_succeeded else 0.0
            memory.recent_success_rate = float(
                np.clip(0.8 * memory.recent_success_rate + 0.2 * observation, 0.0, 1.0)
            )
            memory.online_hurt_rate = float(
                np.clip(0.8 * memory.online_hurt_rate + 0.2 * (1.0 - observation), 0.0, 1.0)
            )

            if cold_succeeded is True and not task_succeeded:
                memory.utility_regression = min(1.0, memory.utility_regression + 0.1)
                if eval_split == "seen":
                    memory.seen_hurt_count += 1
                elif eval_split == "unseen":
                    memory.unseen_hurt_count += 1
            elif cold_succeeded is False and task_succeeded:
                memory.utility_regression = max(0.0, memory.utility_regression - 0.05)
                if eval_split == "seen":
                    memory.seen_help_count += 1
                elif eval_split == "unseen":
                    memory.unseen_help_count += 1

            _apply_recency_scores(self.memories)
            _recompute_q_values(self.memories)
            self.memories = _apply_sleep_promotion_policies(self.memories, prune=False)
            self._memory_index = {mem.memory_id: i for i, mem in enumerate(self.memories)}
            eligible = [
                episode for episode in self.episodes
                if episode.success and episode.passed_code.strip() and episode.failed_code.strip()
            ]
            if eligible:
                _apply_retrieval_thresholds(self.memories, eligible)
            if persist:
                self.save()

    def run_sleep_cycle(self, prune: bool = True) -> dict[str, Any]:
        """Recompute promotion trust, demote harmful memories, and refresh gates."""
        with self._build_lock:
            _apply_recency_scores(self.memories)
            _apply_redundancy_penalties(self.memories)
            _recompute_q_values(self.memories)
            self.memories = _apply_sleep_promotion_policies(self.memories, prune=prune)
            eligible = [
                episode for episode in self.episodes
                if episode.success and episode.passed_code.strip() and episode.failed_code.strip()
            ]
            if eligible:
                _apply_retrieval_thresholds(self.memories, eligible)
            self._memory_index = {mem.memory_id: i for i, mem in enumerate(self.memories)}
            self.save()
            return self.stats()

    def schedule_build_memories(
        self,
        base_model,
        tokenizer,
        **build_kwargs,
    ) -> Future:
        """Schedule ``build_memories`` on a background worker."""
        with self._build_lock:
            if self._build_executor is None:
                self._build_executor = ThreadPoolExecutor(max_workers=1)
            self._build_future = self._build_executor.submit(
                self.build_memories,
                base_model,
                tokenizer,
                **build_kwargs,
            )
            return self._build_future

    def flush_pending_build(self) -> None:
        """Wait for and surface any pending asynchronous build result."""
        with self._build_lock:
            if self._build_future is None:
                return
            future = self._build_future
            self._build_future = None
        future.result()

    def stats(self) -> dict[str, Any]:
        family_counts: dict[str, int] = {}
        for memory in self.memories:
            family_counts[memory.family_label] = family_counts.get(memory.family_label, 0) + 1
        return {
            "episodes": len(self.episodes),
            "memories": len(self.memories),
            "retrievable_memories": sum(1 for memory in self.memories if memory.retrievable),
            "artifact_patches": len(self._patch_bank.patches),
            "mean_promotion": (
                float(np.mean([memory.promotion_score for memory in self.memories]))
                if self.memories else 0.0
            ),
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

    def _merge_memories(
        self,
        new_memories: list[ClusterMemory],
        eligible_episode_ids: set[str],
    ) -> list[ClusterMemory]:
        """Merge freshly built memories with historical ones instead of replacing them."""
        existing_by_id = {memory.memory_id: memory for memory in self.memories}
        existing_by_members = {
            frozenset(memory.member_episode_ids): memory
            for memory in self.memories
        }
        merged: list[ClusterMemory] = []
        retained_ids: set[str] = set()

        for memory in new_memories:
            existing = existing_by_id.get(memory.memory_id)
            if existing is None:
                existing = existing_by_members.get(frozenset(memory.member_episode_ids))
            if existing is not None:
                memory.promotion_score = existing.promotion_score
                memory.q_value = existing.q_value
                memory.reuse_count = existing.reuse_count
                memory.last_used_at = existing.last_used_at
                memory.seen_help_count = existing.seen_help_count
                memory.seen_hurt_count = existing.seen_hurt_count
                memory.unseen_help_count = existing.unseen_help_count
                memory.unseen_hurt_count = existing.unseen_hurt_count
                memory.recent_success_rate = existing.recent_success_rate
                memory.online_hurt_rate = existing.online_hurt_rate
                memory.utility_regression = existing.utility_regression
                memory.created_at = existing.created_at
            merged.append(memory)
            retained_ids.add(memory.memory_id)

        for existing in self.memories:
            if existing.memory_id in retained_ids:
                continue
            if set(existing.member_episode_ids).issubset(eligible_episode_ids):
                merged.append(existing)

        return merged


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


def _prompt_feature_tokens(prompt: str) -> list[str]:
    tokens = re.findall(r"[a-z_][a-z0-9_]{2,}", prompt.lower())
    return [token for token in tokens if token not in STOPWORDS]


def _extract_structural_markers(
    prompts: list[str],
    max_markers: int = DEFAULT_RETRIEVE_MARKERS,
) -> list[str]:
    counts: dict[str, int] = {}
    for prompt in prompts:
        for token in set(_prompt_feature_tokens(prompt)):
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [token for token, _ in ranked[:max_markers]]


def compute_structural_match(task_prompt: str, memory: ClusterMemory) -> float:
    if not memory.structural_markers:
        return 0.0
    query_tokens = set(_prompt_feature_tokens(task_prompt))
    marker_hits = sum(1 for marker in memory.structural_markers if marker in query_tokens)
    overlap = marker_hits / max(len(memory.structural_markers), 1)
    family_match = 1.0 if derive_family_label(task_prompt) == memory.family_label else 0.0
    return float(np.clip(0.7 * overlap + 0.3 * family_match, 0.0, 1.0))


def compute_applicability(
    memory: ClusterMemory,
    query_embedding: np.ndarray,
    task_prompt: str,
) -> float:
    positive = np.asarray(
        memory.positive_prototype or memory.centroid_embedding,
        dtype=np.float32,
    )
    positive_similarity = cosine_similarity(query_embedding, positive)

    negative_similarity = 0.0
    if memory.negative_prototype:
        negative = np.asarray(memory.negative_prototype, dtype=np.float32)
        negative_similarity = cosine_similarity(query_embedding, negative)

    structural_match = compute_structural_match(task_prompt, memory)
    applicability = (
        DEFAULT_APPLICABILITY_POS_WEIGHT * positive_similarity
        - DEFAULT_APPLICABILITY_NEG_WEIGHT * negative_similarity
        + DEFAULT_APPLICABILITY_STRUCT_WEIGHT * structural_match
    )
    return float(np.clip(applicability, 0.0, 1.0))


def _normalized_log_score(value: int, max_value: int) -> float:
    if value <= 0 or max_value <= 0:
        return 0.0
    return float(np.log1p(value) / max(np.log1p(max_value), 1e-8))


def score_memory_promotion(
    memory: ClusterMemory,
    max_support: int,
) -> float:
    total_transfer_outcomes = (
        memory.seen_help_count
        + memory.seen_hurt_count
        + memory.unseen_help_count
        + memory.unseen_hurt_count
    )
    unseen_hurt_rate = (
        memory.unseen_hurt_count / total_transfer_outcomes
        if total_transfer_outcomes
        else 0.0
    )
    support_score = _normalized_log_score(memory.support_count, max_support)
    q = (
        0.28 * float(np.clip(memory.held_out_steering_gain, 0.0, 1.0))
        + 0.22 * float(np.clip(memory.transfer_gain, 0.0, 1.0))
        + 0.12 * float(np.clip(memory.local_support_gain, 0.0, 1.0))
        + 0.10 * float(np.clip(memory.distillation_success, 0.0, 1.0))
        + 0.08 * support_score
        + 0.10 * float(np.clip(memory.recent_success_rate, 0.0, 1.0))
        - 0.10 * float(np.clip(memory.online_hurt_rate, 0.0, 1.0))
        - 0.15 * float(np.clip(memory.utility_regression, 0.0, 1.0))
        - 0.15 * float(np.clip(unseen_hurt_rate, 0.0, 1.0))
        - 0.10 * float(np.clip(memory.redundancy_penalty, 0.0, 1.0))
    )
    return float(np.clip(q, 0.0, 1.0))


def score_memory_use(
    memory: ClusterMemory,
    query_embedding: np.ndarray,
    task_prompt: str,
    *,
    max_reuse: int = 0,
) -> float:
    applicability = compute_applicability(memory, query_embedding, task_prompt)
    reuse_score = _normalized_log_score(memory.reuse_count, max_reuse)
    recent_success = float(
        np.clip(
            memory.recent_success_rate
            if (memory.reuse_count > 0 or memory.recent_success_rate > 0.0)
            else memory.transfer_gain,
            0.0,
            1.0,
        )
    )
    trust_now = (
        DEFAULT_USE_TRANSFER_WEIGHT * float(np.clip(memory.transfer_gain, 0.0, 1.0))
        + DEFAULT_USE_RECENT_SUCCESS_WEIGHT * recent_success
        + DEFAULT_USE_REUSE_WEIGHT * reuse_score
        - DEFAULT_USE_ONLINE_HURT_WEIGHT * float(np.clip(memory.online_hurt_rate, 0.0, 1.0))
    )
    trust_now = float(np.clip(trust_now, 0.0, 1.0))
    return float(np.clip(applicability * trust_now, 0.0, 1.0))


def score_memory_final_use(
    memory: ClusterMemory,
    query_embedding: np.ndarray,
    task_prompt: str,
    *,
    max_reuse: int = 0,
) -> float:
    use_score = score_memory_use(
        memory,
        query_embedding,
        task_prompt,
        max_reuse=max_reuse,
    )
    return float(
        np.clip(
            use_score + DEFAULT_FINAL_USE_PROMOTE_WEIGHT * memory.promotion_score,
            0.0,
            1.0,
        )
    )


def score_memory_retrieval(
    memory: ClusterMemory,
    query_embedding: np.ndarray,
    task_prompt: str,
) -> float:
    """Backward-compatible alias for the final retrieval-time ranking score."""
    return score_memory_final_use(memory, query_embedding, task_prompt, max_reuse=memory.reuse_count)


def _select_top_memories(
    scored: list[tuple[float, float, float, ClusterMemory]],
    top_k: int,
) -> list[ClusterMemory]:
    if not scored:
        return []
    if top_k <= 1 or len(scored) == 1:
        return [scored[0][3]]

    top_final_use = scored[0][0]
    if top_final_use < DEFAULT_MULTI_MEMORY_CONFIDENCE:
        return [scored[0][3]]

    selected: list[ClusterMemory] = [scored[0][3]]
    for final_use, _, _, memory in scored[1:]:
        if len(selected) >= top_k:
            break
        if final_use < DEFAULT_MULTI_MEMORY_CONFIDENCE:
            break
        if (top_final_use - final_use) > DEFAULT_MULTI_MEMORY_MARGIN:
            break
        selected.append(memory)
    return selected


def _compute_negative_prototype(
    centroid: np.ndarray,
    group: list[EpisodeRecord],
    all_episodes: list[EpisodeRecord],
    limit: int,
) -> tuple[list[str], list[float]]:
    member_ids = {episode.episode_id for episode in group}
    candidates = [
        episode for episode in all_episodes
        if episode.episode_id not in member_ids
    ]
    if not candidates:
        return [], []
    ranked = sorted(
        candidates,
        key=lambda episode: cosine_similarity(
            centroid,
            np.asarray(episode.task_embedding, dtype=np.float32),
        ),
        reverse=True,
    )
    selected = ranked[:max(limit, 1)]
    prototype = np.mean(
        np.asarray([episode.task_embedding for episode in selected], dtype=np.float32),
        axis=0,
    )
    return [episode.episode_id for episode in selected], prototype.astype(np.float32).tolist()


def _apply_retrieval_thresholds(
    memories: list[ClusterMemory],
    episodes: list[EpisodeRecord],
) -> None:
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    for memory in memories:
        positive_examples = [
            episode_by_id[episode_id]
            for episode_id in memory.member_episode_ids
            if episode_id in episode_by_id
        ]
        if not positive_examples:
            memory.retrieval_threshold = 1.0
            memory.retrievable = False
            continue

        positive_scores = [
            compute_applicability(
                memory,
                np.asarray(episode.task_embedding, dtype=np.float32),
                episode.prompt,
            )
            for episode in positive_examples
        ]

        negative_pool = [
            episode for episode in episodes
            if episode.episode_id not in set(memory.member_episode_ids)
        ]
        negative_scores = sorted(
            (
                compute_applicability(
                    memory,
                    np.asarray(episode.task_embedding, dtype=np.float32),
                    episode.prompt,
                )
                for episode in negative_pool
            ),
            reverse=True,
        )[:DEFAULT_RETRIEVE_NEGATIVE_POOL]

        positive_anchor = float(np.percentile(positive_scores, 25))
        negative_anchor = float(np.percentile(negative_scores, 85)) if negative_scores else 0.0
        midpoint = (positive_anchor + negative_anchor) / 2.0
        memory.retrieval_threshold = float(np.clip(midpoint, -1.0, 1.0))
        if memory.retrieval_threshold >= max(positive_scores):
            memory.retrieval_threshold = float(max(positive_scores) - 1e-4)


def _cluster_episodes(
    episodes: list[EpisodeRecord],
    similarity_threshold: float,
    min_support: int,
) -> list[list[EpisodeRecord]]:
    """Cluster episodes with single-link connected components inside each family.

    This is intentionally transitive: if A~B and B~C clear ``similarity_threshold``,
    all three can land in the same cluster even when A and C do not. ``min_support``
    is applied to the final connected component size, not to all-pairs cliques.
    """
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
    """Pool hidden states from standard Hugging Face decoder models.

    Assumes ``output_hidden_states=True`` returns ``hidden_states`` where
    ``hidden_states[0]`` is the embedding output and ``hidden_states[1..]`` are
    transformer layer outputs, which is why this code indexes
    ``hidden_states[layer_idx + 1]``. This also assumes ``model.config`` exposes
    ``num_hidden_layers``; non-standard models may require different indexing or
    an explicit layer count.
    """
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
    local_gains = [
        _measure_patch_teacher_forcing_gain(
            base_model,
            tokenizer,
            patch,
            episode.prompt,
            episode.passed_code,
        )
        for episode in train_eps
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

    memory.local_support_gain = float(np.mean(local_gains)) if local_gains else 0.0
    memory.held_out_steering_gain = float(np.mean(holdout_gains)) if holdout_gains else 0.0
    memory.transfer_rate = float(np.mean([gain > 0 for gain in holdout_gains])) if holdout_gains else 0.0
    positive_holdout = [max(gain, 0.0) for gain in holdout_gains]
    memory.transfer_gain = float(np.mean(positive_holdout)) if positive_holdout else 0.0
    harms = [max(-gain, 0.0) for gain in control_gains]
    memory.negative_steering_penalty = float(np.mean(harms)) if harms else 0.0
    memory.distillation_success = 1.0 if (
        memory.local_support_gain > 0
        and memory.held_out_steering_gain > 0
        and memory.transfer_rate >= 0.5
        and memory.negative_steering_penalty <= 0.1
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
    max_support = max((memory.support_count for memory in memories), default=0)
    for memory in memories:
        total_transfer_outcomes = (
            memory.seen_help_count
            + memory.seen_hurt_count
            + memory.unseen_help_count
            + memory.unseen_hurt_count
        )
        positive_transfer_outcomes = memory.seen_help_count + memory.unseen_help_count
        negative_transfer_outcomes = memory.seen_hurt_count + memory.unseen_hurt_count
        if total_transfer_outcomes:
            memory.transfer_gain = positive_transfer_outcomes / total_transfer_outcomes
        else:
            memory.transfer_gain = float(np.clip(memory.transfer_gain, 0.0, 1.0))
        memory.promotion_score = score_memory_promotion(memory, max_support=max_support)
        memory.q_value = memory.promotion_score


def _apply_sleep_promotion_policies(
    memories: list[ClusterMemory],
    *,
    prune: bool,
) -> list[ClusterMemory]:
    retained: list[ClusterMemory] = []
    for memory in memories:
        has_patch = bool(memory.distilled_patch_ids) and memory.distillation_success > 0
        harmful_transfer = (
            memory.unseen_hurt_count >= 2
            and memory.unseen_hurt_count > memory.unseen_help_count
        )
        preserve_harm = (
            memory.online_hurt_rate >= 0.60
            or memory.utility_regression >= 0.35
            or harmful_transfer
        )
        should_promote = (
            has_patch
            and memory.promotion_score >= DEFAULT_SLEEP_PROMOTE_MIN_SCORE
            and memory.support_count >= DEFAULT_SLEEP_PROMOTE_MIN_SUPPORT
            and not preserve_harm
        )
        memory.retrievable = should_promote

        should_prune = (
            prune
            and memory.promotion_score <= DEFAULT_SLEEP_PRUNE_MAX_SCORE
            and memory.support_count < DEFAULT_SLEEP_PROMOTE_MIN_SUPPORT
            and preserve_harm
        )
        if should_prune:
            continue
        retained.append(memory)
    return retained


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

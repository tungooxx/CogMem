"""Package runners for the research-only patch experiment flow.

This module extracts the orchestration that previously lived in
``paperspace_patches.ipynb`` into reusable package functions. The notebook can
remain as a thin UI, but the experiment logic should live here.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cogmem.benchmarks.bigcodebench.dataset import (
    load_bigcodebench,
    load_bigcodebench_from_jsonl,
)
from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution
from cogmem.benchmarks.bigcodebench.experiment import (
    build_eval_cache_path,
    load_eval_cache,
    save_eval_cache,
)
from cogmem.benchmarks.bigcodebench.prompts import SYSTEM_PROMPT, extract_code

if TYPE_CHECKING:
    from cogmem.patches.memory_bank import ClusterMemoryBank


DEFAULT_PATCH_SCALE = 0.25


def _new_memory_bank(save_dir: str):
    from cogmem.patches.memory_bank import ClusterMemoryBank

    return ClusterMemoryBank(save_dir)


def _generate_many(*args, **kwargs):
    from cogmem.patches.wake import generate_many_with_model

    return generate_many_with_model(*args, **kwargs)


def _generate_one(*args, **kwargs):
    from cogmem.patches.wake import generate_with_model

    return generate_with_model(*args, **kwargs)


def _best_contrast_pair(*args, **kwargs):
    from cogmem.patches.wake import find_best_contrast_pair

    return find_best_contrast_pair(*args, **kwargs)


def _patched_model(*args, **kwargs):
    from cogmem.patches.compose import PatchedModel

    return PatchedModel(*args, **kwargs)


def _compute_applicability(*args, **kwargs):
    from cogmem.patches.memory_bank import compute_applicability

    return compute_applicability(*args, **kwargs)


def _score_memory_use(*args, **kwargs):
    from cogmem.patches.memory_bank import score_memory_use

    return score_memory_use(*args, **kwargs)


def _score_memory_final_use(*args, **kwargs):
    from cogmem.patches.memory_bank import score_memory_final_use

    return score_memory_final_use(*args, **kwargs)


def _score_memory_promotion(*args, **kwargs):
    from cogmem.patches.memory_bank import score_memory_promotion

    return score_memory_promotion(*args, **kwargs)


@dataclass
class PatchExperimentConfig:
    memory_dir: str = "results/cluster_memories"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    embedder_name: str = "all-MiniLM-L6-v2"
    train_task_count: int = 500
    n_candidates: int = 4
    collect_temperature: float = 0.8
    eval_timeout: int = 30
    episode_progress_filename: str = "episode_recording_progress.json"
    save_every_tasks: int = 1
    cluster_similarity_threshold: float = 0.62
    cluster_min_support: int = 3
    cluster_control_episodes: int = 6
    inspect_unseen_size: int = 200
    sweep_diag_size: int = 30
    sweep_topk_options: tuple[int, ...] = (1, 2, 5)
    eval_top_k: int = 5
    eval_scale: float = DEFAULT_PATCH_SCALE
    eval_cache_version: str = "finaluse_v4"
    unseen_eval_size: int = 50


def _task_prompt(task: dict) -> str:
    return task.get("instruct_prompt", task.get("complete_prompt", ""))


def _to_embedding_list(embedding) -> list[float]:
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return list(embedding)


def _memory_rows(memory_bank: "ClusterMemoryBank", *, limit: int = 10) -> list[dict]:
    rows = []
    max_support = max((memory.support_count for memory in memory_bank.memories), default=0)
    for memory in memory_bank.memories[:limit]:
        rows.append({
            "memory_id": memory.memory_id,
            "family": memory.family_label,
            "support_count": memory.support_count,
            "promotion_score": memory.promotion_score,
            "threshold": memory.retrieval_threshold,
            "retrievable": memory.retrievable,
            "local_gain": memory.local_support_gain,
            "heldout_gain": memory.held_out_steering_gain,
            "transfer_gain": memory.transfer_gain,
            "transfer_online_gain": memory.transfer_online_gain,
            "transfer_rate": memory.transfer_rate,
            "recent_success_rate": memory.recent_success_rate,
            "online_hurt_rate": memory.online_hurt_rate,
            "utility_regression": memory.utility_regression,
            "redundancy_penalty": memory.redundancy_penalty,
            "negative_steering_penalty": memory.negative_steering_penalty,
            "negative_count": len(memory.negative_episode_ids),
            "structural_markers": list(memory.structural_markers[:5]),
            "patch_ids": list(memory.retrievable_payload()["patch_ids"]),
            "q_promote": _score_memory_promotion(memory, max_support=max_support),
        })
    return rows


def prepare_patch_task_split(tasks: list[dict], train_task_count: int) -> tuple[list[dict], list[dict]]:
    return tasks[:train_task_count], tasks[train_task_count:]


def load_patch_tasks(
    *,
    task_jsonl_path: str | None = None,
    version: str = "v0.1.4",
    hard_only: bool = False,
) -> list[dict]:
    if task_jsonl_path:
        return load_bigcodebench_from_jsonl(task_jsonl_path)
    return load_bigcodebench(version=version, hard_only=hard_only)


def load_patch_runtime(
    *,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    embedder_name: str = "all-MiniLM-L6-v2",
    prepare_for_training: bool = True,
):
    import torch
    from peft import prepare_model_for_kbit_training
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    if prepare_for_training:
        base_model = prepare_model_for_kbit_training(base_model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedder = SentenceTransformer(embedder_name, device="cpu")
    return base_model, tokenizer, embedder


def run_patch_episode_recording(
    train_tasks: list[dict],
    base_model,
    tokenizer,
    embedder,
    *,
    config: PatchExperimentConfig | None = None,
    memory_bank: "ClusterMemoryBank" | None = None,
    reset_progress: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    config = config or PatchExperimentConfig()
    bank = memory_bank or _new_memory_bank(config.memory_dir)
    bank.load()

    progress_path = Path(config.memory_dir) / config.episode_progress_filename
    train_signature = {
        "count": len(train_tasks),
        "first_task_id": train_tasks[0]["task_id"] if train_tasks else "",
        "last_task_id": train_tasks[-1]["task_id"] if train_tasks else "",
    }

    if reset_progress and progress_path.exists():
        progress_path.unlink()

    resume_state: dict[str, Any] = {}
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            resume_state = json.load(f)

    same_train_plan = resume_state.get("train_signature") == train_signature
    start_idx = int(resume_state.get("next_index", 0)) if same_train_plan else 0
    total_passed = int(resume_state.get("total_passed", 0)) if same_train_plan else 0
    episodes_before = int(resume_state.get("episodes_before", len(bank.episodes))) if same_train_plan else len(bank.episodes)
    start_time = time.time()

    if verbose:
        print("Resume progress file:", progress_path)
        if start_idx > 0:
            print(f"Resuming Cell 4 from task {start_idx + 1} of {len(train_tasks)}.")
        else:
            print(f"Starting Cell 4 from task 1 of {len(train_tasks)}.")
        if start_idx >= len(train_tasks):
            print("Cell 4 already completed for the current TRAIN_TASKS selection.")

    for i in range(start_idx, len(train_tasks)):
        task = train_tasks[i]
        prompt = _task_prompt(task)
        task_embedding = _to_embedding_list(embedder.encode(prompt))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        candidates = []
        try:
            responses = _generate_many(
                base_model,
                tokenizer,
                messages,
                n_candidates=config.n_candidates,
                temperature=config.collect_temperature,
            )
            for response in responses:
                code = extract_code(response)
                if code and len(code.strip()) > 20:
                    result = evaluate_solution(task, code, timeout=config.eval_timeout, mode="subprocess")
                    candidates.append({"code": code, "passed": result["passed"]})
        except Exception:
            candidates = []

        passes = [candidate for candidate in candidates if candidate["passed"]]
        fails = [candidate for candidate in candidates if not candidate["passed"]]

        if passes:
            total_passed += 1

        if passes and fails:
            best_pair, best_similarity = _best_contrast_pair(passes, fails)
            if best_pair is not None:
                bank.record_episode(
                    task_id=task["task_id"],
                    prompt=prompt,
                    task_embedding=task_embedding,
                    failed_code=best_pair["fail"]["code"],
                    passed_code=best_pair["pass"]["code"],
                    pass_fail_similarity=best_similarity,
                )

        if config.save_every_tasks <= 1 or (i + 1) % config.save_every_tasks == 0:
            bank.save()

        progress_state = {
            "status": "running",
            "train_signature": train_signature,
            "next_index": i + 1,
            "last_task_id": task["task_id"],
            "total_passed": total_passed,
            "episodes_before": episodes_before,
            "episodes_now": len(bank.episodes),
            "updated_at": time.time(),
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress_state, f, indent=2)

        if verbose and ((i + 1) % 10 == 0 or i < 5):
            elapsed = time.time() - start_time
            processed_this_run = max(i + 1 - start_idx, 1)
            rate = processed_this_run / elapsed * 3600 if elapsed > 0 else 0.0
            print(
                "[{}/{}] {} | episodes={} | passes={} | {:.0f}/hr".format(
                    i + 1,
                    len(train_tasks),
                    task["task_id"],
                    len(bank.episodes),
                    total_passed,
                    rate,
                )
            )

    bank.save()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({
            "status": "complete",
            "train_signature": train_signature,
            "next_index": len(train_tasks),
            "last_task_id": train_tasks[-1]["task_id"] if train_tasks else "",
            "total_passed": total_passed,
            "episodes_before": episodes_before,
            "episodes_now": len(bank.episodes),
            "updated_at": time.time(),
        }, f, indent=2)

    elapsed_minutes = (time.time() - start_time) / 60.0
    return {
        "tasks_processed": len(train_tasks),
        "tasks_with_passes": total_passed,
        "new_episodes": len(bank.episodes) - episodes_before,
        "episodes_total": len(bank.episodes),
        "elapsed_minutes": elapsed_minutes,
        "progress_path": str(progress_path),
        "stats": bank.stats(),
    }


def build_patch_memories(
    base_model,
    tokenizer,
    memory_bank: "ClusterMemoryBank",
    *,
    config: PatchExperimentConfig | None = None,
    eval_tasks: list[dict] | None = None,
    embedder=None,
    summary_limit: int = 10,
    sample_tasks: int = 3,
) -> dict[str, Any]:
    config = config or PatchExperimentConfig()
    if not memory_bank.episodes:
        raise RuntimeError("No episodes found. Run episode recording before building memories.")
    import torch

    build_stats = memory_bank.build_memories(
        base_model,
        tokenizer,
        similarity_threshold=config.cluster_similarity_threshold,
        min_support=config.cluster_min_support,
        control_episodes=config.cluster_control_episodes,
    )
    retrievable = [memory for memory in memory_bank.memories if memory.retrievable]
    artifact_rows = []
    for memory in retrievable[:3]:
        loaded = memory_bank.load_patches_for_memories([memory])
        if not loaded:
            artifact_rows.append({
                "memory_id": memory.memory_id,
                "patch_id": None,
                "avg_norm": 0.0,
            })
            continue
        patch = loaded[0]
        norms = [torch.norm(weights["A"]).item() + torch.norm(weights["B"]).item() for weights in patch.lora_weights.values()]
        first_key = list(patch.lora_weights.keys())[0]
        weights = patch.lora_weights[first_key]
        artifact_rows.append({
            "memory_id": memory.memory_id,
            "patch_id": patch.patch_id,
            "norm_a": torch.norm(weights["A"]).item(),
            "norm_b": torch.norm(weights["B"]).item(),
            "avg_norm": sum(norms) / len(norms) if norms else 0.0,
        })
        patch.unload_weights()

    output_rows = []
    if eval_tasks and embedder is not None:
        max_support = max((memory.support_count for memory in memory_bank.memories), default=0)
        max_reuse = max((memory.reuse_count for memory in memory_bank.memories), default=0)
        for task in eval_tasks[:sample_tasks]:
            prompt = _task_prompt(task)
            task_embedding = _to_embedding_list(embedder.encode(prompt))
            task_array = np.asarray(task_embedding, dtype=np.float32)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            out_cold = _generate_one(base_model, tokenizer, messages, temperature=0)
            active_memories, active_patches = memory_bank.get_active_patches(
                task_embedding,
                prompt,
                top_k=config.eval_top_k,
                return_memories=True,
            )
            try:
                with _patched_model(base_model, active_patches, scaling_factor=config.eval_scale):
                    out_patched = _generate_one(base_model, tokenizer, messages, temperature=0)
            finally:
                for patch in active_patches:
                    patch.unload_weights()

            if not active_memories:
                output_rows.append({"task_id": task["task_id"], "abstained": True})
                continue

            best_memory = active_memories[0]
            cold_tokens = out_cold.split()
            patched_tokens = out_patched.split()
            diff = sum(1 for left, right in zip(cold_tokens, patched_tokens) if left != right)
            total = max(len(cold_tokens), len(patched_tokens), 1)
            output_rows.append({
                "task_id": task["task_id"],
                "abstained": False,
                "memory_id": best_memory.memory_id,
                "applicability": _compute_applicability(best_memory, task_array, prompt),
                "q_use": _score_memory_use(best_memory, task_array, prompt, max_reuse=max_reuse),
                "final_use": _score_memory_final_use(best_memory, task_array, prompt, max_reuse=max_reuse),
                "q_promote": _score_memory_promotion(best_memory, max_support=max_support),
                "threshold": best_memory.retrieval_threshold,
                "token_diff_rate": diff / total,
            })

    return {
        "build_stats": build_stats,
        "retrievable_memories": len(retrievable),
        "memory_rows": _memory_rows(memory_bank, limit=summary_limit),
        "artifact_rows": artifact_rows,
        "output_rows": output_rows,
    }


def inspect_unseen_retrieval(
    eval_tasks: list[dict],
    memory_bank: "ClusterMemoryBank",
    embedder,
    *,
    config: PatchExperimentConfig | None = None,
) -> dict[str, Any]:
    config = config or PatchExperimentConfig()
    unseen_subset = eval_tasks[: config.inspect_unseen_size]
    retrievable_memories = [memory for memory in memory_bank.memories if memory.retrievable]
    max_reuse = max((memory.reuse_count for memory in memory_bank.memories), default=0)
    rows = []
    top1_final_use = []
    top1_use = []
    top1_applicability = []
    margins = []
    abstained = 0
    memory_hits = Counter()
    family_hits = Counter()

    for task in unseen_subset:
        prompt = _task_prompt(task)
        task_embedding = np.asarray(_to_embedding_list(embedder.encode(prompt)), dtype=np.float32)
        scored = []
        for memory in retrievable_memories:
            applicability = _compute_applicability(memory, task_embedding, prompt)
            if applicability <= memory.retrieval_threshold:
                continue
            use_score = _score_memory_use(memory, task_embedding, prompt, max_reuse=max_reuse)
            final_use = _score_memory_final_use(memory, task_embedding, prompt, max_reuse=max_reuse)
            scored.append((final_use, use_score, applicability, memory))

        if not scored:
            abstained += 1
            rows.append({
                "task_id": task["task_id"],
                "final_use": 0.0,
                "q_use": 0.0,
                "applicability": 0.0,
                "memory_id": "ABSTAIN",
                "family": "",
                "selected": False,
            })
            continue

        scored.sort(key=lambda item: item[0], reverse=True)
        best_final_use, best_use, best_applicability, best_memory = scored[0]
        top1_final_use.append(best_final_use)
        top1_use.append(best_use)
        top1_applicability.append(best_applicability)
        margins.append(best_applicability - best_memory.retrieval_threshold)
        memory_hits[best_memory.memory_id] += 1
        family_hits[best_memory.family_label] += 1
        rows.append({
            "task_id": task["task_id"],
            "final_use": best_final_use,
            "q_use": best_use,
            "applicability": best_applicability,
            "memory_id": best_memory.memory_id,
            "family": best_memory.family_label,
            "selected": True,
        })

    return {
        "unseen_tasks_inspected": len(unseen_subset),
        "retrievable_memories": len(retrievable_memories),
        "mean_top1_final_use": float(np.mean(top1_final_use)) if top1_final_use else 0.0,
        "mean_top1_q_use": float(np.mean(top1_use)) if top1_use else 0.0,
        "mean_top1_applicability": float(np.mean(top1_applicability)) if top1_applicability else 0.0,
        "mean_applicability_margin": float(np.mean(margins)) if margins else 0.0,
        "abstentions": abstained,
        "abstention_rate": abstained / max(len(unseen_subset), 1),
        "memory_hits": dict(memory_hits),
        "family_hits": dict(family_hits),
        "rows": rows,
    }


def sweep_patch_retrieval_width(
    seen_tasks: list[dict],
    base_model,
    tokenizer,
    memory_bank: "ClusterMemoryBank",
    embedder,
    *,
    config: PatchExperimentConfig | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    config = config or PatchExperimentConfig()
    seen_subset = seen_tasks[: config.sweep_diag_size]
    cached = []
    if verbose:
        print(f"Running gated retrieval sweep on {len(seen_subset)} seen tasks")
    for task in seen_subset:
        prompt = _task_prompt(task)
        task_embedding = _to_embedding_list(embedder.encode(prompt))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        cold_ok = False
        try:
            cold_response = _generate_one(base_model, tokenizer, messages, temperature=0)
            cold_code = extract_code(cold_response)
            cold_result = evaluate_solution(task, cold_code, timeout=config.eval_timeout, mode="subprocess")
            cold_ok = bool(cold_result["passed"])
        except Exception:
            cold_ok = False

        cached.append({
            "task": task,
            "prompt": prompt,
            "embedding": task_embedding,
            "messages": messages,
            "cold_ok": cold_ok,
        })
        if verbose and (len(cached) % 10 == 0 or len(cached) == len(seen_subset)):
            cold_passed = sum(1 for row in cached if row["cold_ok"])
            print(f"  cached cold [{len(cached)}/{len(seen_subset)}] cold={cold_passed}", flush=True)

    cold_passed = sum(1 for row in cached if row["cold_ok"])
    results = []
    for top_k in config.sweep_topk_options:
        if verbose:
            print()
            print(f"-- top_k={top_k}, scale={config.eval_scale} --", flush=True)
        memory_passed = 0
        helped = 0
        hurt = 0
        abstained = 0

        for i, row in enumerate(cached):
            mem_ok = row["cold_ok"]
            active_patches = []
            try:
                active_memories, active_patches = memory_bank.get_active_patches(
                    row["embedding"],
                    row["prompt"],
                    top_k=top_k,
                    return_memories=True,
                )
                if not active_patches:
                    abstained += 1
                else:
                    with _patched_model(base_model, active_patches, scaling_factor=config.eval_scale):
                        mem_response = _generate_one(base_model, tokenizer, row["messages"], temperature=0)
                    mem_code = extract_code(mem_response)
                    mem_result = evaluate_solution(row["task"], mem_code, timeout=config.eval_timeout, mode="subprocess")
                    mem_ok = bool(mem_result["passed"])
            except Exception:
                mem_ok = False
            finally:
                for patch in active_patches:
                    patch.unload_weights()

            if mem_ok:
                memory_passed += 1
            if (not row["cold_ok"]) and mem_ok:
                helped += 1
            elif row["cold_ok"] and (not mem_ok):
                hurt += 1

            if verbose and ((i + 1) % 10 == 0 or i + 1 == len(cached)):
                print(
                    f"  [{i+1}/{len(cached)}] memory={memory_passed} helped={helped} "
                    f"hurt={hurt} abstain={abstained}",
                    flush=True,
                )

        results.append({
            "top_k": top_k,
            "cold_passed": cold_passed,
            "memory_passed": memory_passed,
            "cold_rate": cold_passed / max(len(cached), 1),
            "memory_rate": memory_passed / max(len(cached), 1),
            "delta": (memory_passed - cold_passed) / max(len(cached), 1),
            "helped": helped,
            "hurt": hurt,
            "abstained": abstained,
        })

    results.sort(key=lambda row: (row["delta"], row["memory_rate"], -row["hurt"]), reverse=True)
    return results


def _run_eval_split(
    tasks: list[dict],
    label: str,
    *,
    base_model,
    tokenizer,
    memory_bank: "ClusterMemoryBank",
    embedder,
    config: PatchExperimentConfig,
    logger: logging.Logger,
    verbose: bool = False,
) -> dict[str, Any]:
    cold_passed = 0
    memory_passed = 0
    abstained = 0
    used_memory = 0

    if verbose:
        print()
        print(f"--- {label} COLD + MEMORY EVAL ---")

    for i, task in enumerate(tasks):
        task_id = task.get("task_id", task.get("id", f"{label}_{i}"))
        prompt = _task_prompt(task)
        task_embedding = _to_embedding_list(embedder.encode(prompt))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        cold_ok = False
        mem_ok = False
        active_memories = []
        active_patches = []

        try:
            cold_response = _generate_one(base_model, tokenizer, messages, temperature=0)
            cold_code = extract_code(cold_response)
            cold_result = evaluate_solution(task, cold_code, timeout=config.eval_timeout, mode="subprocess")
            cold_ok = bool(cold_result["passed"])

            active_memories, active_patches = memory_bank.get_active_patches(
                task_embedding,
                prompt,
                top_k=config.eval_top_k,
                return_memories=True,
            )
            if not active_patches:
                abstained += 1
                mem_ok = cold_ok
            else:
                used_memory += 1
                with _patched_model(base_model, active_patches, scaling_factor=config.eval_scale):
                    mem_response = _generate_one(base_model, tokenizer, messages, temperature=0)
                mem_code = extract_code(mem_response)
                mem_result = evaluate_solution(task, mem_code, timeout=config.eval_timeout, mode="subprocess")
                mem_ok = bool(mem_result["passed"])
        except Exception as exc:
            logger.warning(
                "Task %s failed during generate_with_model/evaluate_solution/PatchedModel: %s: %s\n%s",
                task_id,
                type(exc).__name__,
                exc,
                "".join(traceback.format_exc(limit=3)),
            )
            mem_ok = False
        finally:
            for patch in active_patches:
                patch.unload_weights()

        if cold_ok:
            cold_passed += 1
        if mem_ok:
            memory_passed += 1

        if len(active_memories) == 1:
            memory_bank.update_memory_utility(
                active_memories[0].memory_id,
                task_succeeded=mem_ok,
                cold_succeeded=cold_ok,
                eval_split=label.lower(),
                persist=False,
            )

        if verbose and ((i + 1) % 50 == 0 or i + 1 == len(tasks)):
            print(
                "  [{}/{}] cold: {}/{} ({:.1%}) | memory: {}/{} ({:.1%}) | abstain={}".format(
                    i + 1,
                    len(tasks),
                    cold_passed,
                    i + 1,
                    cold_passed / max(i + 1, 1),
                    memory_passed,
                    i + 1,
                    memory_passed / max(i + 1, 1),
                    abstained,
                )
            )

    cold_rate = cold_passed / max(len(tasks), 1)
    memory_rate = memory_passed / max(len(tasks), 1)
    if verbose:
        print(f"{label} cold result: {cold_passed} / {len(tasks)} ({cold_rate:.1%})")
        print(f"{label} memory result: {memory_passed} / {len(tasks)} ({memory_rate:.1%})")
        print(
            f"{label} memory usage: used={used_memory} abstained={abstained} "
            f"({abstained / max(len(tasks), 1):.1%} abstain)"
        )
    return {
        "label": label,
        "total": len(tasks),
        "cold_passed": cold_passed,
        "memory_passed": memory_passed,
        "cold_rate": cold_rate,
        "memory_rate": memory_rate,
        "delta": memory_rate - cold_rate,
        "used_memory": used_memory,
        "abstained": abstained,
    }


def evaluate_patch_memory_bank(
    train_tasks: list[dict],
    eval_tasks: list[dict],
    base_model,
    tokenizer,
    memory_bank: "ClusterMemoryBank",
    embedder,
    *,
    config: PatchExperimentConfig | None = None,
    force_rerun: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    config = config or PatchExperimentConfig()
    seen_tasks = list(train_tasks)
    unseen_tasks = list(eval_tasks[: config.unseen_eval_size])
    cache_path = build_eval_cache_path(
        config.memory_dir,
        version=config.eval_cache_version,
        seen_tasks=len(seen_tasks),
        unseen_tasks=len(unseen_tasks),
        top_k=config.eval_top_k,
        scale=config.eval_scale,
    )

    logger = logging.getLogger("cogmem.patches.experiment.eval")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    cached = None if force_rerun else load_eval_cache(cache_path)
    if cached:
        seen_eval = cached["seen_eval"]
        unseen_eval = cached["unseen_eval"]
        if verbose:
            print("Loaded cached eval results from", cache_path)
    else:
        seen_eval = _run_eval_split(
            seen_tasks,
            "SEEN",
            base_model=base_model,
            tokenizer=tokenizer,
            memory_bank=memory_bank,
            embedder=embedder,
            config=config,
            logger=logger,
            verbose=verbose,
        )
        unseen_eval = _run_eval_split(
            unseen_tasks,
            "UNSEEN",
            base_model=base_model,
            tokenizer=tokenizer,
            memory_bank=memory_bank,
            embedder=embedder,
            config=config,
            logger=logger,
            verbose=verbose,
        )
        save_eval_cache(cache_path, {"seen_eval": seen_eval, "unseen_eval": unseen_eval})
        if verbose:
            print("Saved eval cache to", cache_path)

    memory_bank.run_sleep_cycle(prune=True)
    memory_bank.save()
    return {
        "seen_eval": seen_eval,
        "unseen_eval": unseen_eval,
        "cache_path": cache_path,
        "used_cache": cached is not None,
    }


def summarize_patch_results(
    memory_bank: "ClusterMemoryBank",
    seen_eval: dict[str, Any],
    unseen_eval: dict[str, Any],
    *,
    config: PatchExperimentConfig | None = None,
) -> dict[str, Any]:
    config = config or PatchExperimentConfig()
    return {
        "episodes": len(memory_bank.episodes),
        "memories": len(memory_bank.memories),
        "artifact_patches": len(memory_bank.artifact_bank.patches),
        "eval_top_k": config.eval_top_k,
        "eval_scale": config.eval_scale,
        "seen_eval": seen_eval,
        "unseen_eval": unseen_eval,
        "memory_stats": memory_bank.stats(),
        "current_formulas": {
            "applicability": "clip(0.60 * pos_sim - 0.25 * neg_sim + 0.15 * structural_match - 0.10 * hard_negative_margin_penalty, 0, 1)",
            "q_use": "applicability * clip(0.45 * transfer_gain + 0.20 * recent_success_rate + 0.15 * log_reuse - 0.20 * online_hurt_rate, 0, 1)",
            "final_use": "clip(Q_use + 0.15 * Q_promote, 0, 1)",
            "q_promote": "0.28 * heldout_gain + 0.16 * transfer_gain + 0.06 * transfer_online_gain + 0.12 * local_support_gain + 0.10 * distillation_success + 0.08 * log_support + 0.10 * recent_success_rate - 0.10 * online_hurt_rate - 0.15 * utility_regression - 0.15 * unseen_hurt_rate - 0.10 * redundancy_penalty",
        },
    }


def run_patch_experiment(
    train_tasks: list[dict],
    eval_tasks: list[dict],
    base_model,
    tokenizer,
    embedder,
    *,
    config: PatchExperimentConfig | None = None,
    reset_progress: bool = False,
    force_rerun_eval: bool = False,
) -> dict[str, Any]:
    config = config or PatchExperimentConfig()
    memory_bank = _new_memory_bank(config.memory_dir)
    memory_bank.load()

    collection = run_patch_episode_recording(
        train_tasks,
        base_model,
        tokenizer,
        embedder,
        config=config,
        memory_bank=memory_bank,
        reset_progress=reset_progress,
    )
    build = build_patch_memories(
        base_model,
        tokenizer,
        memory_bank,
        config=config,
        eval_tasks=eval_tasks,
        embedder=embedder,
    )
    unseen = inspect_unseen_retrieval(
        eval_tasks,
        memory_bank,
        embedder,
        config=config,
    )
    sweep = sweep_patch_retrieval_width(
        train_tasks,
        base_model,
        tokenizer,
        memory_bank,
        embedder,
        config=config,
    )
    evaluation = evaluate_patch_memory_bank(
        train_tasks,
        eval_tasks,
        base_model,
        tokenizer,
        memory_bank,
        embedder,
        config=config,
        force_rerun=force_rerun_eval,
    )
    summary = summarize_patch_results(
        memory_bank,
        evaluation["seen_eval"],
        evaluation["unseen_eval"],
        config=config,
    )
    return {
        "config": asdict(config),
        "collection": collection,
        "build": build,
        "unseen_inspection": unseen,
        "retrieval_sweep": sweep,
        "evaluation": evaluation,
        "summary": summary,
    }


__all__ = [
    "PatchExperimentConfig",
    "prepare_patch_task_split",
    "load_patch_tasks",
    "load_patch_runtime",
    "run_patch_episode_recording",
    "build_patch_memories",
    "inspect_unseen_retrieval",
    "sweep_patch_retrieval_width",
    "evaluate_patch_memory_bank",
    "summarize_patch_results",
    "run_patch_experiment",
]

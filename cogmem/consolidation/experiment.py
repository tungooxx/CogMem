"""Notebook-friendly runners for the new CogMem architecture.

This module keeps the "new architecture" flow thin and package-driven:

    BigCodeBench tasks -> typed episodes -> skill cards -> optional Q-STaR cycle

The notebook should remain a UI wrapper around these helpers rather than
embedding collection and consolidation logic inline.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cogmem.benchmarks.bigcodebench.dataset import (
    load_bigcodebench,
    load_bigcodebench_from_jsonl,
)
from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution
from cogmem.benchmarks.bigcodebench.experiment import materialize_split_views
from cogmem.benchmarks.bigcodebench.prompts import extract_code, format_messages
from cogmem.benchmarks.bigcodebench.runner import run_single_task
from cogmem.benchmarks.bigcodebench.splits import (
    annotate_tasks_with_manifest,
    build_split_manifest,
    load_split_manifest,
    save_split_manifest,
)
from cogmem.config import CogMemConfig
from cogmem.consolidation.abstract import (
    prepare_preference_dataset,
    prepare_skill_training_dataset,
)
from cogmem.consolidation.proceduralize import (
    _episode_domain,
    _episode_hint_tokens,
    _feature_bucket,
    _group_key,
    build_skill_cards,
    rank_skill_cards_for_record,
    rank_skill_cards_for_task,
    score_skill_cards_for_record,
    task_to_skill_record,
)
from cogmem.consolidation.select import filter_manifest_eligible
from cogmem.memory.episodic_store import EpisodicStore, infer_error_family, render_episode_summary_context
from cogmem.memory.memory_bank import MemoryBank
from cogmem.memory.schema import get_episode_helpfulness
from cogmem.memory.skill_store import SkillStore, render_skill_card_context


@dataclass
class NewArchitectureExperimentConfig:
    experiment_dir: str = "results/new_architecture"
    tasks_jsonl_path: str | None = None
    manifest_path: str = "results/new_architecture/bigcodebench_manifest.json"
    memory_bank_path: str = "results/new_architecture/memory_bank.json"
    skills_path: str = "results/new_architecture/skill_cards.json"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    dataset_version: str = "v0.1.4"
    task_limit: int = 300
    train_fraction: float = 0.6
    dev_fraction: float = 0.2
    split_seed: int = 42
    max_tokens: int = 2048
    temperature: float = 0.0
    max_attempts: int = 3
    eval_timeout: int = 30
    episode_progress_filename: str = "episode_collection_progress.json"
    save_every_tasks: int = 1


class TransformersChatClient:
    """Small chat client adapter backed by a local transformers model."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _tokenize_messages(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        import torch

        inputs = self._tokenize_messages(messages)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = inputs.input_ids.shape[1]
        gen_ids = outputs[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)


def _ensure_torch_pytree_compat() -> None:
    """Backfill older torch pytree API expected by newer transformers."""
    import inspect
    import torch.utils._pytree as pytree

    if not hasattr(pytree, "register_pytree_node") and hasattr(pytree, "_register_pytree_node"):
        legacy_register = pytree._register_pytree_node
        supported_kwargs = set(inspect.signature(legacy_register).parameters)

        def register_pytree_node(*args, **kwargs):
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}
            return legacy_register(*args, **filtered_kwargs)

        pytree.register_pytree_node = register_pytree_node


def load_new_arch_tasks(
    *,
    task_jsonl_path: str | None = None,
    version: str = "v0.1.4",
    hard_only: bool = False,
) -> list[dict]:
    if task_jsonl_path:
        return load_bigcodebench_from_jsonl(task_jsonl_path)
    return load_bigcodebench(version=version, hard_only=hard_only)


def prepare_new_arch_task_split(
    tasks: list[dict],
    *,
    config: NewArchitectureExperimentConfig | None = None,
) -> dict[str, Any]:
    config = config or NewArchitectureExperimentConfig()
    limited_tasks = list(tasks[: config.task_limit]) if config.task_limit else list(tasks)
    manifest_path = Path(config.manifest_path)
    if manifest_path.exists():
        manifest = load_split_manifest(str(manifest_path))
        manifest_task_ids = set(manifest.get("task_splits", {}).keys())
        current_task_ids = {task["task_id"] for task in limited_tasks}
        if manifest_task_ids != current_task_ids:
            manifest = build_split_manifest(
                limited_tasks,
                train_fraction=config.train_fraction,
                dev_fraction=config.dev_fraction,
                seed=config.split_seed,
                label="bigcodebench_cl",
                source_benchmark="bigcodebench",
                dataset_version=config.dataset_version,
            )
            save_split_manifest(manifest, str(manifest_path))
    else:
        manifest = build_split_manifest(
            limited_tasks,
            train_fraction=config.train_fraction,
            dev_fraction=config.dev_fraction,
            seed=config.split_seed,
            label="bigcodebench_cl",
            source_benchmark="bigcodebench",
            dataset_version=config.dataset_version,
        )
        save_split_manifest(manifest, str(manifest_path))

    annotated_tasks = annotate_tasks_with_manifest(limited_tasks, manifest)
    split_views = materialize_split_views(annotated_tasks)
    return {
        "manifest": manifest,
        "tasks": annotated_tasks,
        "train_tasks": split_views["train"],
        "dev_tasks": split_views["dev"],
        "test_tasks": split_views["test"],
    }


def load_new_arch_runtime(
    *,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: str | None = None,
):
    import torch

    _ensure_torch_pytree_compat()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    except Exception as exc:
        print(f"4-bit runtime load failed ({type(exc).__name__}: {exc}); retrying without bitsandbytes quantization.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, TransformersChatClient(model, tokenizer)


def _release_new_arch_runtime(model=None, tokenizer=None, llm_client=None) -> None:
    del llm_client
    del tokenizer
    del model

    try:
        import gc

        gc.collect()
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _augment_messages_with_skill_cards(messages: list[dict], skill_cards: list[dict]) -> list[dict]:
    if not skill_cards:
        return messages
    augmented = [dict(message) for message in messages]
    augmented[0]["content"] = (
        augmented[0]["content"]
        + "\n\nWhen activation conditions match, use the retrieved validated procedural memory below."
        + " Apply it as guidance, not as code to copy blindly."
    )
    skill_blocks = "\n\n".join(render_skill_card_context(card) for card in skill_cards)
    augmented[1]["content"] = f"Retrieved procedural memory:\n{skill_blocks}\n\nTask:\n{augmented[1]['content']}"
    return augmented


def _augment_messages_with_episode_summary(messages: list[dict], episode_summary: str) -> list[dict]:
    if not episode_summary:
        return messages
    augmented = [dict(message) for message in messages]
    augmented[0]["content"] = (
        augmented[0]["content"]
        + "\n\nWhen the retrieved episodic memory is relevant, use it as a compact debugging hint."
        + " Reuse the pattern, but still solve the current task directly."
    )
    augmented[1]["content"] = f"{episode_summary}\n\nTask:\n{augmented[1]['content']}"
    return augmented


def _augment_messages_with_memory_route(
    messages: list[dict],
    *,
    skill_cards: list[dict] | None = None,
    episode_summary: str | None = None,
) -> list[dict]:
    augmented = [dict(message) for message in messages]
    if skill_cards:
        augmented = _augment_messages_with_skill_cards(augmented, skill_cards)
    if episode_summary:
        augmented = _augment_messages_with_episode_summary(augmented, episode_summary)
    return augmented


def _task_prompt_hash(task_description: str) -> str | None:
    text = str(task_description or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _episode_route_disabled(episode: dict, config: CogMemConfig | None = None) -> bool:
    runtime_stats = dict(episode.get("runtime_stats", {}) or {})
    retrieved = int(runtime_stats.get("retrieved", 0) or 0)
    hurt = int(runtime_stats.get("hurt", 0) or 0)
    helped = int(runtime_stats.get("helped", 0) or 0)
    min_retrieved = int(getattr(config, "episode_runtime_disable_min_retrieved", 8))
    min_hurt = int(getattr(config, "episode_runtime_disable_min_hurt", 3))
    hurt_rate_threshold = float(getattr(config, "episode_runtime_disable_hurt_rate", 0.10))
    if retrieved < min_retrieved:
        return False
    hurt_rate = hurt / max(retrieved, 1)
    return helped == 0 and hurt >= min_hurt and hurt_rate >= hurt_rate_threshold


def _score_episode_summaries_for_record(
    store: EpisodicStore | list[dict],
    record: dict,
    *,
    limit: int = 1,
    config: CogMemConfig | None = None,
    retry: bool = False,
    exclude_episode_ids: set[str] | None = None,
) -> list[tuple[float, dict]]:
    episodes = list(store) if isinstance(store, EpisodicStore) else list(store)
    record_group = _group_key(record)
    record_domain = _episode_domain(record)
    record_feature = _feature_bucket(record, record_domain)
    record_tokens = _episode_hint_tokens(record)
    error_family = str(record.get("error_family") or infer_error_family(record.get("error")) or "")
    record_prompt_hash = _task_prompt_hash(record.get("task_description", ""))
    required_score = float(
        getattr(
            config,
            "episode_retrieval_retry_min_score" if retry else "episode_retrieval_min_score",
            5.5 if retry else 7.0,
        )
    )
    excluded = set(exclude_episode_ids or set())
    scored: list[tuple[float, dict]] = []
    for episode in episodes:
        if not episode.get("success"):
            continue
        if episode.get("episode_id") in excluded:
            continue
        if _episode_route_disabled(episode, config):
            continue
        episode_group = _group_key(episode)
        episode_domain = _episode_domain(episode)
        episode_feature = _feature_bucket(episode, episode_domain)
        episode_tokens = _episode_hint_tokens(episode)
        trigger_overlap = record_tokens & episode_tokens
        episode_error_family = str(episode.get("error_family") or infer_error_family(episode.get("error")) or "")
        score = 0.0
        if episode_group == record_group:
            score += 3.5
        if episode_domain == record_domain and record_domain != "general":
            score += 2.0
        if episode_feature and episode_feature == record_feature:
            score += 1.5
        if record_prompt_hash and episode.get("prompt_hash") and episode.get("prompt_hash") == record_prompt_hash:
            score += 3.0
        if error_family and episode_error_family and error_family == episode_error_family:
            score += 1.5
        score += 0.5 * len(trigger_overlap)
        score += 2.0 * max(get_episode_helpfulness(episode, 0.0), 0.0)
        runtime_stats = dict(episode.get("runtime_stats", {}) or {})
        retrieved = int(runtime_stats.get("retrieved", 0) or 0)
        if retrieved > 0:
            helped = int(runtime_stats.get("helped", 0) or 0)
            hurt = int(runtime_stats.get("hurt", 0) or 0)
            score += 3.0 * (helped / max(retrieved, 1))
            score -= 4.5 * (hurt / max(retrieved, 1))
        if score >= required_score:
            scored.append((score, episode))
    scored.sort(
        key=lambda item: (
            -item[0],
            -float(get_episode_helpfulness(item[1], 0.0)),
            item[1].get("episode_id", ""),
        )
    )
    return scored[:limit]


def _route_record_for_task(task: dict, *, error_text: str | None = None) -> dict:
    record = task_to_skill_record(task)
    record["task_id"] = task.get("task_id")
    record["prompt_hash"] = _task_prompt_hash(record.get("task_description", ""))
    if error_text:
        record["error"] = error_text
        record["error_family"] = infer_error_family(error_text)
        record["task_description"] = (
            f"{record.get('task_description', '')}\n\nObserved failure:\n{str(error_text)[:500]}"
        ).strip()
    return record


def select_runtime_route(
    task: dict,
    *,
    route_mode: str = "router",
    skill_store: SkillStore | None = None,
    episode_store: EpisodicStore | None = None,
    skill_top_k: int = 1,
    config: CogMemConfig | None = None,
    error_text: str | None = None,
    previous_route_history: list[dict] | None = None,
    initial_skill_cards: list[dict] | None = None,
    initial_episode: dict | None = None,
) -> dict[str, Any]:
    record = _route_record_for_task(task, error_text=error_text)
    seen_skill_ids = {
        skill_id
        for item in list(previous_route_history or [])
        for skill_id in list(item.get("skill_ids", []) or [])
    }
    seen_episode_ids = {
        str(item.get("episode_id"))
        for item in list(previous_route_history or [])
        if item.get("episode_id")
    }
    if error_text:
        initial_skill_cards = None
        initial_episode = None

    if initial_skill_cards is not None and route_mode in {"skill", "router"}:
        skill_scored = [(float("inf"), card) for card in initial_skill_cards[:skill_top_k]]
    elif skill_store is not None and route_mode in {"skill", "router"}:
        skill_scored = [
            (score, card)
            for score, card in score_skill_cards_for_record(
                skill_store,
                record,
                limit=max(skill_top_k + len(seen_skill_ids), skill_top_k),
                promoted_only=True,
                config=config,
            )
            if card.get("skill_id") not in seen_skill_ids
        ][:skill_top_k]
    else:
        skill_scored = []

    allow_episode_route = route_mode in {"episode", "router"} and (
        bool(error_text) or bool(getattr(config, "episode_retrieval_allow_first_attempt", False))
    )

    if initial_episode is not None and allow_episode_route:
        episode_scored = [(float("inf"), initial_episode)]
    elif episode_store is not None and allow_episode_route:
        episode_scored = _score_episode_summaries_for_record(
            episode_store,
            record,
            limit=1,
            config=config,
            retry=bool(error_text),
            exclude_episode_ids=seen_episode_ids if error_text else None,
        )
    else:
        episode_scored = []

    route_scores = {
        "skill": skill_scored[0][0] if skill_scored else None,
        "episode_summary": episode_scored[0][0] if episode_scored else None,
    }
    candidates: list[tuple[str, float]] = []
    if skill_scored:
        candidates.append(("skill", float(skill_scored[0][0])))
    if episode_scored:
        candidates.append(("episode_summary", float(episode_scored[0][0])))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    if not candidates:
        return {
            "selected_route": "none",
            "skill_cards": [],
            "episode": None,
            "route_scores": route_scores,
            "abstain_reason": "no_candidate_above_threshold",
        }
    selected_route = candidates[0][0]
    return {
        "selected_route": selected_route,
        "skill_cards": [card for _, card in skill_scored] if selected_route == "skill" else [],
        "episode": episode_scored[0][1] if selected_route == "episode_summary" else None,
        "route_scores": route_scores,
        "abstain_reason": None,
    }


def _run_task_with_memory_route(
    task: dict,
    llm_client,
    *,
    route_mode: str = "router",
    initial_skill_cards: list[dict] | None = None,
    initial_episode: dict | None = None,
    skill_store: SkillStore | None = None,
    episode_store: EpisodicStore | None = None,
    skill_top_k: int = 1,
    config: CogMemConfig | None = None,
    eval_mode: str = "subprocess",
    eval_timeout: int = 30,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_attempts: int = 1,
) -> dict:
    trajectory = []
    retrieved_route_history: list[dict[str, Any]] = []
    current_skill_cards = list(initial_skill_cards or [])
    current_episode = initial_episode

    for attempt in range(1, max_attempts + 1):
        prev_error = trajectory[-1].get("error", "Unknown error") if attempt > 1 else None
        route_selection = select_runtime_route(
            task,
            route_mode=route_mode,
            skill_store=skill_store,
            episode_store=episode_store,
            skill_top_k=skill_top_k,
            config=config,
            error_text=prev_error,
            previous_route_history=retrieved_route_history,
            initial_skill_cards=current_skill_cards if attempt == 1 else None,
            initial_episode=current_episode if attempt == 1 else None,
        )
        selected_route = route_selection["selected_route"]
        current_skill_cards = list(route_selection.get("skill_cards", []) or [])
        current_episode = route_selection.get("episode")
        route_entry = {
            "attempt": attempt,
            "selected_route": selected_route,
            "skill_ids": [card["skill_id"] for card in current_skill_cards],
            "episode_id": current_episode.get("episode_id") if current_episode else None,
            "route_scores": dict(route_selection.get("route_scores", {}) or {}),
            "abstain_reason": route_selection.get("abstain_reason"),
        }
        retrieved_route_history.append(route_entry)

        if attempt > 1:
            retry_prompt = (
                f"Your previous attempt failed with this error:\n"
                f"{prev_error[:500]}\n\n"
                f"Please fix the code and try again."
            )

        episode_summary = (
            render_episode_summary_context(
                current_episode,
                max_code_lines=int(getattr(config, "episode_summary_max_code_lines", 8)),
            )
            if current_episode is not None else None
        )
        messages = _augment_messages_with_memory_route(
            format_messages(task, use_instruct=True),
            skill_cards=current_skill_cards,
            episode_summary=episode_summary,
        )
        if attempt > 1:
            messages.append({"role": "user", "content": retry_prompt})

        try:
            response = llm_client.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            trajectory.append(
                {
                    "attempt": attempt,
                    "code": "",
                    "response": "",
                    "test_result": "ERROR",
                    "error": str(exc),
                }
            )
            continue

        code = extract_code(response, task)
        result = evaluate_solution(task, code, timeout=eval_timeout, mode=eval_mode)
        passed = result["passed"]
        trajectory.append(
            {
                "attempt": attempt,
                "code": code,
                "response": response,
                "test_result": "PASS" if passed else "FAIL",
                "error": result.get("error") if not passed else None,
            }
        )
        if passed:
            break

    success = any(step["test_result"] == "PASS" for step in trajectory)
    return {
        "task_id": task["task_id"],
        "success": success,
        "num_attempts": len(trajectory),
        "error": trajectory[-1].get("error") if trajectory else None,
        "selected_route": retrieved_route_history[-1]["selected_route"] if retrieved_route_history else "none",
        "abstained": route_mode != "none" and all(item.get("selected_route") == "none" for item in retrieved_route_history),
        "retrieved_skill_ids": list(
            dict.fromkeys(
                skill_id
                for item in retrieved_route_history
                for skill_id in list(item.get("skill_ids", []) or [])
            )
        ),
        "retrieved_skill_history": [list(item.get("skill_ids", []) or []) for item in retrieved_route_history],
        "retrieved_episode_id": next(
            (
                item.get("episode_id")
                for item in reversed(retrieved_route_history)
                if item.get("episode_id")
            ),
            None,
        ),
        "retrieved_route_history": retrieved_route_history,
    }


def _run_task_with_skill_cards(
    task: dict,
    llm_client,
    *,
    retrieved_skill_cards: list[dict],
    skill_store: SkillStore | None = None,
    skill_top_k: int = 1,
    config: CogMemConfig | None = None,
    eval_mode: str = "subprocess",
    eval_timeout: int = 30,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_attempts: int = 1,
) -> dict:
    return _run_task_with_memory_route(
        task,
        llm_client,
        route_mode="skill",
        initial_skill_cards=retrieved_skill_cards,
        skill_store=skill_store,
        skill_top_k=skill_top_k,
        config=config,
        eval_mode=eval_mode,
        eval_timeout=eval_timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        max_attempts=max_attempts,
    )


def evaluate_new_arch_model(
    eval_tasks: list[dict],
    *,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: str | None = None,
    skill_cards_path: str | None = None,
    skill_store: SkillStore | None = None,
    episode_memory_path: str | None = None,
    episode_store: EpisodicStore | None = None,
    route_mode: str = "none",
    skill_top_k: int = 1,
    config: CogMemConfig | None = None,
    task_limit: int | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_attempts: int = 1,
    eval_timeout: int = 30,
    verbose: bool = False,
) -> dict[str, Any]:
    tasks = list(eval_tasks[:task_limit]) if task_limit else list(eval_tasks)
    model = tokenizer = llm_client = None
    rows: list[dict[str, Any]] = []
    passed = 0
    start_time = time.time()
    if skill_store is None and skill_cards_path:
        skill_store = SkillStore.load(skill_cards_path)
    if episode_store is None and episode_memory_path:
        loaded_episode_store = EpisodicStore.load(episode_memory_path)
        episode_store = (
            EpisodicStore(filter_manifest_eligible(list(loaded_episode_store), config))
            if config is not None else loaded_episode_store
        )
    selected_skill_ids: Counter[str] = Counter()
    selected_episode_ids: Counter[str] = Counter()
    selected_route_counts: Counter[str] = Counter()
    abstain_count = 0

    try:
        model, tokenizer, llm_client = load_new_arch_runtime(
            model_name=model_name,
            adapter_path=adapter_path,
        )

        for idx, task in enumerate(tasks):
            episode = _run_task_with_memory_route(
                task,
                llm_client,
                route_mode=route_mode,
                skill_store=skill_store,
                episode_store=episode_store,
                skill_top_k=skill_top_k,
                config=config,
                eval_mode="subprocess",
                eval_timeout=eval_timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                max_attempts=max_attempts,
            )
            success = bool(episode.get("success"))
            if success:
                passed += 1
            selected_route = str(episode.get("selected_route", "none") or "none")
            selected_route_counts[selected_route] += 1
            if route_mode != "none" and bool(episode.get("abstained")):
                abstain_count += 1
            for skill_id in episode.get("retrieved_skill_ids", []) or []:
                selected_skill_ids[skill_id] += 1
            retrieved_episode_id = episode.get("retrieved_episode_id")
            if retrieved_episode_id:
                selected_episode_ids[str(retrieved_episode_id)] += 1

            rows.append(
                {
                    "task_id": task["task_id"],
                    "passed": success,
                    "attempts": int(episode.get("num_attempts", 0) or 0),
                    "error": episode.get("error"),
                    "selected_route": selected_route,
                    "abstained": bool(episode.get("abstained")),
                    "retrieved_skill_ids": list(episode.get("retrieved_skill_ids", []) or []),
                    "retrieved_episode_id": retrieved_episode_id,
                    "retrieved_skill_history": list(episode.get("retrieved_skill_history", []) or []),
                    "retrieved_route_history": list(episode.get("retrieved_route_history", []) or []),
                }
            )

            if verbose and ((idx + 1) % 10 == 0 or idx < 5):
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed * 3600 if elapsed > 0 else 0.0
                label_parts = ["adapter" if adapter_path else "base"]
                if route_mode == "skill" and skill_store is not None:
                    label_parts.append("skill")
                elif route_mode == "episode" and episode_store is not None:
                    label_parts.append("episode")
                elif route_mode == "router" and (skill_store is not None or episode_store is not None):
                    label_parts.append("router")
                label = "+".join(label_parts)
                print(
                    "[{}/{}] {} | {}={} | pass_rate={:.1%} | {:.0f}/hr".format(
                        idx + 1,
                        len(tasks),
                        task["task_id"],
                        label,
                        int(success),
                        passed / max(idx + 1, 1),
                        rate,
                    )
                )
    finally:
        _release_new_arch_runtime(model, tokenizer, llm_client)

    task_count = len(tasks)
    return {
        "adapter_path": adapter_path,
        "task_count": task_count,
        "passed": passed,
        "pass_rate": passed / task_count if task_count else 0.0,
        "elapsed_minutes": (time.time() - start_time) / 60.0,
        "route_mode": route_mode,
        "skill_cards_path": skill_cards_path,
        "episode_memory_path": episode_memory_path,
        "selected_skill_ids": dict(selected_skill_ids),
        "selected_episode_ids": dict(selected_episode_ids),
        "selected_route_counts": dict(selected_route_counts),
        "abstain_count": abstain_count,
        "rows": rows,
    }


def _task_domain_label(task: dict) -> str:
    libs = [str(lib).strip().lower() for lib in task.get("libs", []) or [] if str(lib).strip()]
    if libs:
        return libs[0]
    description = str(task.get("instruct_prompt") or task.get("complete_prompt") or "").lower()
    for domain in ("pandas", "numpy", "matplotlib", "glob", "pathlib", "json", "csv", "regex", "datetime"):
        if domain in description:
            return "file_io" if domain in {"glob", "pathlib", "json", "csv"} else domain
    return "general"


def _episode_to_eval_task(episode: dict) -> dict[str, Any]:
    return {
        "task_id": episode.get("task_id"),
        "instruct_prompt": episode.get("task_description", "") or "",
        "complete_prompt": episode.get("task_description", "") or "",
        "entry_point": episode.get("entry_point", "") or "",
        "libs": list(episode.get("libs", []) or []),
        "task_type": episode.get("task_type", "bigcodebench") or "bigcodebench",
        "error": episode.get("error"),
    }


def _build_skill_utility(
    tasks: list[dict],
    baseline_rows: dict[str, dict[str, Any]],
    routed_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    utility: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_id = task["task_id"]
        routed = routed_rows.get(task_id, {})
        skill_ids = list(routed.get("retrieved_skill_ids", []) or [])
        if not skill_ids:
            continue
        baseline = baseline_rows.get(task_id, {})
        routed_pass = bool(routed.get("passed"))
        baseline_pass = bool(baseline.get("passed"))
        error_family = infer_error_family(routed.get("error"))
        domain = _task_domain_label(task)
        for skill_id in skill_ids:
            stats = utility.setdefault(
                skill_id,
                {
                    "retrieved": 0,
                    "helped": 0,
                    "hurt": 0,
                    "preserved_success": 0,
                    "preserved_failure": 0,
                    "passed": 0,
                    "failed": 0,
                    "domains": Counter(),
                    "error_families": Counter(),
                    "task_ids": [],
                },
            )
            stats["retrieved"] += 1
            stats["passed"] += int(routed_pass)
            stats["failed"] += int(not routed_pass)
            if routed_pass and not baseline_pass:
                stats["helped"] += 1
            elif baseline_pass and not routed_pass:
                stats["hurt"] += 1
            elif routed_pass and baseline_pass:
                stats["preserved_success"] += 1
            else:
                stats["preserved_failure"] += 1
            stats["domains"][domain] += 1
            if error_family:
                stats["error_families"][error_family] += 1
            if task_id not in stats["task_ids"]:
                stats["task_ids"].append(task_id)

    summarized: dict[str, dict[str, Any]] = {}
    for skill_id, stats in utility.items():
        retrieved = max(int(stats["retrieved"]), 1)
        summarized[skill_id] = {
            "retrieved": int(stats["retrieved"]),
            "helped": int(stats["helped"]),
            "hurt": int(stats["hurt"]),
            "help_rate": int(stats["helped"]) / retrieved,
            "hurt_rate": int(stats["hurt"]) / retrieved,
            "passed": int(stats["passed"]),
            "failed": int(stats["failed"]),
            "domains": dict(stats["domains"]),
            "error_families": dict(stats["error_families"]),
            "task_ids": stats["task_ids"][:10],
        }
    return summarized


def _build_episode_utility(
    tasks: list[dict],
    baseline_rows: dict[str, dict[str, Any]],
    routed_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    utility: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_id = task["task_id"]
        routed = routed_rows.get(task_id, {})
        episode_id = routed.get("retrieved_episode_id")
        if not episode_id:
            continue
        baseline = baseline_rows.get(task_id, {})
        routed_pass = bool(routed.get("passed"))
        baseline_pass = bool(baseline.get("passed"))
        error_family = infer_error_family(routed.get("error"))
        domain = _task_domain_label(task)
        stats = utility.setdefault(
            str(episode_id),
            {
                "retrieved": 0,
                "helped": 0,
                "hurt": 0,
                "preserved_success": 0,
                "preserved_failure": 0,
                "passed": 0,
                "failed": 0,
                "domains": Counter(),
                "error_families": Counter(),
                "task_ids": [],
            },
        )
        stats["retrieved"] += 1
        stats["passed"] += int(routed_pass)
        stats["failed"] += int(not routed_pass)
        if routed_pass and not baseline_pass:
            stats["helped"] += 1
        elif baseline_pass and not routed_pass:
            stats["hurt"] += 1
        elif routed_pass and baseline_pass:
            stats["preserved_success"] += 1
        else:
            stats["preserved_failure"] += 1
        stats["domains"][domain] += 1
        if error_family:
            stats["error_families"][error_family] += 1
        if task_id not in stats["task_ids"]:
            stats["task_ids"].append(task_id)

    summarized: dict[str, dict[str, Any]] = {}
    for episode_id, stats in utility.items():
        retrieved = max(int(stats["retrieved"]), 1)
        summarized[episode_id] = {
            "retrieved": int(stats["retrieved"]),
            "helped": int(stats["helped"]),
            "hurt": int(stats["hurt"]),
            "help_rate": int(stats["helped"]) / retrieved,
            "hurt_rate": int(stats["hurt"]) / retrieved,
            "passed": int(stats["passed"]),
            "failed": int(stats["failed"]),
            "domains": dict(stats["domains"]),
            "error_families": dict(stats["error_families"]),
            "task_ids": stats["task_ids"][:10],
        }
    return summarized


def _compare_route_results(
    tasks: list[dict],
    baseline_eval: dict[str, Any],
    candidate_eval: dict[str, Any],
) -> dict[str, Any]:
    baseline_rows = {row["task_id"]: row for row in baseline_eval.get("rows", [])}
    candidate_rows = {row["task_id"]: row for row in candidate_eval.get("rows", [])}
    task_ids = [task["task_id"] for task in tasks]
    improved = [
        task_id
        for task_id in task_ids
        if candidate_rows.get(task_id, {}).get("passed") and not baseline_rows.get(task_id, {}).get("passed")
    ]
    regressed = [
        task_id
        for task_id in task_ids
        if baseline_rows.get(task_id, {}).get("passed") and not candidate_rows.get(task_id, {}).get("passed")
    ]
    task_count = max(len(task_ids), 1)
    return {
        "delta_passed": int(candidate_eval.get("passed", 0)) - int(baseline_eval.get("passed", 0)),
        "delta_pass_rate": float(candidate_eval.get("pass_rate", 0.0)) - float(baseline_eval.get("pass_rate", 0.0)),
        "improved_task_ids": improved,
        "regressed_task_ids": regressed,
        "regression_rate": len(regressed) / task_count,
        "selected_route_counts": dict(candidate_eval.get("selected_route_counts", {}) or {}),
        "abstain_count": int(candidate_eval.get("abstain_count", 0) or 0),
        "skill_utility": _build_skill_utility(tasks, baseline_rows, candidate_rows),
        "episode_utility": _build_episode_utility(tasks, baseline_rows, candidate_rows),
    }


def persist_route_memory_utility(
    skill_cards_path: str | None,
    memory_bank_path: str | None,
    comparisons: dict[str, Any],
) -> dict[str, Any]:
    updated_routes: list[str] = []
    updated_skills = 0
    updated_episodes = 0
    if skill_cards_path:
        store = SkillStore.load(skill_cards_path)
        for route_name, route_result in dict(comparisons or {}).items():
            skill_utility = dict(route_result.get("skill_utility", {}) or {})
            if not skill_utility:
                continue
            updated_skills += store.apply_runtime_utility(skill_utility, route_name=route_name)
            updated_routes.append(route_name)
        if updated_skills:
            store.save(skill_cards_path)
    if memory_bank_path:
        store = EpisodicStore.load(memory_bank_path)
        for route_name, route_result in dict(comparisons or {}).items():
            episode_utility = dict(route_result.get("episode_utility", {}) or {})
            if not episode_utility:
                continue
            updated_episodes += store.apply_runtime_utility(episode_utility, route_name=route_name)
            if route_name not in updated_routes:
                updated_routes.append(route_name)
        if updated_episodes:
            store.save(memory_bank_path)
    return {
        "skills_path": skill_cards_path,
        "memory_bank_path": memory_bank_path,
        "updated_skills": updated_skills,
        "updated_episodes": updated_episodes,
        "updated_routes": updated_routes,
    }


def persist_route_skill_utility(
    skill_cards_path: str | None,
    comparisons: dict[str, Any],
    memory_bank_path: str | None = None,
) -> dict[str, Any]:
    return persist_route_memory_utility(skill_cards_path, memory_bank_path, comparisons)


def reset_route_runtime_utility(
    skill_cards_path: str | None = None,
    memory_bank_path: str | None = None,
) -> dict[str, Any]:
    reset_skills = 0
    reset_episodes = 0
    if skill_cards_path:
        store = SkillStore.load(skill_cards_path)
        for card in list(store):
            if card.get("runtime_stats"):
                reset_skills += 1
            store.update(card["skill_id"], runtime_stats={})
        if len(store):
            store.save(skill_cards_path)
    if memory_bank_path:
        store = EpisodicStore.load(memory_bank_path)
        for episode in list(store):
            if episode.get("runtime_stats"):
                reset_episodes += 1
            store.update(episode["episode_id"], runtime_stats={})
        if len(store):
            store.save(memory_bank_path)
    return {
        "skills_path": skill_cards_path,
        "memory_bank_path": memory_bank_path,
        "reset_skills": reset_skills,
        "reset_episodes": reset_episodes,
    }


def compare_new_arch_base_vs_adapter(
    eval_tasks: list[dict],
    *,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: str,
    task_limit: int | None = None,
    config: CogMemConfig | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_attempts: int = 1,
    eval_timeout: int = 30,
    verbose: bool = False,
) -> dict[str, Any]:
    result = compare_new_arch_routes(
        eval_tasks,
        model_name=model_name,
        adapter_path=adapter_path,
        skill_cards_path=None,
        episode_memory_path=None,
        config=config,
        task_limit=task_limit,
        max_tokens=max_tokens,
        temperature=temperature,
        max_attempts=max_attempts,
        eval_timeout=eval_timeout,
        verbose=verbose,
    )
    base_eval = result["base"]
    adapter_eval = result["adapter"]
    return {
        "task_count": result["task_count"],
        "base": base_eval,
        "adapter": adapter_eval,
        "delta_passed": adapter_eval["passed"] - base_eval["passed"],
        "delta_pass_rate": adapter_eval["pass_rate"] - base_eval["pass_rate"],
        "improved_task_ids": result["improved_task_ids"],
        "regressed_task_ids": result["regressed_task_ids"],
    }


def compare_new_arch_routes(
    eval_tasks: list[dict],
    *,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    skill_cards_path: str | None = None,
    skill_store: SkillStore | None = None,
    episode_memory_path: str | None = None,
    episode_store: EpisodicStore | None = None,
    adapter_path: str | None = None,
    skill_top_k: int = 1,
    config: CogMemConfig | None = None,
    task_limit: int | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_attempts: int = 1,
    eval_timeout: int = 30,
    verbose: bool = False,
) -> dict[str, Any]:
    tasks = list(eval_tasks[:task_limit]) if task_limit else list(eval_tasks)
    results: dict[str, Any] = {}

    results["base"] = evaluate_new_arch_model(
        tasks,
        model_name=model_name,
        adapter_path=None,
        skill_cards_path=None,
        episode_memory_path=None,
        route_mode="none",
        config=config,
        max_tokens=max_tokens,
        temperature=temperature,
        max_attempts=max_attempts,
        eval_timeout=eval_timeout,
        verbose=verbose,
    )
    if skill_cards_path or skill_store is not None:
        results["base_plus_skill"] = evaluate_new_arch_model(
            tasks,
            model_name=model_name,
            adapter_path=None,
            skill_cards_path=skill_cards_path,
            skill_store=skill_store,
            episode_memory_path=None,
            route_mode="skill",
            skill_top_k=skill_top_k,
            config=config,
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
            eval_timeout=eval_timeout,
            verbose=verbose,
        )
    if episode_memory_path or episode_store is not None:
        results["base_plus_episode"] = evaluate_new_arch_model(
            tasks,
            model_name=model_name,
            adapter_path=None,
            skill_cards_path=None,
            episode_memory_path=episode_memory_path,
            episode_store=episode_store,
            route_mode="episode",
            config=config,
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
            eval_timeout=eval_timeout,
            verbose=verbose,
        )
    if skill_cards_path or skill_store is not None or episode_memory_path or episode_store is not None:
        results["base_plus_router"] = evaluate_new_arch_model(
            tasks,
            model_name=model_name,
            adapter_path=None,
            skill_cards_path=skill_cards_path,
            skill_store=skill_store,
            episode_memory_path=episode_memory_path,
            episode_store=episode_store,
            route_mode="router",
            skill_top_k=skill_top_k,
            config=config,
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
            eval_timeout=eval_timeout,
            verbose=verbose,
        )
    if adapter_path:
        results["adapter"] = evaluate_new_arch_model(
            tasks,
            model_name=model_name,
            adapter_path=adapter_path,
            skill_cards_path=None,
            episode_memory_path=None,
            route_mode="none",
            config=config,
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
            eval_timeout=eval_timeout,
            verbose=verbose,
        )
    if adapter_path and (skill_cards_path or skill_store is not None or episode_memory_path or episode_store is not None):
        results["adapter_plus_router"] = evaluate_new_arch_model(
            tasks,
            model_name=model_name,
            adapter_path=adapter_path,
            skill_cards_path=skill_cards_path,
            skill_store=skill_store,
            episode_memory_path=episode_memory_path,
            episode_store=episode_store,
            route_mode="router",
            skill_top_k=skill_top_k,
            config=config,
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
            eval_timeout=eval_timeout,
            verbose=verbose,
        )

    comparisons: dict[str, Any] = {}
    if "base_plus_skill" in results:
        comparisons["base_plus_skill_vs_base"] = _compare_route_results(
            tasks,
            results["base"],
            results["base_plus_skill"],
        )
    if "base_plus_episode" in results:
        comparisons["base_plus_episode_vs_base"] = _compare_route_results(
            tasks,
            results["base"],
            results["base_plus_episode"],
        )
    if "base_plus_router" in results:
        comparisons["base_plus_router_vs_base"] = _compare_route_results(
            tasks,
            results["base"],
            results["base_plus_router"],
        )
    if "adapter" in results:
        comparisons["adapter_vs_base"] = _compare_route_results(
            tasks,
            results["base"],
            results["adapter"],
        )
    if "adapter_plus_router" in results and "adapter" in results:
        comparisons["adapter_plus_router_vs_adapter"] = _compare_route_results(
            tasks,
            results["adapter"],
            results["adapter_plus_router"],
        )
    adapter_comparison = comparisons.get("adapter_vs_base", {})

    return {
        **results,
        "task_count": results["base"]["task_count"],
        "comparisons": comparisons,
        "delta_passed": adapter_comparison.get("delta_passed") if adapter_path else None,
        "delta_pass_rate": adapter_comparison.get("delta_pass_rate") if adapter_path else None,
        "improved_task_ids": adapter_comparison.get("improved_task_ids", []),
        "regressed_task_ids": adapter_comparison.get("regressed_task_ids", []),
    }


def _collection_signature(tasks: list[dict]) -> dict[str, Any]:
    return {
        "count": len(tasks),
        "first_task_id": tasks[0]["task_id"] if tasks else "",
        "last_task_id": tasks[-1]["task_id"] if tasks else "",
    }


def run_new_arch_episode_collection(
    train_tasks: list[dict],
    llm_client,
    *,
    config: NewArchitectureExperimentConfig | None = None,
    memory_bank_path: str | None = None,
    reset_progress: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    config = config or NewArchitectureExperimentConfig()
    bank_path = memory_bank_path or config.memory_bank_path
    bank = MemoryBank.load(bank_path)
    progress_path = Path(config.experiment_dir) / config.episode_progress_filename
    signature = _collection_signature(train_tasks)

    if reset_progress and progress_path.exists():
        progress_path.unlink()

    resume_state: dict[str, Any] = {}
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            resume_state = json.load(f)

    same_plan = resume_state.get("train_signature") == signature
    start_idx = int(resume_state.get("next_index", 0)) if same_plan else 0
    episodes_before = len(bank)
    success_before = sum(1 for ep in bank if ep.get("success"))
    start_time = time.time()
    completed_ids = bank.completed_task_ids()
    successes_this_run = 0

    if verbose:
        print("Resume progress file:", progress_path)
        if start_idx > 0:
            print(f"Resuming collection from task {start_idx + 1} of {len(train_tasks)}.")
        else:
            print(f"Starting collection from task 1 of {len(train_tasks)}.")
        print(f"Existing episodes in bank: {len(bank)}")

    for i in range(start_idx, len(train_tasks)):
        task = train_tasks[i]
        task_id = task["task_id"]

        if task_id in completed_ids:
            if verbose and i < 5:
                print(f"  skipping already collected task {task_id}")
            continue

        episode = run_single_task(
            task,
            llm_client,
            eval_mode="subprocess",
            eval_timeout=config.eval_timeout,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            max_attempts=config.max_attempts,
        )
        bank.append(episode)
        completed_ids.add(task_id)
        if episode.get("success"):
            successes_this_run += 1

        if config.save_every_tasks <= 1 or (i + 1) % config.save_every_tasks == 0:
            bank.save(bank_path)

        progress_state = {
            "status": "running",
            "train_signature": signature,
            "next_index": i + 1,
            "last_task_id": task_id,
            "episodes_now": len(bank),
            "successes_this_run": successes_this_run,
            "updated_at": time.time(),
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress_state, f, indent=2)

        if verbose and ((i + 1) % 10 == 0 or i < 5):
            elapsed = time.time() - start_time
            processed = max(i + 1 - start_idx, 1)
            rate = processed / elapsed * 3600 if elapsed > 0 else 0.0
            print(
                "[{}/{}] {} | success={} | episodes={} | {:.0f}/hr".format(
                    i + 1,
                    len(train_tasks),
                    task_id,
                    int(bool(episode.get("success"))),
                    len(bank),
                    rate,
                )
            )

    bank.save(bank_path)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "complete",
                "train_signature": signature,
                "next_index": len(train_tasks),
                "last_task_id": train_tasks[-1]["task_id"] if train_tasks else "",
                "episodes_now": len(bank),
                "successes_this_run": successes_this_run,
                "updated_at": time.time(),
            },
            f,
            indent=2,
        )

    elapsed_minutes = (time.time() - start_time) / 60.0
    metrics = bank.summary_metrics()
    return {
        "tasks_processed": len(train_tasks),
        "new_episodes": len(bank) - episodes_before,
        "episodes_total": len(bank),
        "successes_before": success_before,
        "successes_this_run": successes_this_run,
        "elapsed_minutes": elapsed_minutes,
        "progress_path": str(progress_path),
        "memory_bank_path": bank_path,
        "summary_metrics": metrics,
    }


def _skill_rows(store: SkillStore, *, limit: int = 10) -> list[dict[str, Any]]:
    cards = sorted(
        list(store),
        key=lambda card: (
            {"promoted": 0, "validated": 1, "candidate": 2}.get(card.get("status"), 3),
            -float(card.get("confidence", 0.0) or 0.0),
            -float(card.get("transfer_gain", 0.0) or 0.0),
            -int(card.get("source_episode_count", 0) or 0),
        ),
    )
    rows = []
    for card in cards[:limit]:
        validation = dict(card.get("validation", {}) or {})
        route_test = dict(validation.get("route_test", {}) or {})
        rows.append({
            "skill_id": card["skill_id"],
            "status": card.get("status", "candidate"),
            "task_type": card.get("task_type", "general"),
            "domain": card.get("domain", "general"),
            "error_family": card.get("error_family"),
            "source_episode_count": int(card.get("source_episode_count", 0) or 0),
            "distinct_task_count": int(card.get("distinct_task_count", 0) or 0),
            "confidence": float(card.get("confidence", 0.0) or 0.0),
            "transfer_gain": float(card.get("transfer_gain", 0.0) or 0.0),
            "negative_transfer_rate": float(card.get("negative_transfer_rate", 0.0) or 0.0),
            "matched_episodes": int(validation.get("matched_episodes", 0) or 0),
            "matched_tasks": int(validation.get("matched_tasks", 0) or 0),
            "success_rate": float(validation.get("success_rate", 0.0) or 0.0),
            "route_test_status": route_test.get("status"),
            "route_delta_passed": route_test.get("delta_passed"),
            "route_regression_rate": route_test.get("regression_rate"),
            "triggers": list(card.get("triggers", [])[:5]),
            "plan_steps": list(card.get("plan_steps", [])[:3]),
            "activation_conditions": list(card.get("activation_conditions", [])[:3]),
            "stop_conditions": list(card.get("stop_conditions", [])[:3]),
            "anti_patterns": list(card.get("anti_patterns", [])[:3]),
            "manifest_ids": list(card.get("manifest_ids", [])),
        })
    return rows


def _promote_validated_skill_cards(
    skill_store: SkillStore,
    holdout_episodes: list[dict],
    cogmem_config: CogMemConfig,
) -> SkillStore:
    holdout_tasks_by_id: dict[str, dict[str, Any]] = {}
    for episode in holdout_episodes:
        task_id = episode.get("task_id")
        if task_id and task_id not in holdout_tasks_by_id:
            holdout_tasks_by_id[str(task_id)] = _episode_to_eval_task(episode)
    min_tasks = int(getattr(cogmem_config, "skill_route_promotion_min_tasks", 2))
    min_delta_passed = int(getattr(cogmem_config, "skill_route_promotion_min_delta_passed", 1))
    max_regression_rate = float(getattr(cogmem_config, "skill_route_promotion_max_regression_rate", 0.10))

    for card in list(skill_store):
        if card.get("status") != "validated":
            continue
        validation = dict(card.get("validation", {}) or {})
        matched_task_ids = [
            task_id
            for task_id in list(validation.get("validated_task_ids", []) or [])
            if task_id in holdout_tasks_by_id
        ]
        route_test = {
            "status": "skipped_insufficient_tasks",
            "task_ids": matched_task_ids[:10],
            "delta_passed": None,
            "delta_pass_rate": None,
            "regression_rate": None,
            "improved_task_ids": [],
            "regressed_task_ids": [],
        }
        if len(matched_task_ids) >= min_tasks:
            route_eval = compare_new_arch_routes(
                [holdout_tasks_by_id[task_id] for task_id in matched_task_ids],
                model_name=cogmem_config.active_model_hf,
                skill_store=SkillStore([{**card, "status": "promoted"}]),
                adapter_path=None,
                skill_top_k=cogmem_config.skill_retrieval_top_k,
                config=cogmem_config,
                max_tokens=2048,
                temperature=0.0,
                max_attempts=1,
                eval_timeout=30,
                verbose=False,
            )
            comparison = dict(route_eval.get("comparisons", {}).get("base_plus_skill_vs_base", {}) or {})
            route_test = {
                "status": "promoted" if (
                    comparison.get("delta_passed", 0) >= min_delta_passed
                    and comparison.get("regression_rate", 0.0) <= max_regression_rate
                ) else "validated",
                "task_ids": matched_task_ids[:10],
                "delta_passed": comparison.get("delta_passed"),
                "delta_pass_rate": comparison.get("delta_pass_rate"),
                "regression_rate": comparison.get("regression_rate"),
                "improved_task_ids": list(comparison.get("improved_task_ids", []) or [])[:10],
                "regressed_task_ids": list(comparison.get("regressed_task_ids", []) or [])[:10],
            }
        validation["route_test"] = route_test
        skill_store.update(
            card["skill_id"],
            status="promoted" if route_test.get("status") == "promoted" else "validated",
            validation=validation,
        )
    return skill_store


def build_new_arch_skill_cards(
    memory_bank_path: str,
    cogmem_config: CogMemConfig,
    *,
    skills_path: str,
    summary_limit: int = 10,
) -> dict[str, Any]:
    bank = MemoryBank.load(memory_bank_path)
    eligible_episodes = filter_manifest_eligible(list(bank), cogmem_config)
    eligible_bank = MemoryBank(eligible_episodes)
    holdout_n = min(cogmem_config.min_holdout, len(eligible_episodes))
    holdout_episodes, available_episodes = eligible_bank.stratified_holdout(holdout_n, seed=cogmem_config.seed)

    skill_store = build_skill_cards(
        available_episodes,
        holdout_episodes,
        config=cogmem_config,
        output_path=skills_path,
    )
    skill_store = _promote_validated_skill_cards(skill_store, holdout_episodes, cogmem_config)
    skill_store.save(skills_path)
    promoted_cards = list(skill_store.filter(promoted=True))
    training_pairs = prepare_skill_training_dataset(
        promoted_cards,
        available_episodes,
        config=cogmem_config,
    )
    preference_pairs = prepare_preference_dataset(available_episodes, cogmem_config)
    task_type_counts = Counter(ep.get("task_type", "general") for ep in eligible_episodes)

    return {
        "skills_path": skills_path,
        "episodes_total": len(bank),
        "eligible_episodes": len(eligible_episodes),
        "available_episodes": len(available_episodes),
        "holdout_episodes": len(holdout_episodes),
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "skill_summary": skill_store.summary(),
        "promoted_skill_ids": [card["skill_id"] for card in promoted_cards],
        "training_pairs": len(training_pairs),
        "preference_pairs": len(preference_pairs),
        "skill_rows": _skill_rows(skill_store, limit=summary_limit),
    }


def run_new_arch_qstar_cycle(
    memory_bank_path: str,
    cogmem_config: CogMemConfig,
    *,
    cycle: int = 0,
    skill_cards_path: str | None = None,
    eval_tasks: list[dict] | None = None,
    eval_task_limit: int | None = None,
) -> dict[str, Any]:
    from cogmem.consolidation.pipeline import run_qstar_cycle

    bank = MemoryBank.load(memory_bank_path)
    eligible_episodes = filter_manifest_eligible(list(bank), cogmem_config)
    eligible_bank = MemoryBank(eligible_episodes)
    holdout_n = min(cogmem_config.min_holdout, len(eligible_episodes))
    holdout_episodes, available_episodes = eligible_bank.stratified_holdout(holdout_n, seed=cogmem_config.seed)
    skill_store = SkillStore.load(skill_cards_path) if skill_cards_path else SkillStore([])
    promoted_cards = list(skill_store.filter(promoted=True))
    skill_training_pairs = prepare_skill_training_dataset(
        promoted_cards,
        available_episodes,
        config=cogmem_config,
    )
    route_gate = None
    if skill_cards_path and eval_tasks:
        route_eval = compare_new_arch_routes(
            eval_tasks,
            model_name=cogmem_config.active_model_hf,
            skill_cards_path=skill_cards_path,
            episode_memory_path=memory_bank_path,
            adapter_path=None,
            skill_top_k=cogmem_config.skill_retrieval_top_k,
            config=cogmem_config,
            task_limit=eval_task_limit or cogmem_config.skill_route_gate_task_limit,
            max_tokens=2048,
            temperature=0.0,
            max_attempts=1,
            eval_timeout=30,
            verbose=False,
        )
        route_gate = route_eval.get("comparisons", {}).get("base_plus_router_vs_base", {})
        persist_route_memory_utility(skill_cards_path, memory_bank_path, route_eval.get("comparisons", {}))
        min_delta_passed = int(getattr(cogmem_config, "skill_route_gate_min_delta_passed", 1))
        max_regression_rate = float(getattr(cogmem_config, "skill_route_gate_max_regression_rate", 0.10))
        if (
            route_gate.get("delta_passed", 0) < min_delta_passed
            or route_gate.get("regression_rate", 0.0) > max_regression_rate
        ):
            return {
                "cycle": cycle,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model": cogmem_config.active_model_hf,
                "total_episodes": len(bank),
                "high_q_episodes": sum(
                    1
                    for ep in available_episodes
                    if float(ep.get("episode_helpfulness", ep.get("q_value", 0.0)) or 0.0) >= cogmem_config.q_threshold
                ),
                "preference_pairs": len(prepare_preference_dataset(available_episodes, cogmem_config)),
                "generator_path": None,
                "verifier_path": None,
                "skill_cards_path": skill_cards_path,
                "skill_cards_total": skill_store.summary().get("total", 0),
                "skill_cards_promoted": skill_store.summary().get("promoted", 0),
                "training_source": "skill_cards",
                "training_examples": len(skill_training_pairs),
                "adapter_registry_path": cogmem_config.adapter_registry_path,
                "verification": {},
                "status": "skipped_route_gate",
                "route_gate": route_gate,
            }

    result = run_qstar_cycle(
        memory_bank_path,
        cogmem_config,
        cycle=cycle,
        run_task_fn=None,
        existing_skill_cards_path=skill_cards_path,
    )
    if route_gate is not None:
        result["route_gate"] = route_gate
    return result


__all__ = [
    "NewArchitectureExperimentConfig",
    "TransformersChatClient",
    "load_new_arch_tasks",
    "prepare_new_arch_task_split",
    "load_new_arch_runtime",
    "evaluate_new_arch_model",
    "compare_new_arch_base_vs_adapter",
    "compare_new_arch_routes",
    "persist_route_memory_utility",
    "persist_route_skill_utility",
    "reset_route_runtime_utility",
    "run_new_arch_episode_collection",
    "build_new_arch_skill_cards",
    "run_new_arch_qstar_cycle",
]

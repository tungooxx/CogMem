"""Notebook-friendly runners for the new CogMem architecture.

This module keeps the "new architecture" flow thin and package-driven:

    BigCodeBench tasks -> typed episodes -> skill cards -> optional Q-STaR cycle

The notebook should remain a UI wrapper around these helpers rather than
embedding collection and consolidation logic inline.
"""

from __future__ import annotations

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
    build_skill_cards,
    rank_skill_cards_for_record,
    rank_skill_cards_for_task,
    task_to_skill_record,
)
from cogmem.consolidation.select import filter_manifest_eligible
from cogmem.memory.episodic_store import infer_error_family
from cogmem.memory.memory_bank import MemoryBank
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


def _rerank_retry_skill_cards(
    task: dict,
    skill_store: SkillStore,
    *,
    previous_skill_ids: list[str],
    error_text: str,
    skill_top_k: int,
) -> list[dict]:
    retry_record = task_to_skill_record(task)
    retry_record["error"] = error_text
    retry_record["error_family"] = infer_error_family(error_text)
    if error_text:
        retry_record["task_description"] = (
            f"{retry_record.get('task_description', '')}\n\nObserved failure:\n{error_text[:500]}"
        ).strip()
    reranked = rank_skill_cards_for_record(
        skill_store,
        retry_record,
        limit=max(skill_top_k + len(previous_skill_ids), skill_top_k),
        promoted_only=True,
    )
    preferred = [card for card in reranked if card["skill_id"] not in set(previous_skill_ids)]
    if preferred:
        return preferred[:skill_top_k]
    return reranked[:skill_top_k]


def _run_task_with_skill_cards(
    task: dict,
    llm_client,
    *,
    retrieved_skill_cards: list[dict],
    skill_store: SkillStore | None = None,
    skill_top_k: int = 1,
    eval_mode: str = "subprocess",
    eval_timeout: int = 30,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_attempts: int = 1,
) -> dict:
    trajectory = []
    retrieved_skill_history: list[list[str]] = []
    current_skill_cards = list(retrieved_skill_cards)

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            prev_error = trajectory[-1].get("error", "Unknown error")
            if skill_store is not None:
                reranked = _rerank_retry_skill_cards(
                    task,
                    skill_store,
                    previous_skill_ids=[
                        skill_id
                        for history in retrieved_skill_history
                        for skill_id in history
                    ],
                    error_text=prev_error,
                    skill_top_k=skill_top_k,
                )
                if reranked:
                    current_skill_cards = reranked
            retry_prompt = (
                f"Your previous attempt failed with this error:\n"
                f"{prev_error[:500]}\n\n"
                f"Please fix the code and try again."
            )

        messages = _augment_messages_with_skill_cards(
            format_messages(task, use_instruct=True),
            current_skill_cards,
        )
        if attempt > 1:
            messages.append({"role": "user", "content": retry_prompt})
        retrieved_skill_history.append([card["skill_id"] for card in current_skill_cards])

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
        "retrieved_skill_ids": list(dict.fromkeys(skill_id for history in retrieved_skill_history for skill_id in history)),
        "retrieved_skill_history": retrieved_skill_history,
    }


def evaluate_new_arch_model(
    eval_tasks: list[dict],
    *,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: str | None = None,
    skill_cards_path: str | None = None,
    skill_top_k: int = 1,
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
    skill_store = SkillStore.load(skill_cards_path) if skill_cards_path else None
    selected_skill_ids: Counter[str] = Counter()

    try:
        model, tokenizer, llm_client = load_new_arch_runtime(
            model_name=model_name,
            adapter_path=adapter_path,
        )

        for idx, task in enumerate(tasks):
            retrieved_skill_cards = (
                rank_skill_cards_for_task(
                    skill_store,
                    task,
                    limit=skill_top_k,
                    promoted_only=True,
                )
                if skill_store is not None
                else []
            )
            if retrieved_skill_cards:
                episode = _run_task_with_skill_cards(
                    task,
                    llm_client,
                    retrieved_skill_cards=retrieved_skill_cards,
                    skill_store=skill_store,
                    skill_top_k=skill_top_k,
                    eval_mode="subprocess",
                    eval_timeout=eval_timeout,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_attempts=max_attempts,
                )
            else:
                episode = run_single_task(
                    task,
                    llm_client,
                    eval_mode="subprocess",
                    eval_timeout=eval_timeout,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_attempts=max_attempts,
                )
            success = bool(episode.get("success"))
            if success:
                passed += 1
            for skill_id in episode.get("retrieved_skill_ids", []) or []:
                selected_skill_ids[skill_id] += 1

            rows.append(
                {
                    "task_id": task["task_id"],
                    "passed": success,
                    "attempts": int(episode.get("num_attempts", 0) or 0),
                    "error": episode.get("error"),
                    "retrieved_skill_ids": list(episode.get("retrieved_skill_ids", []) or []),
                    "retrieved_skill_history": list(episode.get("retrieved_skill_history", []) or []),
                }
            )

            if verbose and ((idx + 1) % 10 == 0 or idx < 5):
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed * 3600 if elapsed > 0 else 0.0
                label_parts = ["adapter" if adapter_path else "base"]
                if skill_store is not None:
                    label_parts.append("skill")
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
        "skill_cards_path": skill_cards_path,
        "selected_skill_ids": dict(selected_skill_ids),
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
        "skill_utility": _build_skill_utility(tasks, baseline_rows, candidate_rows),
    }


def persist_route_skill_utility(skill_cards_path: str | None, comparisons: dict[str, Any]) -> dict[str, Any]:
    if not skill_cards_path:
        return {"skills_path": None, "updated_skills": 0, "updated_routes": []}
    store = SkillStore.load(skill_cards_path)
    updated_routes: list[str] = []
    updated_skills = 0
    for route_name, route_result in dict(comparisons or {}).items():
        skill_utility = dict(route_result.get("skill_utility", {}) or {})
        if not skill_utility:
            continue
        updated_skills += store.apply_runtime_utility(skill_utility, route_name=route_name)
        updated_routes.append(route_name)
    if updated_routes:
        store.save(skill_cards_path)
    return {
        "skills_path": skill_cards_path,
        "updated_skills": updated_skills,
        "updated_routes": updated_routes,
    }


def compare_new_arch_base_vs_adapter(
    eval_tasks: list[dict],
    *,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: str,
    task_limit: int | None = None,
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
    adapter_path: str | None = None,
    skill_top_k: int = 1,
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
        max_tokens=max_tokens,
        temperature=temperature,
        max_attempts=max_attempts,
        eval_timeout=eval_timeout,
        verbose=verbose,
    )
    if skill_cards_path:
        results["base_plus_skill"] = evaluate_new_arch_model(
            tasks,
            model_name=model_name,
            adapter_path=None,
            skill_cards_path=skill_cards_path,
            skill_top_k=skill_top_k,
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
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
            eval_timeout=eval_timeout,
            verbose=verbose,
        )
    if adapter_path and skill_cards_path:
        results["adapter_plus_skill"] = evaluate_new_arch_model(
            tasks,
            model_name=model_name,
            adapter_path=adapter_path,
            skill_cards_path=skill_cards_path,
            skill_top_k=skill_top_k,
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
    if "adapter" in results:
        comparisons["adapter_vs_base"] = _compare_route_results(
            tasks,
            results["base"],
            results["adapter"],
        )
    if "adapter_plus_skill" in results and "adapter" in results:
        comparisons["adapter_plus_skill_vs_adapter"] = _compare_route_results(
            tasks,
            results["adapter"],
            results["adapter_plus_skill"],
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
            0 if card.get("status") == "promoted" else 1,
            -float(card.get("confidence", 0.0) or 0.0),
            -float(card.get("transfer_gain", 0.0) or 0.0),
            -int(card.get("source_episode_count", 0) or 0),
        ),
    )
    rows = []
    for card in cards[:limit]:
        validation = dict(card.get("validation", {}) or {})
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
            "triggers": list(card.get("triggers", [])[:5]),
            "plan_steps": list(card.get("plan_steps", [])[:3]),
            "activation_conditions": list(card.get("activation_conditions", [])[:3]),
            "stop_conditions": list(card.get("stop_conditions", [])[:3]),
            "anti_patterns": list(card.get("anti_patterns", [])[:3]),
            "manifest_ids": list(card.get("manifest_ids", [])),
        })
    return rows


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
            adapter_path=None,
            skill_top_k=cogmem_config.skill_retrieval_top_k,
            task_limit=eval_task_limit or cogmem_config.skill_route_gate_task_limit,
            max_tokens=2048,
            temperature=0.0,
            max_attempts=1,
            eval_timeout=30,
            verbose=False,
        )
        route_gate = route_eval.get("comparisons", {}).get("base_plus_skill_vs_base", {})
        persist_route_skill_utility(skill_cards_path, route_eval.get("comparisons", {}))
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
    "persist_route_skill_utility",
    "run_new_arch_episode_collection",
    "build_new_arch_skill_cards",
    "run_new_arch_qstar_cycle",
]

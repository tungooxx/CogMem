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
from cogmem.benchmarks.bigcodebench.experiment import materialize_split_views
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
from cogmem.consolidation.proceduralize import build_skill_cards
from cogmem.consolidation.select import filter_manifest_eligible
from cogmem.memory.memory_bank import MemoryBank
from cogmem.memory.skill_store import SkillStore


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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, TransformersChatClient(model, tokenizer)


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
            "confidence": float(card.get("confidence", 0.0) or 0.0),
            "transfer_gain": float(card.get("transfer_gain", 0.0) or 0.0),
            "negative_transfer_rate": float(card.get("negative_transfer_rate", 0.0) or 0.0),
            "matched_episodes": int(validation.get("matched_episodes", 0) or 0),
            "success_rate": float(validation.get("success_rate", 0.0) or 0.0),
            "triggers": list(card.get("triggers", [])[:5]),
            "plan_steps": list(card.get("plan_steps", [])[:3]),
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
) -> dict[str, Any]:
    from cogmem.consolidation.pipeline import run_qstar_cycle

    return run_qstar_cycle(memory_bank_path, cogmem_config, cycle=cycle, run_task_fn=None)


__all__ = [
    "NewArchitectureExperimentConfig",
    "TransformersChatClient",
    "load_new_arch_tasks",
    "prepare_new_arch_task_split",
    "load_new_arch_runtime",
    "run_new_arch_episode_collection",
    "build_new_arch_skill_cards",
    "run_new_arch_qstar_cycle",
]

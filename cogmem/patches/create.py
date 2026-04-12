"""Create cognitive patches from experience contrasts.

Method A: From fail-to-pass contrast (Approach 1: micro-finetune)
  Train a tiny LoRA on (prompt, passed_code).
  The resulting adapter captures "how to solve this type of problem."

Method B: From programmatic mutation
  Mutate passing code, test if it breaks, create patch from contrast.

Method C: From cluster of similar experiences
  Group similar episodes, train one patch on the cluster.
"""

import gc
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from cogmem.patches.patch import CognitivePatch

DEFAULT_PATCH_RANK = 2
DEFAULT_PATCH_TRAIN_STEPS = 30
DEFAULT_PATCH_LR = 2e-3
DEFAULT_PATCH_LOG_EVERY_STEPS = 1


@dataclass
class PatchTrainingStats:
    """Compact training trace for one micro-finetune run."""

    total_steps: int
    final_loss: float | None
    loss_history: list[dict[str, float | int | None]] = field(default_factory=list)


class _PatchTrainingTraceCallback(TrainerCallback):
    """Collect and optionally print loss entries during micro-training."""

    def __init__(self, total_steps: int, show_progress: bool = False):
        self.total_steps = total_steps
        self.show_progress = show_progress
        self.loss_history: list[dict[str, float | int | None]] = []
        self._seen_steps: set[int] = set()

    def on_log(self, args, state, control, logs=None, **kwargs):
        log_entry = dict(logs or {})
        if "epoch" not in log_entry and state.epoch is not None:
            log_entry["epoch"] = state.epoch
        entry = _normalize_loss_entry(
            log_entry,
            fallback_step=state.global_step,
        )
        if entry is None:
            return

        step = int(entry["step"])
        if step in self._seen_steps:
            return

        self._seen_steps.add(step)
        self.loss_history.append(entry)

        if self.show_progress:
            epoch = entry["epoch"]
            epoch_text = "n/a" if epoch is None else f"{epoch:.2f}"
            print(
                f"    step {step}/{self.total_steps} | "
                f"epoch={epoch_text} | loss={entry['loss']:.4f}"
            )


def create_patch_from_contrast(
    base_model,
    tokenizer,
    task_prompt: str,
    failed_code: str,
    passed_code: str,
    patch_id: str,
    rank: int = DEFAULT_PATCH_RANK,
    n_steps: int = DEFAULT_PATCH_TRAIN_STEPS,
    lr: float = DEFAULT_PATCH_LR,
    show_progress: bool = False,
    log_every_steps: int = DEFAULT_PATCH_LOG_EVERY_STEPS,
    return_stats: bool = False,
) -> CognitivePatch | tuple[CognitivePatch, PatchTrainingStats]:
    """Create a patch by micro-finetuning on the passing example.

    Approach 1: Train a fresh LoRA for a few steps on
    (prompt, passed_code). The resulting adapter captures
    "how to solve this type of problem."

    Args:
        base_model: Frozen base model (4-bit quantized, on GPU).
        tokenizer: Tokenizer for the model.
        task_prompt: The task description.
        failed_code: Code that failed (for metadata, not used in training).
        passed_code: Code that passed (training target).
        patch_id: Unique identifier for this patch.
        rank: LoRA rank (2-4).
        n_steps: Number of optimizer steps.
        lr: Learning rate for micro-training.
        show_progress: If True, print compact per-step loss lines.
        log_every_steps: Logging interval for trainer loss events.
        return_stats: If True, return ``(patch, stats)``.

    Returns:
        CognitivePatch with tiny LoRA weights, optionally with training stats.
    """
    _ensure_clean_base_model(base_model)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    from cogmem.benchmarks.bigcodebench.prompts import SYSTEM_PROMPT
    from datasets import Dataset

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
        {"role": "assistant", "content": passed_code},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    tok = tokenizer(text, truncation=True, max_length=1024, padding=False)
    tok["labels"] = tok["input_ids"].copy()
    dataset = Dataset.from_list([tok])

    tmp_dir = tempfile.mkdtemp(prefix=f"cogmem_patch_{patch_id}_")
    try:
        stats = _train_patch_adapter(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            output_dir=tmp_dir,
            n_steps=n_steps,
            lr=lr,
            show_progress=show_progress,
            log_every_steps=log_every_steps,
        )
        lora_weights = _extract_lora_weights(model)
    finally:
        _cleanup_patch_training(model, tmp_dir)

    desc = _describe_contrast(failed_code, passed_code)
    patch = CognitivePatch(
        patch_id=patch_id,
        embedding=[],
        lora_weights=lora_weights,
        rank=rank,
        source_task_id="",
        source_type="fail_to_pass",
        description=desc,
    )
    return (patch, stats) if return_stats else patch


def create_patch_from_example(
    base_model,
    tokenizer,
    task_prompt: str,
    code: str,
    patch_id: str,
    rank: int = DEFAULT_PATCH_RANK,
    n_steps: int = DEFAULT_PATCH_TRAIN_STEPS,
    lr: float = DEFAULT_PATCH_LR,
    show_progress: bool = False,
    log_every_steps: int = DEFAULT_PATCH_LOG_EVERY_STEPS,
    return_stats: bool = False,
) -> CognitivePatch | tuple[CognitivePatch, PatchTrainingStats]:
    """Create a patch from a single successful example."""
    return create_patch_from_contrast(
        base_model,
        tokenizer,
        task_prompt,
        "",
        code,
        patch_id,
        rank=rank,
        n_steps=n_steps,
        lr=lr,
        show_progress=show_progress,
        log_every_steps=log_every_steps,
        return_stats=return_stats,
    )


def create_patch_from_cluster(
    base_model,
    tokenizer,
    examples: list[dict],
    patch_id: str,
    rank: int = 4,
    n_steps: int = DEFAULT_PATCH_TRAIN_STEPS,
    lr: float = 5e-4,
    show_progress: bool = False,
    log_every_steps: int = DEFAULT_PATCH_LOG_EVERY_STEPS,
    return_stats: bool = False,
) -> CognitivePatch | tuple[CognitivePatch, PatchTrainingStats]:
    """Create a patch from a cluster of similar successful episodes.

    Args:
        examples: List of {"prompt": str, "code": str} dicts.

    Raises:
        ValueError: If examples list is empty.
    """
    if not examples:
        raise ValueError("Cannot create cluster patch from empty examples list")
    _ensure_clean_base_model(base_model)

    from cogmem.benchmarks.bigcodebench.prompts import SYSTEM_PROMPT
    from datasets import Dataset

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    data = []
    for ex in examples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["code"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tok = tokenizer(text, truncation=True, max_length=1024, padding=False)
        tok["labels"] = tok["input_ids"].copy()
        data.append(tok)

    dataset = Dataset.from_list(data)

    tmp_dir = tempfile.mkdtemp(prefix=f"cogmem_patch_{patch_id}_")
    try:
        stats = _train_patch_adapter(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            output_dir=tmp_dir,
            n_steps=n_steps,
            lr=lr,
            gradient_accumulation_steps=min(4, len(data)),
            show_progress=show_progress,
            log_every_steps=log_every_steps,
        )
        lora_weights = _extract_lora_weights(model)
    finally:
        _cleanup_patch_training(model, tmp_dir)

    patch = CognitivePatch(
        patch_id=patch_id,
        embedding=[],
        lora_weights=lora_weights,
        rank=rank,
        source_type="cluster",
        description=f"Cluster of {len(examples)} similar episodes",
    )
    return (patch, stats) if return_stats else patch


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _ensure_clean_base_model(base_model) -> None:
    """Fail fast if a prior PEFT run left LoRA layers attached to the base model."""
    if not hasattr(base_model, "modules"):
        return
    if any(isinstance(module, BaseTunerLayer) for module in base_model.modules()):
        raise RuntimeError(
            "base_model already contains PEFT layers from a previous patch training run. "
            "Reload the base model and rerun patch creation."
        )

def _train_patch_adapter(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    n_steps: int,
    lr: float,
    gradient_accumulation_steps: int = 1,
    show_progress: bool = False,
    log_every_steps: int = DEFAULT_PATCH_LOG_EVERY_STEPS,
) -> PatchTrainingStats:
    """Train one tiny adapter and return a compact loss trace."""
    trace_callback = _PatchTrainingTraceCallback(
        total_steps=n_steps,
        show_progress=show_progress,
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=n_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=max(1, gradient_accumulation_steps),
        learning_rate=lr,
        lr_scheduler_type="cosine",  # smooth decay: high LR early, gentle finish
        logging_strategy="steps",
        logging_steps=max(1, log_every_steps),
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        fp16=False,
        seed=42,
        disable_tqdm=True,  # use compact text logs instead of the notebook widget
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        callbacks=[trace_callback],
    )
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)

    train_result = trainer.train()
    loss_history = trace_callback.loss_history or _extract_loss_history(
        trainer.state.log_history
    )
    final_loss = _extract_final_loss(trainer.state.log_history)
    if final_loss is None and getattr(train_result, "training_loss", None) is not None:
        final_loss = float(train_result.training_loss)

    return PatchTrainingStats(
        total_steps=int(trainer.state.global_step or n_steps),
        final_loss=final_loss,
        loss_history=loss_history,
    )


def _cleanup_patch_training(model, tmp_dir: str) -> None:
    """Remove adapter state and temporary files."""
    can_unload = hasattr(model, "base_model") and hasattr(model.base_model, "unload")
    can_disable = hasattr(model, "disable_adapter_layers")
    can_delete = hasattr(model, "delete_adapter")
    cleanup_errors: list[tuple[str, Exception]] = []

    def run_disable_and_delete() -> None:
        if can_disable:
            try:
                model.disable_adapter_layers()
            except Exception as e:
                cleanup_errors.append(("disable_adapter_layers", e))
        if can_delete:
            try:
                model.delete_adapter("default")
            except Exception as e:
                cleanup_errors.append(("delete_adapter", e))

    try:
        if can_unload:
            try:
                model.base_model.unload()
            except Exception as e:
                cleanup_errors.append(("base_model.unload", e))
                run_disable_and_delete()
        else:
            run_disable_and_delete()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

        if Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir)

    if cleanup_errors:
        details = ", ".join(
            f"{name}: {type(err).__name__}: {err}" for name, err in cleanup_errors
        )
        raise RuntimeError(f"Patch cleanup failed ({details})") from cleanup_errors[0][1]


def _normalize_loss_entry(
    entry: dict,
    fallback_step: int | None = None,
) -> dict[str, float | int | None] | None:
    """Normalize one trainer log entry into the public loss-history shape."""
    if "loss" not in entry:
        return None

    step = entry.get("step", fallback_step)
    if step is None:
        return None

    epoch = entry.get("epoch")
    return {
        "step": int(step),
        "epoch": None if epoch is None else float(epoch),
        "loss": float(entry["loss"]),
    }


def _extract_loss_history(
    log_history: list[dict] | None,
) -> list[dict[str, float | int | None]]:
    """Extract monotonic loss entries from Trainer log history."""
    history: list[dict[str, float | int | None]] = []
    seen_steps: set[int] = set()
    next_step = 1

    for entry in log_history or []:
        normalized = _normalize_loss_entry(entry, fallback_step=next_step)
        if normalized is None:
            continue

        step = int(normalized["step"])
        if step in seen_steps:
            continue

        history.append(normalized)
        seen_steps.add(step)
        next_step = step + 1

    return history


def _extract_final_loss(log_history: list[dict] | None) -> float | None:
    """Get the final logged loss for a training run."""
    for entry in reversed(log_history or []):
        if "loss" in entry:
            return float(entry["loss"])
        if "train_loss" in entry:
            return float(entry["train_loss"])
    return None


def _extract_lora_weights(peft_model) -> dict:
    """Extract LoRA A and B matrices from a peft model."""
    weights = {}
    for name, param in peft_model.named_parameters():
        if "lora_" in name and param.requires_grad:
            parts = name.split(".")
            layer_key = ".".join(
                p for p in parts if p not in ("lora_A", "lora_B", "default")
            )
            for prefix in ("base_model.model.", "base_model."):
                if layer_key.startswith(prefix):
                    layer_key = layer_key[len(prefix):]
                    break
            ab = "A" if "lora_A" in name else "B"

            if layer_key not in weights:
                weights[layer_key] = {}
            weights[layer_key][ab] = param.detach().cpu().clone()

    return weights


def _describe_contrast(failed_code: str, passed_code: str) -> str:
    """Generate a short description of what changed between fail and pass."""
    if not failed_code:
        return "learned pattern from successful example"

    failed_lines = set(failed_code.strip().split("\n"))
    passed_lines = set(passed_code.strip().split("\n"))

    added = passed_lines - failed_lines
    removed = failed_lines - passed_lines

    parts = []
    if added:
        sample = next(iter(added)).strip()[:60]
        parts.append(f"added: {sample}")
    if removed:
        sample = next(iter(removed)).strip()[:60]
        parts.append(f"removed: {sample}")

    return "; ".join(parts) if parts else "code restructured"

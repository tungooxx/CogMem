"""Create cognitive patches from experience contrasts.

Method A: From fail→pass contrast (Approach 1: micro-finetune)
  Train rank-2 LoRA for 5-10 steps on (prompt, passed_code).
  The resulting LoRA captures "how to solve this type of problem."

Method B: From programmatic mutation
  Mutate passing code, test if it breaks, create patch from contrast.

Method C: From cluster of similar experiences
  Group similar episodes, train rank-4 LoRA on the cluster.
"""

import gc
import tempfile
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from cogmem.patches.patch import CognitivePatch


def create_patch_from_contrast(
    base_model,
    tokenizer,
    task_prompt: str,
    failed_code: str,
    passed_code: str,
    patch_id: str,
    rank: int = 2,
    n_steps: int = 10,
    lr: float = 1e-3,
) -> CognitivePatch:
    """Create a patch by micro-finetuning on the passing example.

    Approach 1: Train a fresh rank-2 LoRA for a few steps on
    (prompt, passed_code). The resulting tiny adapter captures
    "how to solve this type of problem."

    Args:
        base_model: Frozen base model (4-bit quantized, on GPU).
        tokenizer: Tokenizer for the model.
        task_prompt: The task description.
        failed_code: Code that failed (for metadata, not used in training).
        passed_code: Code that passed (training target).
        patch_id: Unique identifier for this patch.
        rank: LoRA rank (2-4).
        n_steps: Number of gradient steps (5-10).
        lr: Learning rate (high, since few steps).

    Returns:
        CognitivePatch with tiny LoRA weights.
    """
    # Create a fresh rank-2 LoRA (zero-initialized)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # base_model must have prepare_model_for_kbit_training called ONCE by caller
    model = get_peft_model(base_model, lora_config)
    # Disable gradient checkpointing for micro-training (5 steps, not worth the overhead)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # Prepare single training example
    from cogmem.benchmarks.bigcodebench.prompts import SYSTEM_PROMPT

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

    from datasets import Dataset

    dataset = Dataset.from_list([tok])

    # Train for a few steps
    tmp_dir = tempfile.mkdtemp(prefix=f"cogmem_patch_{patch_id}_")
    training_args = TrainingArguments(
        output_dir=tmp_dir,
        num_train_epochs=n_steps,  # n_steps on 1 example = n_steps epochs
        per_device_train_batch_size=1,
        learning_rate=lr,
        logging_steps=n_steps,
        save_strategy="no",
        report_to="none",
        fp16=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    trainer.train()

    # Extract LoRA weights before cleanup
    lora_weights = _extract_lora_weights(model)

    # Cleanup: properly remove PEFT adapter to undo hooks on base_model.
    # merge_and_unload() would permanently modify base_model weights.
    # delete_adapter + get_base_model cleanly restores the original model.
    del trainer
    try:
        model.disable_adapter_layers()
        model.delete_adapter("default")
    except Exception as e:
        print(f"Warning: adapter cleanup failed: {type(e).__name__}: {e}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Clean up temp dir
    import shutil
    if Path(tmp_dir).exists():
        shutil.rmtree(tmp_dir)

    # Create description from the contrast
    desc = _describe_contrast(failed_code, passed_code)

    return CognitivePatch(
        patch_id=patch_id,
        embedding=[],  # caller should set this
        lora_weights=lora_weights,
        rank=rank,
        source_task_id="",  # caller should set this
        source_type="fail_to_pass",
        description=desc,
    )


def create_patch_from_example(
    base_model,
    tokenizer,
    task_prompt: str,
    code: str,
    patch_id: str,
    rank: int = 2,
    n_steps: int = 5,
    lr: float = 1e-3,
) -> CognitivePatch:
    """Create a patch from a single successful example.

    Simpler than contrast — just learn this one pattern.
    """
    return create_patch_from_contrast(
        base_model, tokenizer,
        task_prompt, "", code,
        patch_id, rank, n_steps, lr,
    )


def create_patch_from_cluster(
    base_model,
    tokenizer,
    examples: list[dict],
    patch_id: str,
    rank: int = 4,
    n_steps: int = 20,
    lr: float = 5e-4,
) -> CognitivePatch:
    """Create a patch from a cluster of similar successful episodes.

    More signal than single-experience patches, captures shared pattern.

    Args:
        examples: List of {"prompt": str, "code": str} dicts.

    Raises:
        ValueError: If examples list is empty.
    """
    if not examples:
        raise ValueError("Cannot create cluster patch from empty examples list")

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

    from peft import prepare_model_for_kbit_training

    prepare_model_for_kbit_training(base_model)
    model = get_peft_model(base_model, lora_config)

    # Prepare training data
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
    training_args = TrainingArguments(
        output_dir=tmp_dir,
        num_train_epochs=max(1, n_steps // len(data)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=min(4, len(data)),
        learning_rate=lr,
        logging_steps=n_steps,
        save_strategy="no",
        report_to="none",
        fp16=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    trainer.train()
    lora_weights = _extract_lora_weights(model)

    # Cleanup: properly remove PEFT adapter to undo hooks on base_model.
    del trainer
    try:
        model.disable_adapter_layers()
        model.delete_adapter("default")
    except Exception as e:
        print(f"Warning: adapter cleanup failed: {type(e).__name__}: {e}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    import shutil
    if Path(tmp_dir).exists():
        shutil.rmtree(tmp_dir)

    return CognitivePatch(
        patch_id=patch_id,
        embedding=[],
        lora_weights=lora_weights,
        rank=rank,
        source_type="cluster",
        description=f"Cluster of {len(examples)} similar episodes",
    )


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _extract_lora_weights(peft_model) -> dict:
    """Extract LoRA A and B matrices from a peft model."""
    weights = {}
    for name, param in peft_model.named_parameters():
        if "lora_" in name and param.requires_grad:
            # name like: model.layers.0.self_attn.q_proj.lora_A.default.weight
            parts = name.split(".")
            # Extract layer identifier: strip only lora_A/lora_B/default tokens,
            # keep ".weight" so the key matches model.named_parameters() in compose.py
            layer_key = ".".join(p for p in parts if p not in ("lora_A", "lora_B", "default"))
            ab = "A" if "lora_A" in name else "B"

            if layer_key not in weights:
                weights[layer_key] = {}
            weights[layer_key][ab] = param.detach().cpu().clone()

    return weights


def _describe_contrast(failed_code: str, passed_code: str) -> str:
    """Generate a short description of what changed between fail and pass."""
    if not failed_code:
        return "learned pattern from successful example"

    # Simple diff: find lines that differ
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

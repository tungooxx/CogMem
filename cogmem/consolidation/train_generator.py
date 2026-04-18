"""Generator DoRA training: two-stage SFT then DPO.

Stage 1 (SFT): Learn good code patterns from high-Q episodes.
Stage 2 (DPO): Learn to AVOID bad patterns using Q-value preference pairs.

DPO builds on SFT — starts from the SFT checkpoint and sharpens it.
This is the standard recipe (DeepSeek, Llama): SFT first, then DPO.
"""

import gc
import inspect
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from cogmem.consolidation.adapter_registry import register_adapter_artifact


def _make_bnb_config(bits: int) -> BitsAndBytesConfig:
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def _load_model_for_training(model_name: str, *, bits: int):
    model_kwargs = {
        "device_map": {"": 0},
        "torch_dtype": torch.float16,
        "attn_implementation": "eager",
    }
    if bits > 0:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=_make_bnb_config(bits),
                **model_kwargs,
            )
            return model, True
        except Exception as exc:
            print(
                f"  Quantized training load failed ({type(exc).__name__}: {exc}); "
                "retrying with plain fp16 weights."
            )
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, False


def _uses_kbit_weights(model) -> bool:
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if getattr(candidate, "is_loaded_in_4bit", False) or getattr(candidate, "is_loaded_in_8bit", False):
            return True
    return False


def _precision_kwargs() -> dict[str, bool]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": True}


class _KbitSafeTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        if _uses_kbit_weights(model):
            return model
        return super()._move_model_to_device(model, device)


def _resolve_use_dora(requested: bool) -> bool:
    if not requested:
        return False
    if "use_dora" in inspect.signature(LoraConfig.__init__).parameters:
        return True
    print("  PEFT does not support DoRA in this environment; falling back to plain LoRA.")
    return False


def _make_lora_config(config, use_dora: bool = True) -> LoraConfig:
    resolved_use_dora = _resolve_use_dora(use_dora)
    kwargs = dict(
        r=config.generator_rank,
        lora_alpha=config.generator_alpha,
        lora_dropout=config.generator_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    if resolved_use_dora:
        kwargs["use_dora"] = True
    return LoraConfig(**kwargs)


def train_generator_full(
    sft_episodes: list[dict],
    pref_dataset: Dataset | None,
    config,
    cycle: int = 0,
    *,
    source_skill_card_ids: list[str] | None = None,
    training_manifest_ids: list[str] | None = None,
    compatible_families: list[str] | None = None,
    registry_path: str | None = None,
    adapter_role: str = "global",
    dev_gain: float = 0.0,
    sft_pairs: list[dict] | None = None,
) -> str:
    """Two-stage generator training: SFT then DPO.

    Returns path to final adapter (DPO if enough pairs, else SFT only).
    """
    sft_path, sft_loss = train_generator_sft(
        sft_episodes,
        config,
        cycle,
        source_skill_card_ids=source_skill_card_ids,
        training_manifest_ids=training_manifest_ids,
        compatible_families=compatible_families,
        registry_path=registry_path,
        adapter_role=adapter_role,
        dev_gain=dev_gain,
        sft_pairs=sft_pairs,
    )
    print(f"  Stage 1 (SFT): loss={sft_loss}, path={sft_path}")

    if pref_dataset is not None and len(pref_dataset) >= config.min_dpo_pairs:
        dpo_path = train_generator_dpo(
            pref_dataset,
            config,
            cycle,
            sft_path,
            source_skill_card_ids=source_skill_card_ids,
            training_manifest_ids=training_manifest_ids,
            compatible_families=compatible_families,
            registry_path=registry_path,
            adapter_role=adapter_role,
            dev_gain=dev_gain,
        )
        print(f"  Stage 2 (DPO): path={dpo_path}")
        return dpo_path

    print(f"  Skipping DPO: {len(pref_dataset) if pref_dataset else 0} pairs "
          f"(need >= {config.min_dpo_pairs})")
    return sft_path


# -------------------------------------------------------------------------
# Stage 1: SFT on high-Q episodes
# -------------------------------------------------------------------------

def train_generator_sft(
    high_episodes: list[dict],
    config,
    cycle: int = 0,
    *,
    source_skill_card_ids: list[str] | None = None,
    training_manifest_ids: list[str] | None = None,
    compatible_families: list[str] | None = None,
    registry_path: str | None = None,
    adapter_role: str = "global",
    dev_gain: float = 0.0,
    sft_pairs: list[dict] | None = None,
) -> tuple[str, float | None]:
    """Train DoRA adapter via SFT on high-Q episodes."""
    output_dir = str(Path(config.adapters_dir) / f"generator_sft_v{cycle}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    precision_kwargs = _precision_kwargs()

    tokenizer = AutoTokenizer.from_pretrained(config.active_model_hf)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, use_kbit = _load_model_for_training(
        config.active_model_hf,
        bits=config.quantization_bits,
    )
    if use_kbit:
        model = prepare_model_for_kbit_training(model)

    lora_config = _make_lora_config(config, use_dora=config.use_dora)

    gc.collect()
    torch.cuda.empty_cache()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build SFT dataset from promoted skill-card pairs, with episode fallback.
    dataset = _prepare_sft_dataset(sft_pairs or high_episodes, tokenizer, config)
    print(f"  SFT dataset: {len(dataset)} examples")

    gc.collect()
    torch.cuda.empty_cache()

    trainer = _KbitSafeTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.generator_sft_epochs,
            per_device_train_batch_size=config.generator_batch_size,
            gradient_accumulation_steps=4,
            learning_rate=config.generator_sft_lr,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            **precision_kwargs,
            optim="paged_adamw_8bit" if use_kbit else "adamw_torch",
            report_to="none",
            seed=config.seed,
            gradient_checkpointing=True,
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    print("  Training generator SFT...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    final_loss = _extract_final_loss(trainer)

    with open(f"{output_dir}/training_log.json", "w") as f:
        json.dump({
            "type": "generator_sft",
            "cycle": cycle,
            "base_model": config.active_model_hf,
            "episodes": len(high_episodes),
            "skill_sft_pairs": len(sft_pairs or []),
            "training_source": "skill_cards" if sft_pairs else "episodes",
            "dataset_size": len(dataset),
            "final_loss": final_loss,
            "use_dora": config.use_dora,
        }, f, indent=2)

    if registry_path:
        register_adapter_artifact(
            registry_path,
            output_dir,
            base_model=config.active_model_hf,
            adapter_role=adapter_role,
            training_manifest_ids=training_manifest_ids or [],
            source_skill_card_ids=source_skill_card_ids or [],
            compatible_families=compatible_families or [],
            dev_gain=dev_gain,
            metadata={
                "trainer": "generator_sft",
                "cycle": cycle,
                "final_loss": final_loss,
            },
        )

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir, final_loss


# -------------------------------------------------------------------------
# Stage 2: DPO on Q-value preference pairs
# -------------------------------------------------------------------------

def train_generator_dpo(
    pref_dataset: Dataset,
    config,
    cycle: int = 0,
    sft_adapter_path: str | None = None,
    *,
    source_skill_card_ids: list[str] | None = None,
    training_manifest_ids: list[str] | None = None,
    compatible_families: list[str] | None = None,
    registry_path: str | None = None,
    adapter_role: str = "global",
    dev_gain: float = 0.0,
) -> str:
    """DPO starting from SFT adapter. Learns what to avoid.

    The preference pairs come from Q-values:
    - High-Q code is "chosen" (what the model should produce)
    - Low-Q code is "rejected" (what the model should avoid)
    """
    from trl import DPOConfig, DPOTrainer

    output_dir = str(Path(config.adapters_dir) / f"generator_v{cycle}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    precision_kwargs = _precision_kwargs()

    # Load base model
    base_model, use_kbit = _load_model_for_training(
        config.active_model_hf,
        bits=config.quantization_bits,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.active_model_hf)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Merge SFT adapter into base (DPO continues from SFT)
    if sft_adapter_path:
        model = PeftModel.from_pretrained(base_model, sft_adapter_path)
        model = model.merge_and_unload()
    else:
        model = base_model

    if use_kbit:
        model = prepare_model_for_kbit_training(model)

    gc.collect()
    torch.cuda.empty_cache()

    lora_config = _make_lora_config(config, use_dora=config.use_dora)

    class _KbitSafeDPOTrainer(DPOTrainer):
        def _move_model_to_device(self, model, device):
            if _uses_kbit_weights(model):
                return model
            return super()._move_model_to_device(model, device)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.generator_dpo_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=config.generator_dpo_lr,
        beta=config.generator_dpo_beta,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        **precision_kwargs,
        optim="paged_adamw_8bit" if use_kbit else "adamw_torch",
        seed=config.seed,
        report_to="none",
        max_length=2048,
        max_prompt_length=512,
        gradient_checkpointing=True,
    )

    trainer = _KbitSafeDPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=pref_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"  Training generator DPO ({len(pref_dataset)} pairs)...")
    trainer.train()
    trainer.save_model(output_dir)

    final_loss = _extract_final_loss(trainer)

    with open(f"{output_dir}/training_log.json", "w") as f:
        json.dump({
            "type": "generator_sft_dpo",
            "cycle": cycle,
            "base_model": config.active_model_hf,
            "sft_base": sft_adapter_path,
            "preference_pairs": len(pref_dataset),
            "dpo_beta": config.generator_dpo_beta,
            "final_loss": final_loss,
            "use_dora": config.use_dora,
        }, f, indent=2)

    if registry_path:
        register_adapter_artifact(
            registry_path,
            output_dir,
            base_model=config.active_model_hf,
            adapter_role=adapter_role,
            training_manifest_ids=training_manifest_ids or [],
            source_skill_card_ids=source_skill_card_ids or [],
            compatible_families=compatible_families or [],
            dev_gain=dev_gain,
            metadata={
                "trainer": "generator_dpo",
                "cycle": cycle,
                "sft_adapter_path": sft_adapter_path,
                "final_loss": final_loss,
            },
        )

    del model, trainer
    if sft_adapter_path:
        del base_model
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _prepare_sft_dataset(
    examples: list[dict], tokenizer, config,
) -> Dataset:
    """Convert high-Q episodes or prebuilt SFT pairs to tokenized data.

    CRITICAL: Uses the SAME prompt format as evaluation (format_messages)
    so the model learns to respond to the exact same prompts it sees at test time.
    """
    from cogmem.benchmarks.bigcodebench.prompts import SYSTEM_PROMPT

    pairs = []
    for example in examples:
        if "messages" in example:
            pairs.append({"messages": example["messages"]})
            continue

        if "instruction" in example and "response" in example:
            instruction = example.get("instruction", "")
            response = example.get("response", "")
        else:
            response = (
                example.get("final_code")
                or example.get("generated_code")
                or example.get("script")
            )
            instruction = example.get("task_description", "")

        if not instruction or not response:
            continue

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
        })

    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tok = tokenizer(
            text, truncation=True,
            max_length=config.generator_max_seq_length, padding=False,
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    dataset = Dataset.from_list(pairs)
    return dataset.map(tokenize, remove_columns=["messages"])


def _extract_final_loss(trainer) -> float | None:
    if trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry:
                return entry["loss"]
    return None

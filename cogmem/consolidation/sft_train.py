"""SFT DoRA training on high-Q episodes (Step A of consolidation).

Teaches the model WHAT the correct patterns look like by supervised
fine-tuning on proven-successful solutions. Uses DoRA (Weight-Decomposed
Low-Rank Adaptation) which outperforms standard LoRA by 22-37%.

Uses 8-bit quantization by default because DoRA is incompatible with
4-bit Linear4bit layers in peft <= 0.13.

Runs on A4000 16GB in ~15-20 minutes for typical episode counts.
"""

import gc
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def prepare_sft_dataset(
    high_episodes: list[dict],
    tokenizer,
    config,
    benchmark: str = "bigcodebench",
) -> Dataset:
    """Convert high-Q episodes to tokenized SFT training data.

    Q-value weighting via duplication: higher Q -> more copies.
    Q=1.0 -> 3 copies, Q=0.7 -> 2 copies.
    """
    pairs = []
    for ep in high_episodes:
        if not ep.get("script"):
            continue

        instruction = ep.get("task_description", "")
        response = ep["script"]

        # Q-weighted duplication
        q = max(ep.get("q_value", 0.0), 0.01)
        copies = max(1, round(q * 3))

        for _ in range(copies):
            pairs.append({
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ]
            })

    print(f"  SFT dataset: {len(pairs)} examples "
          f"(from {len(high_episodes)} episodes, Q-weighted)")

    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tok = tokenizer(
            text,
            truncation=True,
            max_length=config.sft_max_seq_length,
            padding=False,
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    dataset = Dataset.from_list(pairs)
    dataset = dataset.map(tokenize, remove_columns=["messages"])
    return dataset


def _make_bnb_config(config) -> BitsAndBytesConfig:
    """Build quantization config. 8-bit default (DoRA compatible)."""
    if config.quantization_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def train_sft_dora(
    high_episodes: list[dict],
    config,
    benchmark: str = "bigcodebench",
    output_name: str = "sft_dora",
) -> tuple[str, float | None]:
    """Train DoRA adapter via SFT on high-Q episodes.

    Args:
        high_episodes: Episodes with Q >= q_high_threshold.
        config: ConsolidationConfig.
        benchmark: "bigcodebench" or "alfworld".
        output_name: Subdirectory name under adapters_dir.

    Returns:
        (adapter_path, final_loss)
    """
    output_dir = str(Path(config.adapters_dir) / output_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bnb_config = _make_bnb_config(config)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.sft_lora_rank,
        lora_alpha=config.sft_lora_alpha,
        lora_dropout=config.sft_lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=config.sft_use_dora,
    )

    gc.collect()
    torch.cuda.empty_cache()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = prepare_sft_dataset(high_episodes, tokenizer, config, benchmark)
    print(f"  Tokenized: {len(dataset)} examples")

    gc.collect()
    torch.cuda.empty_cache()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.sft_epochs,
            per_device_train_batch_size=config.sft_batch_size,
            gradient_accumulation_steps=config.sft_gradient_accumulation,
            learning_rate=config.sft_lr,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="none",
            seed=config.seed,
            gradient_checkpointing=True,
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    print(f"  Training SFT DoRA adapter...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Extract final loss
    final_loss = None
    if trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry:
                final_loss = entry["loss"]
                break

    print(f"  SFT DoRA saved to {output_dir}, final loss: {final_loss}")

    # Save training log
    with open(f"{output_dir}/training_log.json", "w") as f:
        json.dump({
            "method": "sft_dora",
            "base_model": config.base_model,
            "episodes": len(high_episodes),
            "dataset_size": len(dataset),
            "final_loss": final_loss,
            "config": {
                "rank": config.sft_lora_rank,
                "alpha": config.sft_lora_alpha,
                "lr": config.sft_lr,
                "epochs": config.sft_epochs,
                "use_dora": config.sft_use_dora,
                "quantization_bits": config.quantization_bits,
            },
        }, f, indent=2)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir, final_loss

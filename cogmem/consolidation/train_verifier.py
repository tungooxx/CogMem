"""Verifier DoRA training via DPO on Q-value preference pairs.

THE KEY INNOVATION: Q-values become the preference signal for DPO.
The verifier learns to permanently distinguish good code from bad code.

After training, the verifier can score ANY code for ANY task —
it has internalized what "good code" looks like. This is NOT a reward
model sitting outside the LLM — it's a DoRA adapter ON the same base model.
"""

import gc
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def train_verifier(
    preference_dataset: Dataset,
    config,
    cycle: int = 0,
) -> str:
    """Train verifier DoRA adapter via DPO.

    Args:
        preference_dataset: Dataset with "prompt", "chosen", "rejected" columns.
        config: CogMemConfig.
        cycle: Current Q-STaR cycle number.

    Returns:
        Path to saved verifier adapter.
    """
    from trl import DPOConfig, DPOTrainer

    output_dir = str(Path(config.adapters_dir) / f"verifier_v{cycle}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bits = config.quantization_bits
    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        config.active_model_hf,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.active_model_hf)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    gc.collect()
    torch.cuda.empty_cache()

    lora_config = LoraConfig(
        r=config.verifier_rank,
        lora_alpha=config.verifier_alpha,
        lora_dropout=config.verifier_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=config.use_dora,
    )

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.verifier_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=config.verifier_lr,
        beta=config.verifier_beta,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        seed=config.seed,
        report_to="none",
        max_length=2048,
        max_prompt_length=512,
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=preference_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"  Training verifier DPO ({len(preference_dataset)} pairs)...")
    trainer.train()
    trainer.save_model(output_dir)

    final_loss = None
    if trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry:
                final_loss = entry["loss"]
                break

    with open(f"{output_dir}/training_log.json", "w") as f:
        json.dump({
            "type": "verifier_dpo",
            "cycle": cycle,
            "base_model": config.active_model_hf,
            "preference_pairs": len(preference_dataset),
            "dpo_beta": config.verifier_beta,
            "final_loss": final_loss,
            "use_dora": config.use_dora,
        }, f, indent=2)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir

"""Local LoRA training using HuggingFace PEFT + QLoRA.

Runs on a single GPU (A4000 16GB or similar). No API costs.

Usage:
    from cogmem.consolidation.train_lora_local import train_lora_local
    result = train_lora_local("data/train.jsonl", config)
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


def _load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _format_chat(example: dict, tokenizer) -> dict:
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(text, truncation=True, max_length=2048, padding=False)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def train_lora_local(
    jsonl_path: str,
    config,
    policy_name: str = "q_top_k",
) -> dict:
    adapter_dir = f"{config.adapters_dir}/{policy_name}"
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)

    model_name = config.base_model

    # QLoRA: 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and process data
    raw_data = _load_jsonl(jsonl_path)
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        lambda x: _format_chat(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=adapter_dir,
        num_train_epochs=config.lora_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=config.lora_learning_rate,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    trainer.train()

    # Save adapter
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    return {
        "status": "completed",
        "adapter_dir": adapter_dir,
        "model_id": f"local-{policy_name}",
        "job_id": "local",
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
    }

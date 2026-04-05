import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class CogMemConfig:
    # Paths
    memory_bank_path: str = "results/memory_bank.json"
    adapters_dir: str = "adapters/"
    logs_dir: str = "logs/"
    replay_buffer_path: str = ""

    # Selection
    q_threshold: float = 0.7
    min_cluster_size: int = 3

    # LoRA training
    lora_provider: str = "together"
    together_api_key: str = ""
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_epochs: int = 3
    lora_learning_rate: float = 1e-5
    lora_batch_size: str = "max"

    # Verification
    verification_holdout: int = 20
    regression_threshold: float = 0.05
    eval_seeds: list[int] | None = None

    # Router
    consolidation_match_threshold: float = 0.75
    retrieval_min_q: float = 0.3

    # Evaluation
    eval_model_api_base: str = "http://localhost:11434/v1"
    eval_model: str = "qwen2.5:3b"

    # Providers
    groq_api_key: str = ""
    ollama_api_base: str = "http://localhost:11434/v1"

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.eval_seeds is None:
            self.eval_seeds = [42, 123, 456]

    @classmethod
    def from_env(cls) -> "CogMemConfig":
        cfg = cls()
        cfg.together_api_key = os.environ.get("TOGETHER_API_KEY", "")
        cfg.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "CogMemConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class ConsolidationConfig:
    """Config for the consolidation pipeline.

    Q-values weight how episodes are used during SFT/DoRA training.
    Higher Q episodes get more copies (Q-weighted duplication).
    Low Q episodes are kept as episodic memory only.
    """

    # --- Paths ---
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    adapters_dir: str = "adapters/"
    experiments_dir: str = "results/experiments/"

    # --- Q-value thresholds ---
    q_high_threshold: float = 0.7
    q_mid_low: float = 0.3

    # --- Quantization ---
    # 8-bit is default: DoRA works with Linear8bitLt.
    # 4-bit (Linear4bit) is broken with DoRA in peft <= 0.13.
    quantization_bits: int = 8

    # --- SFT DoRA training ---
    sft_lora_rank: int = 16
    sft_lora_alpha: int = 32
    sft_lora_dropout: float = 0.05
    sft_epochs: int = 3
    sft_lr: float = 1e-4
    sft_batch_size: int = 4
    sft_gradient_accumulation: int = 4
    sft_max_seq_length: int = 2048
    sft_use_dora: bool = True
    sft_early_stop_loss: float = 0.8

    # --- Verification ---
    holdout_fraction: float = 0.15
    regression_threshold: float = 0.05
    eval_seeds: list[int] | None = None

    # --- General ---
    seed: int = 42

    def __post_init__(self):
        if self.eval_seeds is None:
            self.eval_seeds = [42, 123, 456]

    @classmethod
    def from_yaml(cls, path: str) -> "ConsolidationConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

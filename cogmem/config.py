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
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
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
    eval_model: str = "llama3.2:3b"

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

import os
from dataclasses import dataclass, field, fields
from typing import List, Optional

import yaml


@dataclass
class CogMemConfig:
    """Full CogMem configuration for Q-STaR consolidation.

    Supports two models on the same architecture:
    - Coder 3B: starts ~12% on BCB-Hard, proves CogMem refines existing skill
    - General 3B: starts ~5% on BCB-Hard, proves CogMem teaches new skill
    """

    # --- Seeds ---
    seed: int = 42
    eval_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # --- Paths ---
    project_dir: str = "."
    alfworld_memory_bank: str = "results/alfworld/memory_bank.json"
    bigcodebench_memory_bank: str = "results/bigcodebench/memory_bank.json"
    adapters_dir: str = "results/adapters"
    experiments_dir: str = "results/experiments"
    logs_dir: str = "logs/"

    # --- Episode collection ---
    alfworld_episodes: int = 300
    bigcodebench_episodes: int = 148
    bigcodebench_split: str = "instruct"
    bigcodebench_max_attempts: int = 3
    bigcodebench_eval_label: str = "bigcodebench_cl"

    # --- Models ---
    # Primary: Coder 3B (proves refinement)
    coder_model_ollama: str = "qwen2.5-coder:3b"
    coder_model_hf: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    # Secondary: General 3B (proves learning from scratch)
    general_model_ollama: str = "qwen2.5:3b"
    general_model_hf: str = "Qwen/Qwen2.5-3B-Instruct"
    # Active model (set to either coder or general for each run)
    active_model_ollama: str = "qwen2.5:3b"
    active_model_hf: str = "Qwen/Qwen2.5-3B-Instruct"
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Curriculum learning (for general model) ---
    use_curriculum: bool = True  # default matches active_model=general
    bigcodebench_full_tasks: int = 1140

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434/v1"

    # --- Groq (cold baselines only) ---
    groq_api_key: str = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY", "")
    )
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # --- Q-learning ---
    q_alpha: float = 0.3
    q_gamma: float = 0.0
    q_initial: float = 0.5
    q_threshold: float = 0.7

    # --- Split lineage / contamination guards ---
    allowed_manifest_ids: List[str] = field(default_factory=list)
    blocked_manifest_ids: List[str] = field(default_factory=list)
    require_manifest_match: bool = False

    # --- Generator DoRA: Stage 1 (SFT) ---
    generator_rank: int = 16
    generator_alpha: int = 32
    generator_dropout: float = 0.05
    generator_sft_epochs: int = 10
    generator_sft_lr: float = 1e-4
    generator_batch_size: int = 4
    generator_max_seq_length: int = 2048
    quantization_bits: int = 8  # 8-bit required: DoRA incompatible with 4-bit Linear4bit
    use_dora: bool = True

    # --- Generator DoRA: Stage 2 (DPO) ---
    generator_dpo_epochs: int = 6
    generator_dpo_lr: float = 5e-5
    generator_dpo_beta: float = 0.1
    min_dpo_pairs: int = 10

    # --- Verifier DoRA (DPO) ---
    verifier_rank: int = 16
    verifier_alpha: int = 32
    verifier_dropout: float = 0.05
    verifier_epochs: int = 6
    verifier_lr: float = 5e-5
    verifier_beta: float = 0.1
    min_q_gap: float = 0.2

    # --- Early stopping (critical for small datasets) ---
    early_stopping_patience: int = 3
    eval_steps: int = 20

    # --- Verifier-guided generation ---
    num_candidates: int = 8
    verifier_temperature: float = 0.8

    # --- Iterative cycles ---
    max_cycles: int = 8
    plateau_threshold: float = 0.01

    # --- Verification ---
    holdout_fraction: float = 0.15
    min_holdout: int = 20
    regression_threshold: float = 0.05

    # --- Memory bank ---
    memory_bank_path: str = "results/bigcodebench/memory_bank_sequential.json"

    # --- Legacy compat (used by old train_lora.py, run_experiment_1) ---
    replay_buffer_path: str = ""
    lora_provider: str = "local"
    together_api_key: str = ""
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_epochs: int = 3
    lora_learning_rate: float = 1e-5
    lora_batch_size: str = "max"
    verification_holdout: int = 20

    def set_active_model(self, model_type: str) -> None:
        """Set active model to 'coder' or 'general'."""
        if model_type == "coder":
            self.active_model_ollama = self.coder_model_ollama
            self.active_model_hf = self.coder_model_hf
            self.base_model = self.coder_model_hf
            self.use_curriculum = False
        elif model_type == "general":
            self.active_model_ollama = self.general_model_ollama
            self.active_model_hf = self.general_model_hf
            self.base_model = self.general_model_hf
            self.use_curriculum = True
        else:
            raise ValueError(f"model_type must be 'coder' or 'general', got {model_type!r}")

    @classmethod
    def from_env(cls) -> "CogMemConfig":
        cfg = cls()
        cfg.together_api_key = os.environ.get("TOGETHER_API_KEY", "")
        cfg.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "CogMemConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

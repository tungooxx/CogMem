import os
from cogmem.config import CogMemConfig


def test_default_config():
    cfg = CogMemConfig()
    assert cfg.q_threshold == 0.7
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_epochs == 3
    assert cfg.base_model == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.verification_holdout == 20
    assert cfg.regression_threshold == 0.05
    # New Q-STaR fields
    assert cfg.generator_rank == 16
    assert cfg.generator_dpo_beta == 0.1
    assert cfg.verifier_rank == 16
    assert cfg.num_candidates == 8
    assert cfg.use_dora is True


def test_set_active_model():
    cfg = CogMemConfig()
    cfg.set_active_model("general")
    assert cfg.active_model_hf == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.active_model_ollama == "qwen2.5:3b"
    assert cfg.base_model == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.use_curriculum is True

    cfg.set_active_model("coder")
    assert cfg.active_model_hf == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert cfg.active_model_ollama == "qwen2.5-coder:3b"
    assert cfg.base_model == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert cfg.use_curriculum is False


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-123")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key-456")
    cfg = CogMemConfig.from_env()
    assert cfg.together_api_key == "test-key-123"
    assert cfg.groq_api_key == "groq-key-456"


def test_config_from_yaml(tmp_path):
    yaml_content = "q_threshold: 0.8\nlora_rank: 8\n"
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(yaml_content)
    cfg = CogMemConfig.from_yaml(str(yaml_path))
    assert cfg.q_threshold == 0.8
    assert cfg.lora_rank == 8
    assert cfg.lora_epochs == 3  # unchanged default

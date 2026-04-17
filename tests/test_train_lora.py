import pytest
from unittest.mock import MagicMock, patch
from cogmem.config import CogMemConfig
from cogmem.consolidation.train_lora import (
    upload_training_file,
    start_lora_job,
    wait_for_job,
    download_adapter,
    train_lora_together,
)


@pytest.fixture
def config():
    return CogMemConfig(together_api_key="test-key")


@patch("cogmem.consolidation.train_lora.Together")
def test_upload_training_file(mock_together_cls, tmp_path, config):
    mock_client = MagicMock()
    mock_together_cls.return_value = mock_client
    mock_client.files.upload.return_value = MagicMock(id="file-abc123")

    jsonl_path = str(tmp_path / "train.jsonl")
    with open(jsonl_path, "w") as f:
        f.write('{"messages": [{"role": "user", "content": "test"}]}\n')

    file_id = upload_training_file(jsonl_path, config)
    assert file_id == "file-abc123"
    mock_client.files.upload.assert_called_once()


@patch("cogmem.consolidation.train_lora.Together")
def test_start_lora_job(mock_together_cls, config):
    mock_client = MagicMock()
    mock_together_cls.return_value = mock_client
    mock_client.fine_tuning.create.return_value = MagicMock(id="ft-job-xyz")

    job_id = start_lora_job("file-abc123", config, suffix="test-run")
    assert job_id == "ft-job-xyz"
    call_kwargs = mock_client.fine_tuning.create.call_args[1]
    assert call_kwargs["training_file"] == "file-abc123"
    assert call_kwargs["lora"] is True
    assert call_kwargs["n_epochs"] == config.lora_epochs


@patch("cogmem.consolidation.train_lora.Together")
def test_wait_for_job_completed(mock_together_cls, config):
    mock_client = MagicMock()
    mock_together_cls.return_value = mock_client
    mock_client.fine_tuning.retrieve.return_value = MagicMock(
        status="completed", output_name="user/model-ft-xyz"
    )

    result = wait_for_job("ft-job-xyz", config, poll_interval=0.01)
    assert result["status"] == "completed"
    assert result["model_id"] == "user/model-ft-xyz"


@patch("cogmem.consolidation.train_lora.Together")
def test_wait_for_job_failed(mock_together_cls, config):
    mock_client = MagicMock()
    mock_together_cls.return_value = mock_client
    mock_client.fine_tuning.retrieve.return_value = MagicMock(
        status="failed", output_name=None
    )

    with pytest.raises(RuntimeError, match="failed"):
        wait_for_job("ft-job-xyz", config, poll_interval=0.01)


@patch("cogmem.consolidation.train_lora.register_adapter_artifact")
@patch("cogmem.consolidation.train_lora.download_adapter")
@patch("cogmem.consolidation.train_lora.wait_for_job")
@patch("cogmem.consolidation.train_lora.start_lora_job")
@patch("cogmem.consolidation.train_lora.upload_training_file")
def test_train_lora_together_registers_adapter_metadata(
    mock_upload,
    mock_start,
    mock_wait,
    mock_download,
    mock_register,
    config,
):
    mock_upload.return_value = "file-abc123"
    mock_start.return_value = "ft-job-xyz"
    mock_wait.return_value = {"status": "completed", "model_id": "user/model-ft-xyz", "job_id": "ft-job-xyz"}
    mock_register.return_value = type("Record", (), {"adapter_id": "adapter_123"})()

    result = train_lora_together(
        "train.jsonl",
        config,
        policy_name="q_top_k",
        source_skill_card_ids=["skill_a"],
        training_manifest_ids=["manifest_a"],
        compatible_families=["file_io"],
        adapter_role="family",
        dev_gain=0.12,
        registry_path="results/adapters/registry.json",
    )

    assert result["adapter_id"] == "adapter_123"
    kwargs = mock_register.call_args.kwargs
    assert kwargs["training_manifest_ids"] == ["manifest_a"]
    assert kwargs["source_skill_card_ids"] == ["skill_a"]
    assert kwargs["compatible_families"] == ["file_io"]

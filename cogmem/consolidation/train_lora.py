import time

from together import Together

from cogmem.consolidation.adapter_registry import register_adapter_artifact


def _client(config) -> Together:
    return Together(api_key=config.together_api_key)


def upload_training_file(jsonl_path: str, config) -> str:
    client = _client(config)
    resp = client.files.upload(file=jsonl_path)
    return resp.id


def start_lora_job(file_id: str, config, suffix: str = "cogmem") -> str:
    client = _client(config)
    resp = client.fine_tuning.create(
        training_file=file_id,
        model=config.base_model,
        n_epochs=config.lora_epochs,
        learning_rate=config.lora_learning_rate,
        batch_size=config.lora_batch_size,
        lora=True,
        lora_r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        suffix=suffix,
        n_checkpoints=1,
    )
    return resp.id


def wait_for_job(
    job_id: str, config, poll_interval: float = 30.0, max_wait: float = 7200.0
) -> dict:
    client = _client(config)
    elapsed = 0.0
    while elapsed < max_wait:
        status = client.fine_tuning.retrieve(id=job_id)
        if status.status == "completed":
            return {"status": "completed", "model_id": status.output_name, "job_id": job_id}
        if status.status == "failed":
            raise RuntimeError(f"LoRA training job {job_id} failed")
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"LoRA training job {job_id} timed out after {max_wait}s")


def download_adapter(job_id: str, output_dir: str, config) -> str:
    client = _client(config)
    client.fine_tuning.download(id=job_id, output=output_dir, checkpoint_type="adapter")
    return output_dir


def train_lora_together(
    jsonl_path: str,
    config,
    policy_name: str = "q_top_k",
    *,
    source_skill_card_ids: list[str] | None = None,
    training_manifest_ids: list[str] | None = None,
    compatible_families: list[str] | None = None,
    adapter_role: str = "family",
    dev_gain: float = 0.0,
    registry_path: str | None = None,
) -> dict:
    file_id = upload_training_file(jsonl_path, config)
    suffix = f"cogmem-{policy_name}"
    job_id = start_lora_job(file_id, config, suffix=suffix)
    result = wait_for_job(job_id, config)

    adapter_dir = f"{config.adapters_dir}/{policy_name}"
    download_adapter(job_id, adapter_dir, config)
    result["adapter_dir"] = adapter_dir
    if registry_path:
        record = register_adapter_artifact(
            registry_path,
            adapter_dir,
            base_model=config.base_model,
            adapter_role=adapter_role,
            training_manifest_ids=training_manifest_ids or [],
            source_skill_card_ids=source_skill_card_ids or [],
            compatible_families=compatible_families or [],
            dev_gain=dev_gain,
            metadata={
                "trainer": "together",
                "policy_name": policy_name,
                "job_id": job_id,
                "jsonl_path": jsonl_path,
            },
        )
        result["adapter_id"] = record.adapter_id

    return result

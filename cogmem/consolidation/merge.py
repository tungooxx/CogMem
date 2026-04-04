"""Merge SFT and GRPO DoRA adapters into a single adapter (Step C).

After training SFT (memorization) and GRPO (reasoning) adapters separately,
merge them so the final model has both capabilities.

Two strategies:
1. weighted_average: Linear interpolation of adapter weights.
   Simple, works when adapters share the same architecture.
2. sequential: Load both adapters via PEFT multi-adapter, then merge.
   More complex but preserves both capabilities better.
"""

import gc
import json
import os
import shutil
from pathlib import Path

import torch


def merge_adapters(
    sft_adapter_path: str,
    grpo_adapter_path: str,
    config,
    output_name: str = "merged_dora",
) -> str:
    """Merge two adapter state dicts via weighted average.

    merged_weight = sft_weight * sft + grpo_weight * grpo

    Args:
        sft_adapter_path: Path to SFT DoRA adapter directory.
        grpo_adapter_path: Path to GRPO DoRA adapter directory.
        config: ConsolidationConfig with sft_weight and grpo_weight.
        output_name: Subdirectory name for merged adapter.

    Returns:
        Path to merged adapter directory.
    """
    output_dir = str(Path(config.adapters_dir) / output_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("  Merging adapters:")
    print(f"    SFT:  {sft_adapter_path} (weight: {config.sft_weight})")
    print(f"    GRPO: {grpo_adapter_path} (weight: {config.grpo_weight})")

    # Load adapter weights — try safetensors first, fall back to bin
    sft_state = _load_adapter_state(sft_adapter_path)
    grpo_state = _load_adapter_state(grpo_adapter_path)

    # Weighted average of matching keys
    merged_state = {}
    all_keys = set(sft_state.keys()) | set(grpo_state.keys())

    merged_count = 0
    sft_only = 0
    grpo_only = 0

    for key in all_keys:
        if key in sft_state and key in grpo_state:
            merged_state[key] = (
                config.sft_weight * sft_state[key]
                + config.grpo_weight * grpo_state[key]
            )
            merged_count += 1
        elif key in sft_state:
            merged_state[key] = sft_state[key]
            sft_only += 1
        else:
            merged_state[key] = grpo_state[key]
            grpo_only += 1

    print(
        f"  Merged: {merged_count} shared, "
        f"{sft_only} SFT-only, {grpo_only} GRPO-only"
    )

    # Copy adapter config from SFT (same architecture)
    for fname in os.listdir(sft_adapter_path):
        if fname in (
            "adapter_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
        ):
            shutil.copy(
                os.path.join(sft_adapter_path, fname),
                os.path.join(output_dir, fname),
            )

    # Save merged weights
    _save_adapter_state(merged_state, output_dir)

    # Save merge log
    with open(os.path.join(output_dir, "merge_log.json"), "w") as f:
        json.dump({
            "sft_adapter": sft_adapter_path,
            "grpo_adapter": grpo_adapter_path,
            "sft_weight": config.sft_weight,
            "grpo_weight": config.grpo_weight,
            "method": config.merge_method,
            "merged_params": merged_count,
            "sft_only_params": sft_only,
            "grpo_only_params": grpo_only,
        }, f, indent=2)

    print(f"  Merged adapter saved to {output_dir}")
    return output_dir


def merge_sequential(
    sft_adapter_path: str,
    grpo_adapter_path: str,
    config,
    output_name: str = "merged_sequential",
) -> str:
    """Load both adapters via PEFT, merge into base weights.

    Alternative to weighted_average. Loads SFT adapter first,
    then GRPO adapter as a second named adapter. PEFT applies both.
    Then merge_and_unload() folds everything into base weights.

    Returns path to merged full model (not adapter — full weights).
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    output_dir = str(Path(config.adapters_dir) / output_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )

    # Load SFT adapter
    model = PeftModel.from_pretrained(base, sft_adapter_path, adapter_name="sft")

    # Load GRPO adapter
    model.load_adapter(grpo_adapter_path, adapter_name="grpo")

    # Activate both adapters
    model.set_adapter(["sft", "grpo"])

    # Merge into base weights
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)

    print(f"  Sequential merge saved to {output_dir}")

    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


def _load_adapter_state(adapter_path: str) -> dict[str, torch.Tensor]:
    """Load adapter weights from safetensors or bin format."""
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        return load_file(safetensors_path, device="cpu")

    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(
        f"No adapter weights found in {adapter_path}. "
        f"Expected adapter_model.safetensors or adapter_model.bin"
    )


def _save_adapter_state(state: dict[str, torch.Tensor], output_dir: str) -> None:
    """Save adapter weights. Prefer safetensors if available."""
    try:
        from safetensors.torch import save_file
        save_file(state, os.path.join(output_dir, "adapter_model.safetensors"))
    except ImportError:
        torch.save(state, os.path.join(output_dir, "adapter_model.bin"))

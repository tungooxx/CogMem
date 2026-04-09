"""Dynamic patch composition — apply micro-LoRA patches to base model.

At inference, compose selected patches onto the model:
  patched_weight = base_weight + sum(B @ A * q_value for each patch)

This is additive and reversible — the base model is never modified permanently.
Composition of 5-10 rank-2 patches adds < 50MB memory on top of the base model.
"""

import torch
import torch.nn as nn


def compose_patches(
    model: nn.Module,
    patches: list,
    scaling_factor: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the combined LoRA delta from all active patches.

    Returns a dict of {param_name: delta_tensor} that can be applied
    and reversed on the model's weights.

    Each patch contributes: B @ A * q_value * scaling_factor
    Patches stack linearly (additive composition).
    """
    deltas = {}

    for patch in patches:
        if not patch.lora_weights:
            continue

        q_scale = patch.q_value * scaling_factor

        for layer_key, ab in patch.lora_weights.items():
            a = ab.get("A")  # shape: (rank, hidden_dim) or (rank, in_features)
            b = ab.get("B")  # shape: (hidden_dim, rank) or (out_features, rank)

            if a is None or b is None:
                continue

            # Compute LoRA delta: B @ A * scale
            delta = (b.float() @ a.float()) * q_scale

            if layer_key in deltas:
                deltas[layer_key] = deltas[layer_key] + delta
            else:
                deltas[layer_key] = delta

    return deltas


def apply_deltas(model: nn.Module, deltas: dict[str, torch.Tensor]) -> None:
    """Add computed deltas to model weights (in-place).

    Call this before generation, then remove_deltas after.
    """
    state = model.state_dict()
    for name, delta in deltas.items():
        if name in state:
            device = state[name].device
            dtype = state[name].dtype
            state[name].add_(delta.to(device=device, dtype=dtype))


def remove_deltas(model: nn.Module, deltas: dict[str, torch.Tensor]) -> None:
    """Remove deltas from model weights (reverse of apply_deltas)."""
    state = model.state_dict()
    for name, delta in deltas.items():
        if name in state:
            device = state[name].device
            dtype = state[name].dtype
            state[name].sub_(delta.to(device=device, dtype=dtype))


class PatchedModel:
    """Context manager for temporarily applying patches to a model.

    Usage:
        with PatchedModel(model, patches) as patched:
            output = patched.generate(...)

    Automatically applies patches on enter, removes on exit.
    """

    def __init__(self, model: nn.Module, patches: list, scaling_factor: float = 1.0):
        self.model = model
        self.patches = patches
        self.scaling_factor = scaling_factor
        self.deltas = None

    def __enter__(self):
        self.deltas = compose_patches(self.model, self.patches, self.scaling_factor)
        apply_deltas(self.model, self.deltas)
        return self.model

    def __exit__(self, *args):
        if self.deltas:
            remove_deltas(self.model, self.deltas)
            self.deltas = None

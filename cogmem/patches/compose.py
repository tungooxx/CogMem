"""Dynamic patch composition — apply micro-LoRA patches to base model.

For quantized (4-bit) models, we can't modify weights directly.
Instead, we use forward hooks: intercept each linear layer's output
and add the LoRA delta: output = original_output + input @ A.T @ B.T * scale

This is mathematically equivalent to W' = W + B @ A, but applied
at the activation level rather than the weight level.
"""

import torch
import torch.nn as nn

from cogmem.patches.patch import CognitivePatch


def compose_patches(
    model: nn.Module,
    patches: list[CognitivePatch],
    scaling_factor: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the combined LoRA delta from all active patches.

    Returns a dict of {param_name: delta_tensor}.
    Each patch contributes: B @ A * q_value * scaling_factor
    """
    deltas = {}

    for patch in patches:
        if not patch.lora_weights:
            continue

        q_scale = patch.q_value * scaling_factor

        for layer_key, ab in patch.lora_weights.items():
            a = ab.get("A")
            b = ab.get("B")
            if a is None or b is None:
                continue

            delta = (b.float() @ a.float()) * q_scale

            if layer_key in deltas:
                deltas[layer_key] = deltas[layer_key] + delta
            else:
                deltas[layer_key] = delta

    return deltas


def _find_module(model: nn.Module, name: str) -> nn.Module | None:
    """Find a module by its parameter name (strip '.weight' suffix)."""
    # name like: model.layers.0.self_attn.q_proj.weight
    # module path: model.layers.0.self_attn.q_proj
    parts = name.replace(".weight", "").replace(".bias", "").split(".")
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        elif part.isdigit() and hasattr(module, "__getitem__"):
            module = module[int(part)]
        else:
            return None
    return module


class PatchedModel:
    """Context manager that applies patches via forward hooks.

    For 4-bit models, we can't modify quantized weights. Instead,
    we register hooks on each attention projection layer that add
    the LoRA delta to the layer's output:

        hooked_output = original_output + input @ A.T @ B.T * scale

    This is mathematically equivalent to:
        (W + B @ A) @ input.T = W @ input.T + (B @ A) @ input.T

    Hooks are automatically removed when the context manager exits.
    """

    def __init__(self, model: nn.Module, patches: list[CognitivePatch],
                 scaling_factor: float = 1.0):
        self.model = model
        self.patches = patches
        self.scaling_factor = scaling_factor
        self._hooks = []

    def __enter__(self):
        # Build per-module delta lookup
        # Group LoRA A,B by module path (without .weight)
        module_deltas = {}  # module_path -> {"A": tensor, "B": tensor}

        for patch in self.patches:
            if not patch.lora_weights:
                continue

            q_scale = patch.q_value * self.scaling_factor

            for layer_key, ab in patch.lora_weights.items():
                a = ab.get("A")
                b = ab.get("B")
                if a is None or b is None:
                    continue

                # module path = layer_key without .weight
                mod_path = layer_key.replace(".weight", "")

                if mod_path not in module_deltas:
                    module_deltas[mod_path] = {"A": a.float() * q_scale, "B": b.float()}
                else:
                    # Stack: add scaled A, keep B (they share the same B if same layer)
                    # Actually for multiple patches, we need full delta
                    existing = module_deltas[mod_path]
                    # Combine: delta = B1 @ A1 + B2 @ A2
                    # Store as pre-computed delta instead
                    if "delta" not in existing:
                        existing["delta"] = existing["B"] @ existing["A"]
                    existing["delta"] = existing["delta"] + (b.float() @ (a.float() * q_scale))

        # Register forward hooks
        for mod_path, weights in module_deltas.items():
            module = _find_module(self.model, mod_path)
            if module is None:
                continue

            # Pre-compute delta if not already
            if "delta" in weights:
                delta = weights["delta"]
            else:
                delta = weights["B"] @ weights["A"]

            # Capture delta in closure
            def make_hook(d):
                def hook(module, input, output):
                    # output shape: (batch, seq_len, out_features)
                    # delta shape: (out_features, in_features)
                    # input[0] shape: (batch, seq_len, in_features)
                    if isinstance(input, tuple) and len(input) > 0:
                        x = input[0]
                        # LoRA: output += x @ delta.T
                        device = output.device
                        dtype = output.dtype
                        lora_out = x.float() @ d.to(device).T
                        return output + lora_out.to(dtype)
                    return output
                return hook

            handle = module.register_forward_hook(make_hook(delta))
            self._hooks.append(handle)

        return self.model

    def __exit__(self, *args):
        # Remove all hooks
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

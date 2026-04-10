"""Dynamic patch composition via forward hooks for quantized models.

4-bit quantized weights can't be modified directly. Instead we register
forward hooks that add LoRA deltas at the activation level:

    hooked_output = original_output + input @ delta.T

Deltas are precomputed once (sum all patches per layer), not per forward pass.
"""

import torch
import torch.nn as nn

from cogmem.patches.patch import CognitivePatch


class PatchedModel:
    """Context manager that applies patches via forward hooks.

    Usage:
        with PatchedModel(model, patches) as patched:
            output = patched.generate(...)

    Hooks auto-removed on exit.
    """

    def __init__(self, model: nn.Module, patches: list[CognitivePatch],
                 scaling_factor: float = 1.0):
        self.model = model
        self.patches = patches
        self.scaling_factor = scaling_factor
        self._hooks = []

    def __enter__(self):
        self._hooks = compose_patches(self.model, self.patches, self.scaling_factor)
        return self.model

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def compose_patches(
    model: nn.Module,
    patches: list[CognitivePatch],
    scaling_factor: float = 1.0,
) -> list:
    """Register forward hooks that apply LoRA deltas per layer.

    Precomputes combined delta per layer (sum of all patches),
    then registers one hook per layer. Returns list of hook handles
    that the caller must remove after generation.

    Args:
        model: Base model (can be quantized).
        patches: Active patches with loaded lora_weights.
        scaling_factor: Global scaling for all patches.

    Returns:
        List of hook handles. Remove them after generation.
    """
    hooks = []

    # Find model layers
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h

    if layers is None:
        print("Warning: unrecognized model architecture — no layers found for patch composition")
        return hooks

    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj"]

    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "self_attn"):
            continue

        for proj_name in proj_names:
            module = getattr(layer.self_attn, proj_name, None)
            if module is None:
                continue

            # Sum all patch deltas for this layer+projection
            combined_delta = None

            for patch in patches:
                if not patch.lora_weights:
                    continue

                # Try multiple key formats
                possible_keys = [
                    f"model.layers.{layer_idx}.self_attn.{proj_name}.weight",
                    f"layers.{layer_idx}.self_attn.{proj_name}.weight",
                    f"model.model.layers.{layer_idx}.self_attn.{proj_name}.weight",
                ]

                matched_key = None
                for key in possible_keys:
                    if key in patch.lora_weights:
                        matched_key = key
                        break

                if matched_key is None:
                    continue

                ab = patch.lora_weights[matched_key]
                a = ab.get("A")
                b = ab.get("B")
                if a is None or b is None:
                    continue

                device = module.weight.device
                delta = (b.float().to(device) @ a.float().to(device)) * patch.q_value * scaling_factor

                if combined_delta is None:
                    combined_delta = delta
                else:
                    combined_delta = combined_delta + delta

            if combined_delta is not None:
                def make_hook(d):
                    def hook(module, inp, output):
                        x = inp[0].float()
                        lora_out = x @ d.T
                        return output + lora_out.to(output.dtype)
                    return hook

                h = module.register_forward_hook(make_hook(combined_delta))
                hooks.append(h)

    return hooks

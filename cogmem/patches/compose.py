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

                # Shape check: B @ A must produce (out_features, in_features)
                if b.dim() != 2 or a.dim() != 2 or b.shape[1] != a.shape[0]:
                    print(f"Warning: shape mismatch for {patch.patch_id} at {matched_key}: "
                          f"B={tuple(b.shape)} A={tuple(a.shape)}, skipping")
                    continue

                device = module.weight.device
                
                # Apply proper LoRA scaling: lora_alpha / rank
                # In create.py, lora_alpha = rank * 2, so scaling is always 2.0
                # We also apply the patch's q_value and the global scaling_factor
                lora_scale = (patch.rank * 2) / patch.rank if hasattr(patch, 'rank') and patch.rank else 2.0
                
                delta = (b.float().to(device) @ a.float().to(device)) * lora_scale * patch.q_value * scaling_factor

                # Validate against the logical projection shape, not packed quantized storage.
                expected_shape = _get_logical_projection_shape(module)
                if expected_shape is not None and delta.shape != expected_shape:
                    print(f"Warning: delta shape {tuple(delta.shape)} != expected shape {tuple(expected_shape)} "
                          f"for {patch.patch_id} at {proj_name}, skipping")
                    continue

                if combined_delta is None:
                    combined_delta = delta
                else:
                    combined_delta = combined_delta + delta

            if combined_delta is not None:
                def make_hook(d):
                    def hook(module, inp, output):
                        x = inp[0].float()
                        if x.shape[-1] != d.shape[1] or output.shape[-1] != d.shape[0]:
                            return output
                        lora_out = x @ d.T
                        return output + lora_out.to(output.dtype)
                    return hook

                try:
                    h = module.register_forward_hook(make_hook(combined_delta))
                    hooks.append(h)
                except Exception:
                    # Rollback all hooks on failure
                    for handle in hooks:
                        handle.remove()
                    hooks.clear()
                    raise

    return hooks


def _get_logical_projection_shape(module: nn.Module) -> tuple[int, int] | None:
    """Return the logical (out_features, in_features) shape for a projection module."""
    out_features = getattr(module, "out_features", None)
    in_features = getattr(module, "in_features", None)
    if out_features is not None and in_features is not None:
        return int(out_features), int(in_features)

    weight = getattr(module, "weight", None)
    if weight is not None and getattr(weight, "dim", lambda: 0)() == 2:
        return tuple(int(v) for v in weight.shape)

    return None

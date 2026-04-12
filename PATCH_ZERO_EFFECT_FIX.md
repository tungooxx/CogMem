# CogMem Patch Zero-Effect Bug — Root Cause Analysis & Fix

## Summary

The diagnostic showed that 144 hooks were correctly registered but patches had **zero effect** on model output at both greedy (temp=0) and low-temperature (0.01) sampling. This document explains the root cause and all fixes applied.

---

## Root Cause Analysis

The zero-effect bug has **two compounding causes**:

### Cause 1: Missing LoRA Scaling Factor (Primary)

In `cogmem/patches/compose.py`, the delta was computed as:

```python
# BEFORE (broken)
delta = (b.float().to(device) @ a.float().to(device)) * patch.q_value * scaling_factor
```

In standard LoRA (and PEFT), the forward pass applies a scaling factor of `lora_alpha / rank` at inference time. This scaling is **not baked into the saved A and B matrices** — it is applied dynamically. The original code omitted this scaling entirely and used `q_value = 0.5` instead, which is **4× weaker** than the correct `lora_alpha / rank = 4 / 2 = 2.0`.

| Factor | Value | Effect |
|---|---|---|
| Correct LoRA scaling (`lora_alpha / rank`) | 2.0 | Standard PEFT behavior |
| Applied scaling (`q_value`) | 0.5 | 4× too weak |
| Resulting delta magnitude | ~0.0005 max | Insufficient to flip argmax |

### Cause 2: Insufficient Training Steps (Secondary)

In `cogmem/patches/create.py`, patches were created with:

```python
# BEFORE (too few steps)
n_steps: int = 10,
lr: float = 1e-3,
```

PEFT initializes the LoRA `B` matrix to **zeros**. With only 10 training steps at `lr=1e-3`, the `B` matrix barely moves from zero, producing a near-zero delta even after applying the correct scaling. The effect on hidden states was estimated at `max ≈ 0.0007`, far below the logit margin of 2–10 needed to flip the greedy argmax.

---

## Why 144 Hooks Were Registered But Had Zero Effect

This is the key diagnostic puzzle. The answer is:

1. **Key matching succeeded** — the saved weight keys (`model.layers.N.self_attn.q_proj.weight`) correctly matched the `possible_keys` list in `compose_patches()`.
2. **Shape check passed** — `delta.shape == module.weight.shape` because `B @ A` produces `(out_features, in_features)` which matches the quantized weight shape.
3. **`combined_delta` was non-None** — even a near-zero delta is not `None`, so the hook was registered.
4. **The hook fired but added near-zero values** — the hook correctly computed `output + x @ delta.T`, but `delta` was so small that the addition had no measurable effect on the argmax.

---

## Fixes Applied

### Fix 1: Correct LoRA Scaling in `compose.py`

```python
# AFTER (fixed)
lora_scale = (patch.rank * 2) / patch.rank  # = 2.0 (lora_alpha / rank)
delta = (b.float().to(device) @ a.float().to(device)) * lora_scale * patch.q_value * scaling_factor
```

This applies the same scaling that PEFT uses during its own forward pass, ensuring the patch delta has the correct magnitude.

### Fix 2: More Training Steps in `create.py`

```python
# AFTER (fixed) — create_patch_from_contrast
n_steps: int = 30,   # was 10
lr: float = 2e-3,    # was 1e-3

# AFTER (fixed) — create_patch_from_example
n_steps: int = 20,   # was 5
lr: float = 2e-3,    # was 1e-3
```

More steps and a higher learning rate ensure the `B` matrix accumulates meaningful values during micro-finetuning.

---

## What You Need to Do

### For Existing Patches (Already Created)

Your existing 23–30 patches were created with the old settings (10 steps, `lr=1e-3`). Their `B` matrices may be near-zero. You have two options:

**Option A: Re-create patches** (recommended for best results)
- Delete the existing patch bank directory (`/notebooks/cogmem_patches`)
- Re-run Cell 4 (patch creation) — it will now use 30 steps and `lr=2e-3`
- New patches will have stronger `B` matrices

**Option B: Use higher `scaling_factor`** (quick test with existing patches)
- Use `PatchedModel(base_model, active, scaling_factor=5.0)` to amplify weak patches
- This is a workaround, not a permanent fix

### Diagnostic Cell (Run First)

Add `debug_patch_cell.py` as a new cell in your notebook to verify:
1. `||B||` norms for your patches (should be `> 1e-3`)
2. Key format is correct
3. Hook fires and modifies output
4. Which `scaling_factor` first produces a different output

---

## Architecture Note

The overall approach (forward hooks on 4-bit quantized attention projections) is **architecturally correct**. The mechanism works — hooks register, fire, and modify outputs. The problem was purely in the numerical magnitude of the applied delta. With the fixes applied, patches should produce measurable behavioral changes.

The two-cause nature of this bug (missing LoRA scaling + insufficient training) meant that even with correct hooks, the effect was invisible to greedy decoding.

---

## Commit

```
84b35eb fix: apply proper LoRA scaling and increase training steps for cognitive patches
```

Files changed:
- `cogmem/patches/compose.py` — apply `lora_alpha/rank` scaling
- `cogmem/patches/create.py` — increase `n_steps` and `lr`

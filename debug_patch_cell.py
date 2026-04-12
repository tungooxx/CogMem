"""
Cell 4d: Deep diagnostic for patch composition (add this to your notebook)

Run this BEFORE Cell 4c to understand what's happening with your patches.
"""

import torch
import numpy as np
from cogmem.patches.compose import PatchedModel

print("=" * 60)
print("PATCH COMPOSITION DEEP DIAGNOSTIC")
print("=" * 60)

# ── Step 1: Check B matrix norms ──────────────────────────────────────────
print("\n[1] B matrix norms (should be > 1e-3 for patches to have effect):")
for patch in patch_bank.patches[:5]:
    patch_bank.load_weights(patch)
    b_norms = []
    a_norms = []
    for layer_key, ab in patch.lora_weights.items():
        if 'q_proj' in layer_key:  # just check q_proj for brevity
            b_norms.append(torch.norm(ab['B']).item())
            a_norms.append(torch.norm(ab['A']).item())
    avg_b = np.mean(b_norms) if b_norms else 0
    avg_a = np.mean(a_norms) if a_norms else 0
    print(f"  {patch.patch_id[:40]}: ||B||={avg_b:.5f}, ||A||={avg_a:.5f}, q={patch.q_value:.2f}")
    patch.unload_weights()

# ── Step 2: Check key format ───────────────────────────────────────────────
print("\n[2] Key format in saved patches (first patch):")
first_patch = patch_bank.patches[0]
patch_bank.load_weights(first_patch)
keys = list(first_patch.lora_weights.keys())
print(f"  Total keys: {len(keys)}")
print(f"  First 3 keys: {keys[:3]}")
print(f"  Expected format: 'model.layers.N.self_attn.q_proj.weight'")
first_patch.unload_weights()

# ── Step 3: Verify delta magnitude ────────────────────────────────────────
print("\n[3] Delta magnitude analysis (with fix applied):")
first_patch = patch_bank.patches[0]
patch_bank.load_weights(first_patch)
key = keys[0]
ab = first_patch.lora_weights[key]
A, B = ab['A'].float(), ab['B'].float()
rank = first_patch.rank
lora_scale = (rank * 2) / rank  # = 2.0
delta = (B @ A) * lora_scale * first_patch.q_value
print(f"  rank={rank}, lora_alpha={rank*2}, lora_scale={lora_scale}")
print(f"  ||B||={torch.norm(B):.5f}, ||A||={torch.norm(A):.5f}")
print(f"  ||delta (B@A)||={torch.norm(B@A):.5f}")
print(f"  ||delta * lora_scale * q_value||={torch.norm(delta):.5f}")
print(f"  max(|delta|)={torch.max(torch.abs(delta)):.5f}")
if torch.norm(B) < 1e-4:
    print("  ⚠️  WARNING: B is near-zero! Patches were not trained enough.")
    print("  ⚠️  Re-run patch creation with n_steps=30, lr=2e-3")
else:
    print("  ✓ B is non-zero. Patches have signal.")
first_patch.unload_weights()

# ── Step 4: Verify hook actually fires and modifies output ─────────────────
print("\n[4] Hook fire verification (synthetic test):")
# Create a tiny test to verify the hook modifies the output
test_module = torch.nn.Linear(8, 8, bias=False)
test_input = torch.randn(1, 4, 8)
original_out = test_module(test_input)

test_delta = torch.eye(8) * 0.1  # 10% identity perturbation
def test_hook(module, inp, output):
    x = inp[0].float()
    lora_out = x @ test_delta.T
    return output + lora_out.to(output.dtype)

handle = test_module.register_forward_hook(test_hook)
hooked_out = test_module(test_input)
handle.remove()

diff = (hooked_out - original_out).abs().max().item()
print(f"  Hook modification test: max diff = {diff:.4f}")
print(f"  Hook fires correctly: {diff > 1e-6}")

# ── Step 5: Full end-to-end test with real model ───────────────────────────
print("\n[5] End-to-end test with scaling_factor=5.0:")
task = EVAL_TASKS[0]
prompt = task.get('instruct_prompt', task.get('complete_prompt', ''))
messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]

active = list(patch_bank.patches)[:5]  # use first 5 patches
for p in active:
    patch_bank.load_weights(p)

torch.manual_seed(42)
out_cold = generate_with_model(base_model, tokenizer, messages, temperature=0)

# Try with increasing scaling factors
for sf in [1.0, 2.0, 5.0, 10.0]:
    torch.manual_seed(42)
    with PatchedModel(base_model, active, scaling_factor=sf):
        out_patched = generate_with_model(base_model, tokenizer, messages, temperature=0)
    same = out_cold == out_patched
    print(f"  scaling_factor={sf}: {'IDENTICAL ❌' if same else 'DIFFERENT ✓'}")
    if not same:
        break

for p in active:
    p.unload_weights()

print("\n" + "=" * 60)
print("SUMMARY:")
print("  If B norms are near-zero: re-create patches with more steps")
print("  If B norms are OK but output identical: increase scaling_factor")
print("  If scaling_factor=10 still identical: check key format mismatch")
print("=" * 60)

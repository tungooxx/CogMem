"""Sleep mode — consolidate cognitive patches.

1. MERGE: combine similar high-Q patches into stronger patches
2. PRUNE: remove low-Q patches that never help
3. PROMOTE: very high-Q patches become permanent (always active)
4. GROW: increase rank of merged patches for more capacity
"""

import numpy as np
import torch

from cogmem.patches.bank import PatchBank
from cogmem.patches.patch import CognitivePatch


def run_sleep_cycle(patch_bank: PatchBank, config: dict | None = None) -> dict:
    """Consolidate patches during sleep phase.

    Args:
        patch_bank: Bank to consolidate.
        config: Optional config overrides.

    Returns:
        Stats about what changed.
    """
    cfg = {
        "merge_similarity": 0.8,
        "merge_min_q": 0.6,
        "prune_max_q": 0.2,
        "prune_min_visits": 5,
        "promote_min_q": 0.9,
        "promote_min_visits": 20,
    }
    if config:
        cfg.update(config)

    stats = {
        "before": len(patch_bank.patches),
        "merged": 0,
        "pruned": 0,
        "promoted": [],
    }

    # 1. MERGE similar high-Q patches
    stats["merged"] = _merge_similar(patch_bank, cfg)

    # 2. PRUNE low-Q patches
    stats["pruned"] = _prune_bad(patch_bank, cfg)

    # 3. PROMOTE star patches
    stats["promoted"] = _find_promotable(patch_bank, cfg)

    stats["after"] = len(patch_bank.patches)

    print(f"Sleep cycle: {stats['before']} → {stats['after']} patches")
    print(f"  Merged: {stats['merged']}")
    print(f"  Pruned: {stats['pruned']}")
    print(f"  Promotable: {len(stats['promoted'])}")

    patch_bank.save()
    return stats


def _merge_similar(bank: PatchBank, cfg: dict) -> int:
    """Find similar high-Q patches and merge them."""
    if len(bank.patches) < 2:
        return 0

    embs = np.array([p.embedding for p in bank.patches])
    merged_count = 0
    merged_ids = set()

    for i in range(len(bank.patches)):
        if bank.patches[i].patch_id in merged_ids:
            continue
        if bank.patches[i].q_value < cfg["merge_min_q"]:
            continue

        for j in range(i + 1, len(bank.patches)):
            if bank.patches[j].patch_id in merged_ids:
                continue
            if bank.patches[j].q_value < cfg["merge_min_q"]:
                continue

            # Check similarity
            sim = float(embs[i] @ embs[j] / (
                np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-8
            ))

            if sim >= cfg["merge_similarity"]:
                # Merge j into i (weighted by Q)
                _merge_patch_weights(bank.patches[i], bank.patches[j])
                merged_ids.add(bank.patches[j].patch_id)
                merged_count += 1

    # Remove merged patches
    bank.patches = [p for p in bank.patches if p.patch_id not in merged_ids]
    bank._index = {p.patch_id: i for i, p in enumerate(bank.patches)}
    bank._embeddings = None

    return merged_count


def _merge_patch_weights(target: CognitivePatch, source: CognitivePatch) -> None:
    """Merge source patch into target via Q-weighted average."""
    w_t = target.q_value
    w_s = source.q_value
    total = w_t + w_s + 1e-8

    for layer_key in set(target.lora_weights) | set(source.lora_weights):
        t_ab = target.lora_weights.get(layer_key, {})
        s_ab = source.lora_weights.get(layer_key, {})

        merged = {}
        for ab in ("A", "B"):
            t_mat = t_ab.get(ab)
            s_mat = s_ab.get(ab)

            if t_mat is not None and s_mat is not None:
                merged[ab] = (t_mat.float() * w_t + s_mat.float() * w_s) / total
            elif t_mat is not None:
                merged[ab] = t_mat
            elif s_mat is not None:
                merged[ab] = s_mat

        target.lora_weights[layer_key] = merged

    # Update metadata
    target.q_value = (w_t * target.q_visits + w_s * source.q_visits) / max(
        target.q_visits + source.q_visits, 1
    )
    target.q_visits += source.q_visits
    target.q_successes += source.q_successes
    target.q_failures += source.q_failures
    target.description += f" + merged({source.patch_id})"


def _prune_bad(bank: PatchBank, cfg: dict) -> int:
    """Remove patches that have been tried but never help."""
    before = len(bank.patches)
    bank.patches = [
        p for p in bank.patches
        if not (p.q_value < cfg["prune_max_q"] and p.q_visits >= cfg["prune_min_visits"])
    ]
    bank._index = {p.patch_id: i for i, p in enumerate(bank.patches)}
    bank._embeddings = None
    return before - len(bank.patches)


def _find_promotable(bank: PatchBank, cfg: dict) -> list[str]:
    """Find patches that are candidates for permanent merging.

    These patches have been reliably helpful across many tasks.
    Returns list of patch_ids that could be promoted to a persistent adapter.
    """
    return [
        p.patch_id for p in bank.patches
        if p.q_value >= cfg["promote_min_q"]
        and p.q_visits >= cfg["promote_min_visits"]
    ]

from types import SimpleNamespace

import numpy as np
import torch

from cogmem.patches.memory_bank import (
    ClusterMemory,
    ClusterMemoryBank,
    _apply_redundancy_penalties,
    compute_top_contrast_directions,
)
from cogmem.patches.patch import CognitivePatch


def _dummy_model():
    return SimpleNamespace(config=SimpleNamespace(num_hidden_layers=8))


def test_cluster_memory_bank_save_roundtrip(tmp_path):
    bank = ClusterMemoryBank(str(tmp_path / "cluster_memories"))
    episode = bank.record_episode(
        task_id="BigCodeBench/1",
        prompt="Draw a histogram with pandas and matplotlib",
        task_embedding=[0.1, 0.2, 0.3],
        failed_code="print('fail')",
        passed_code="print('pass')",
        pass_fail_similarity=0.72,
    )
    bank.memories = [
        ClusterMemory(
            memory_id="memory_plotting_1",
            family_label="plotting",
            centroid_embedding=[0.1, 0.2, 0.3],
            member_episode_ids=[episode.episode_id],
            support_count=1,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            q_value=0.4,
        )
    ]
    bank.save()

    loaded = ClusterMemoryBank(str(tmp_path / "cluster_memories"))
    loaded.load()

    assert len(loaded.episodes) == 1
    assert loaded.episodes[0].task_id == "BigCodeBench/1"
    assert len(loaded.memories) == 1
    assert loaded.memories[0].family_label == "plotting"


def test_compute_top_contrast_directions_prefers_dominant_axis():
    deltas = np.array(
        [
            [5.0, 0.1, 0.0],
            [4.5, 0.2, 0.0],
            [5.5, -0.1, 0.0],
        ],
        dtype=np.float32,
    )
    directions, explained = compute_top_contrast_directions(deltas, top_k=2)

    assert len(directions) == 2
    assert abs(directions[0][0]) > abs(directions[0][1])
    assert explained[0] > 0.9


def test_active_patches_skip_non_retrievable_memories(tmp_path):
    bank = ClusterMemoryBank(str(tmp_path / "cluster_memories"))
    patch = CognitivePatch(
        patch_id="patch_a",
        embedding=[0.1, 0.2],
        lora_weights={"layer": {"A": torch.zeros((1, 1)), "B": torch.zeros((1, 1))}},
    )
    bank.artifact_bank.add(patch)
    bank.memories = [
        ClusterMemory(
            memory_id="memory_a",
            family_label="plotting",
            centroid_embedding=[0.1, 0.2],
            member_episode_ids=["ep1"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            distilled_patch_ids=["patch_a"],
            distillation_success=0.0,
            retrievable=False,
        )
    ]
    bank.save()
    bank.load()

    active = bank.get_active_patches([0.1, 0.2], "plot a histogram", top_k=1)

    assert active == []


def test_retrieval_prefers_higher_q_when_similarity_is_close(tmp_path):
    bank = ClusterMemoryBank(str(tmp_path / "cluster_memories"))
    for patch_id in ("patch_hi", "patch_lo"):
        bank.artifact_bank.add(
            CognitivePatch(
                patch_id=patch_id,
                embedding=[0.0, 0.0],
                lora_weights={"layer": {"A": torch.zeros((1, 1)), "B": torch.zeros((1, 1))}},
            )
        )

    bank.memories = [
        ClusterMemory(
            memory_id="memory_low",
            family_label="plotting",
            centroid_embedding=[1.0, 0.0],
            member_episode_ids=["ep1", "ep2", "ep3"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            distilled_patch_ids=["patch_lo"],
            distillation_success=1.0,
            retrievable=True,
            q_value=0.10,
        ),
        ClusterMemory(
            memory_id="memory_high",
            family_label="plotting",
            centroid_embedding=[0.98, 0.02],
            member_episode_ids=["ep4", "ep5", "ep6"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            distilled_patch_ids=["patch_hi"],
            distillation_success=1.0,
            retrievable=True,
            q_value=0.90,
        ),
    ]
    bank.save()
    bank.load()

    selected = bank.get_active_memories([1.0, 0.0], "plot a dataframe histogram", top_k=1)

    assert selected[0].memory_id == "memory_high"


def test_redundancy_penalty_hits_near_duplicate_memories():
    memories = [
        ClusterMemory(
            memory_id="m1",
            family_label="plotting",
            centroid_embedding=[1.0, 0.0],
            member_episode_ids=["ep1", "ep2", "ep3"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
        ),
        ClusterMemory(
            memory_id="m2",
            family_label="plotting",
            centroid_embedding=[0.999, 0.001],
            member_episode_ids=["ep4", "ep5", "ep6"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
        ),
        ClusterMemory(
            memory_id="m3",
            family_label="file_io",
            centroid_embedding=[0.0, 1.0],
            member_episode_ids=["ep7", "ep8", "ep9"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
        ),
    ]

    _apply_redundancy_penalties(memories)

    assert memories[0].redundancy_penalty > 0.8
    assert memories[1].redundancy_penalty > 0.8
    assert memories[2].redundancy_penalty == 0.0


def test_build_memories_distills_and_retrieves_positive_cluster(monkeypatch, tmp_path):
    bank = ClusterMemoryBank(str(tmp_path / "cluster_memories"))
    for idx in range(3):
        bank.record_episode(
            task_id=f"BigCodeBench/{idx}",
            prompt="Draw a histogram using pandas and matplotlib",
            task_embedding=[1.0, 0.0, 0.0],
            failed_code=f"fail_{idx}",
            passed_code=f"pass_{idx}",
            pass_fail_similarity=0.8,
        )

    monkeypatch.setattr(
        "cogmem.patches.memory_bank._compute_episode_delta",
        lambda *args, **kwargs: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    def fake_create_patch(*args, **kwargs):
        patch_id = kwargs["patch_id"]
        patch = CognitivePatch(
            patch_id=patch_id,
            embedding=[1.0, 0.0, 0.0],
            lora_weights={"layer": {"A": torch.zeros((1, 1)), "B": torch.zeros((1, 1))}},
        )
        stats = SimpleNamespace(total_steps=5, final_loss=0.1, loss_history=[])
        return patch, stats

    monkeypatch.setattr(
        "cogmem.patches.memory_bank.create_patch_from_cluster",
        fake_create_patch,
    )
    monkeypatch.setattr(
        "cogmem.patches.memory_bank._measure_patch_teacher_forcing_gain",
        lambda *args, **kwargs: 0.4,
    )

    stats = bank.build_memories(_dummy_model(), tokenizer=None)

    assert stats["memories"] == 1
    assert bank.memories[0].retrievable is True
    active = bank.get_active_patches([1.0, 0.0, 0.0], "plot a histogram", top_k=1)
    assert len(active) == 1

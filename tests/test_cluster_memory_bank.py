from types import SimpleNamespace

import numpy as np
import torch

from cogmem.patches.memory_bank import (
    ClusterMemory,
    ClusterMemoryBank,
    _apply_redundancy_penalties,
    _apply_sleep_promotion_policies,
    _recompute_q_values,
    compute_top_contrast_directions,
    score_memory_final_use,
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


def test_build_memories_keeps_existing_bank_when_no_eligible_episodes(tmp_path):
    bank = ClusterMemoryBank(str(tmp_path / "cluster_memories"))
    bank.record_episode(
        task_id="BigCodeBench/1",
        prompt="plot a histogram",
        task_embedding=[0.1, 0.2, 0.3],
        failed_code="",
        passed_code="print('pass')",
        pass_fail_similarity=0.0,
        success=False,
    )
    bank.memories = [
        ClusterMemory(
            memory_id="memory_existing",
            family_label="plotting",
            centroid_embedding=[0.1, 0.2, 0.3],
            member_episode_ids=["ep1", "ep2", "ep3"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            q_value=0.5,
        )
    ]
    before_stats = bank.stats()

    stats = bank.build_memories(_dummy_model(), tokenizer=None)

    assert stats == before_stats
    assert len(bank.memories) == 1
    assert bank.memories[0].memory_id == "memory_existing"


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


def test_retrieval_prefers_higher_final_use_when_similarity_is_close(tmp_path):
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
            transfer_gain=0.55,
            recent_success_rate=0.55,
            promotion_score=0.10,
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
            transfer_gain=0.50,
            recent_success_rate=0.50,
            promotion_score=0.85,
        ),
    ]
    bank.save()
    bank.load()

    selected = bank.get_active_memories([1.0, 0.0], "plot a dataframe histogram", top_k=1)

    assert selected[0].memory_id == "memory_high"


def test_default_retrieval_uses_top1_unless_confidence_is_very_high(tmp_path):
    bank = ClusterMemoryBank(str(tmp_path / "cluster_memories"))
    for patch_id in ("patch_a", "patch_b"):
        bank.artifact_bank.add(
            CognitivePatch(
                patch_id=patch_id,
                embedding=[0.0, 0.0],
                lora_weights={"layer": {"A": torch.zeros((1, 1)), "B": torch.zeros((1, 1))}},
            )
        )

    bank.memories = [
        ClusterMemory(
            memory_id="memory_a",
            family_label="plotting",
            centroid_embedding=[1.0, 0.0],
            member_episode_ids=["ep1", "ep2", "ep3"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            positive_prototype=[1.0, 0.0],
            structural_markers=["histogram"],
            distilled_patch_ids=["patch_a"],
            distillation_success=1.0,
            retrievable=True,
            transfer_gain=0.60,
            recent_success_rate=0.60,
            promotion_score=0.40,
            retrieval_threshold=0.40,
        ),
        ClusterMemory(
            memory_id="memory_b",
            family_label="plotting",
            centroid_embedding=[0.99, 0.01],
            member_episode_ids=["ep4", "ep5", "ep6"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            positive_prototype=[0.99, 0.01],
            structural_markers=["histogram"],
            distilled_patch_ids=["patch_b"],
            distillation_success=1.0,
            retrievable=True,
            transfer_gain=0.58,
            recent_success_rate=0.58,
            promotion_score=0.35,
            retrieval_threshold=0.40,
        ),
    ]

    selected = bank.get_active_memories([1.0, 0.0], "plot a histogram", top_k=5)

    assert len(selected) == 1
    assert selected[0].memory_id == "memory_a"

    bank.memories = [
        ClusterMemory(
            memory_id="memory_a",
            family_label="plotting",
            centroid_embedding=[1.0, 0.0],
            member_episode_ids=["ep1", "ep2", "ep3"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            positive_prototype=[1.0, 0.0],
            structural_markers=["histogram"],
            distilled_patch_ids=["patch_a"],
            distillation_success=1.0,
            retrievable=True,
            transfer_gain=0.95,
            recent_success_rate=0.95,
            promotion_score=0.90,
            reuse_count=10,
            retrieval_threshold=0.40,
        ),
        ClusterMemory(
            memory_id="memory_b",
            family_label="plotting",
            centroid_embedding=[0.999, 0.001],
            member_episode_ids=["ep4", "ep5", "ep6"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            positive_prototype=[0.999, 0.001],
            structural_markers=["histogram"],
            distilled_patch_ids=["patch_b"],
            distillation_success=1.0,
            retrievable=True,
            transfer_gain=0.94,
            recent_success_rate=0.94,
            promotion_score=0.89,
            reuse_count=10,
            retrieval_threshold=0.40,
        ),
    ]

    from cogmem.patches import memory_bank as memory_bank_module
    original_confidence = memory_bank_module.DEFAULT_MULTI_MEMORY_CONFIDENCE
    try:
        memory_bank_module.DEFAULT_MULTI_MEMORY_CONFIDENCE = 0.55
        selected = bank.get_active_memories([1.0, 0.0], "plot a histogram", top_k=5)
    finally:
        memory_bank_module.DEFAULT_MULTI_MEMORY_CONFIDENCE = original_confidence

    selected_ids = {memory.memory_id for memory in selected}
    assert len(selected) > 1
    assert {"memory_a", "memory_b"}.issubset(selected_ids)


def test_retrieval_threshold_rejects_memory_below_gate(tmp_path):
    bank = ClusterMemoryBank(str(tmp_path / "cluster_memories"))
    bank.artifact_bank.add(
        CognitivePatch(
            patch_id="patch_low",
            embedding=[0.0, 0.0],
            lora_weights={"layer": {"A": torch.zeros((1, 1)), "B": torch.zeros((1, 1))}},
        )
    )
    bank.memories = [
        ClusterMemory(
            memory_id="memory_low_gate",
            family_label="networking",
            centroid_embedding=[1.0, 0.0],
            member_episode_ids=["ep1", "ep2", "ep3"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            positive_prototype=[1.0, 0.0],
            negative_prototype=[0.95, 0.05],
            distilled_patch_ids=["patch_low"],
            distillation_success=1.0,
            retrievable=True,
            transfer_gain=0.9,
            recent_success_rate=0.9,
            retrieval_threshold=0.35,
        ),
    ]

    selected = bank.get_active_memories([0.95, 0.05], "open a socket on a url", top_k=1)

    assert selected == []


def test_structural_match_can_break_similarity_tie(tmp_path):
    bank = ClusterMemoryBank(str(tmp_path / "cluster_memories"))
    for patch_id in ("patch_sort", "patch_plot"):
        bank.artifact_bank.add(
            CognitivePatch(
                patch_id=patch_id,
                embedding=[0.0, 0.0],
                lora_weights={"layer": {"A": torch.zeros((1, 1)), "B": torch.zeros((1, 1))}},
            )
        )
    bank.memories = [
        ClusterMemory(
            memory_id="memory_sort",
            family_label="general_code",
            centroid_embedding=[1.0, 0.0],
            member_episode_ids=["ep1", "ep2", "ep3"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            structural_markers=["sorted", "mutation"],
            distilled_patch_ids=["patch_sort"],
            distillation_success=1.0,
            retrievable=True,
            transfer_gain=0.6,
            recent_success_rate=0.6,
        ),
        ClusterMemory(
            memory_id="memory_plot",
            family_label="plotting",
            centroid_embedding=[1.0, 0.0],
            member_episode_ids=["ep4", "ep5", "ep6"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            structural_markers=["histogram", "matplotlib"],
            distilled_patch_ids=["patch_plot"],
            distillation_success=1.0,
            retrievable=True,
            transfer_gain=0.6,
            recent_success_rate=0.6,
        ),
    ]

    selected = bank.get_active_memories(
        [1.0, 0.0],
        "return a sorted copy without mutating the input list",
        top_k=1,
    )

    assert selected[0].memory_id == "memory_sort"


def test_retrievable_payload_groups_metadata_evidence_and_transfer_stats():
    memory = ClusterMemory(
        memory_id="memory_transfer",
        family_label="general_code",
        centroid_embedding=[1.0, 0.0],
        member_episode_ids=["ep1", "ep2", "ep3"],
        support_count=3,
        layer_window=[4, 5, 6, 7],
        token_window=64,
        negative_episode_ids=["ep9"],
        positive_prototype=[1.0, 0.0],
        negative_prototype=[0.0, 1.0],
        structural_markers=["sorted", "mutation"],
        local_support_gain=0.6,
        held_out_steering_gain=0.5,
        transfer_gain=0.4,
        distilled_patch_ids=["patch_transfer"],
        retrieval_threshold=0.3,
    )

    payload = memory.retrievable_payload()

    assert set(payload) == {"cluster_metadata", "evidence", "transfer_stats", "patch_ids"}
    assert payload["cluster_metadata"]["memory_id"] == "memory_transfer"
    assert payload["evidence"]["negative_episode_ids"] == ["ep9"]
    assert payload["transfer_stats"]["local_support_gain"] == 0.6
    assert payload["patch_ids"] == ["patch_transfer"]


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


def test_promotion_score_penalizes_unseen_hurt():
    safe = ClusterMemory(
        memory_id="m_safe",
        family_label="general_code",
        centroid_embedding=[1.0, 0.0],
        member_episode_ids=["ep1", "ep2", "ep3"],
        support_count=3,
        layer_window=[4, 5, 6, 7],
        token_window=64,
        distilled_patch_ids=["patch_safe"],
        distillation_success=1.0,
        held_out_steering_gain=0.4,
        transfer_rate=0.8,
    )
    risky = ClusterMemory(
        memory_id="m_risky",
        family_label="general_code",
        centroid_embedding=[1.0, 0.0],
        member_episode_ids=["ep4", "ep5", "ep6"],
        support_count=3,
        layer_window=[4, 5, 6, 7],
        token_window=64,
        distilled_patch_ids=["patch_risky"],
        distillation_success=1.0,
        held_out_steering_gain=0.4,
        transfer_rate=0.8,
        unseen_hurt_count=3,
    )

    _recompute_q_values([safe, risky])

    assert safe.promotion_score > risky.promotion_score
    assert safe.q_value > risky.q_value


def test_recompute_q_values_preserves_heldout_transfer_gain_and_tracks_online_gain():
    memory = ClusterMemory(
        memory_id="m_transfer",
        family_label="general_code",
        centroid_embedding=[1.0, 0.0],
        member_episode_ids=["ep1", "ep2", "ep3"],
        support_count=3,
        layer_window=[4, 5, 6, 7],
        token_window=64,
        distilled_patch_ids=["patch_transfer"],
        distillation_success=1.0,
        transfer_gain=0.72,
        seen_help_count=1,
        seen_hurt_count=1,
        unseen_help_count=1,
        unseen_hurt_count=1,
    )

    _recompute_q_values([memory])

    assert memory.transfer_gain == 0.72
    assert memory.transfer_online_gain == 0.5


def test_merge_memories_preserves_transfer_counters(tmp_path):
    bank = ClusterMemoryBank(str(tmp_path / "cluster_memories"))
    bank.memories = [
        ClusterMemory(
            memory_id="memory_a",
            family_label="general_code",
            centroid_embedding=[1.0, 0.0],
            member_episode_ids=["ep1", "ep2", "ep3"],
            support_count=3,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            seen_help_count=2,
            seen_hurt_count=1,
            unseen_help_count=3,
            unseen_hurt_count=4,
        )
    ]
    merged = bank._merge_memories(
        [
            ClusterMemory(
                memory_id="memory_a",
                family_label="general_code",
                centroid_embedding=[1.0, 0.0],
                member_episode_ids=["ep1", "ep2", "ep3"],
                support_count=3,
                layer_window=[4, 5, 6, 7],
                token_window=64,
            )
        ],
        {"ep1", "ep2", "ep3"},
    )

    assert merged[0].seen_help_count == 2
    assert merged[0].seen_hurt_count == 1
    assert merged[0].unseen_help_count == 3
    assert merged[0].unseen_hurt_count == 4


def test_unseen_hurt_demotes_memory_from_retrieval():
    risky = ClusterMemory(
        memory_id="m_risky",
        family_label="general_code",
        centroid_embedding=[1.0, 0.0],
        member_episode_ids=["ep4", "ep5", "ep6"],
        support_count=3,
        layer_window=[4, 5, 6, 7],
        token_window=64,
        distilled_patch_ids=["patch_risky"],
        distillation_success=1.0,
        local_support_gain=0.8,
        held_out_steering_gain=0.8,
        transfer_gain=0.7,
        unseen_help_count=0,
        unseen_hurt_count=2,
        retrievable=True,
    )

    _recompute_q_values([risky])
    demoted = _apply_sleep_promotion_policies([risky], prune=False)

    assert demoted[0].retrievable is False


def test_update_memory_utility_recomputes_retrieval_threshold(tmp_path):
    bank = ClusterMemoryBank(str(tmp_path / "cluster_memories"))
    episode = bank.record_episode(
        task_id="BigCodeBench/1",
        prompt="plot a histogram",
        task_embedding=[1.0, 0.0],
        failed_code="fail",
        passed_code="pass",
        pass_fail_similarity=0.8,
    )
    bank.record_episode(
        task_id="BigCodeBench/2",
        prompt="open a socket and check a port",
        task_embedding=[0.0, 1.0],
        failed_code="fail",
        passed_code="pass",
        pass_fail_similarity=0.8,
    )
    bank.memories = [
        ClusterMemory(
            memory_id="memory_plot",
            family_label="plotting",
            centroid_embedding=[1.0, 0.0],
            member_episode_ids=[episode.episode_id],
            support_count=1,
            layer_window=[4, 5, 6, 7],
            token_window=64,
            positive_prototype=[1.0, 0.0],
            negative_prototype=[0.0, 1.0],
            distilled_patch_ids=["patch_plot"],
            distillation_success=1.0,
            retrievable=True,
            q_value=0.1,
            retrieval_threshold=0.0,
        )
    ]
    bank._memory_index = {"memory_plot": 0}

    bank.update_memory_utility("memory_plot", task_succeeded=True, persist=False)

    assert bank.memories[0].retrieval_threshold != 0.0


def test_load_patches_for_memories_uses_final_use_score(tmp_path):
    bank = ClusterMemoryBank(str(tmp_path / "cluster_memories"))
    patch = CognitivePatch(
        patch_id="patch_plot",
        embedding=[1.0, 0.0],
        lora_weights={"layer": {"A": torch.zeros((1, 1)), "B": torch.zeros((1, 1))}},
    )
    bank.artifact_bank.add(patch)
    memory = ClusterMemory(
        memory_id="memory_plot",
        family_label="plotting",
        centroid_embedding=[1.0, 0.0],
        member_episode_ids=["ep1", "ep2", "ep3"],
        support_count=3,
        layer_window=[4, 5, 6, 7],
        token_window=64,
        structural_markers=["histogram"],
        distilled_patch_ids=["patch_plot"],
        distillation_success=1.0,
        retrievable=True,
        promotion_score=0.9,
        transfer_gain=0.5,
        recent_success_rate=0.5,
    )
    bank.memories = [memory]

    loaded = bank.load_patches_for_memories(
        [memory],
        query_embedding=[1.0, 0.0],
        task_prompt="plot a histogram",
    )

    assert len(loaded) == 1
    expected = score_memory_final_use(
        memory,
        np.asarray([1.0, 0.0], dtype=np.float32),
        "plot a histogram",
        max_reuse=0,
    )
    assert loaded[0].q_value == expected
    assert loaded[0].q_value != memory.promotion_score


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

import json
import pytest
from cogmem.consolidation.abstract import (
    episode_to_training_pair,
    prepare_training_dataset,
    q_weighted_duplicates,
    save_as_jsonl,
)


class TestEpisodeToTrainingPair:
    def test_basic_conversion(self, sample_episodes):
        ep = sample_episodes[0]  # successful, clean task
        pair = episode_to_training_pair(ep)
        assert pair["instruction"] == ep["task_description"]
        assert pair["response"] == ep["script"]
        assert pair["weight"] == ep["q_value"]
        assert pair["source_episode"] == ep["episode_id"]

    def test_skips_failed_episodes(self, sample_episodes):
        ep = sample_episodes[7]  # failed
        pair = episode_to_training_pair(ep)
        assert pair is None


class TestQWeightedDuplicates:
    def test_high_q_gets_more_copies(self):
        pair = {"instruction": "x", "response": "y", "weight": 0.9}
        copies = q_weighted_duplicates([pair])
        assert len(copies) == 3  # round(0.9 * 3) = 3

    def test_medium_q_gets_two_copies(self):
        pair = {"instruction": "x", "response": "y", "weight": 0.7}
        copies = q_weighted_duplicates([pair])
        assert len(copies) == 2  # round(0.7 * 3) = 2

    def test_low_q_gets_one_copy(self):
        pair = {"instruction": "x", "response": "y", "weight": 0.2}
        copies = q_weighted_duplicates([pair])
        assert len(copies) == 1  # max(1, round(0.2 * 3)) = 1


class TestPrepareTrainingDataset:
    def test_only_successful(self, sample_episodes, sample_replay_buffer):
        dataset = prepare_training_dataset(
            sample_episodes, replay_buffer=sample_replay_buffer
        )
        # The fixture has 10 successful episodes + 2 replay buffer entries = 12 base pairs
        # Note: only successful episodes with non-empty scripts are included
        source_eps = {p.get("source_episode") for p in dataset}
        failed_ids = {ep["episode_id"] for ep in sample_episodes if not ep["success"]}
        assert not source_eps & failed_ids

    def test_includes_replay_buffer(self, sample_episodes, sample_replay_buffer):
        dataset = prepare_training_dataset(
            sample_episodes, replay_buffer=sample_replay_buffer
        )
        replay_instructions = {r["instruction"] for r in sample_replay_buffer}
        dataset_instructions = {p["instruction"] for p in dataset}
        assert replay_instructions.issubset(dataset_instructions)


class TestSaveAsJsonl:
    def test_jsonl_format(self, tmp_path):
        pairs = [
            {"instruction": "do task", "response": "step 1\nstep 2", "weight": 0.9},
        ]
        path = str(tmp_path / "train.jsonl")
        save_as_jsonl(pairs, path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # round(0.9 * 3) = 3 copies due to Q-weighted duplication
        obj = json.loads(lines[0])
        assert "messages" in obj
        assert obj["messages"][0]["role"] == "user"
        assert obj["messages"][1]["role"] == "assistant"

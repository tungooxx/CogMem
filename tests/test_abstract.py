import json
import pytest
from cogmem.consolidation.abstract import (
    episode_to_training_pair,
    prepare_skill_training_dataset,
    prepare_training_dataset,
    q_weighted_duplicates,
    save_as_jsonl,
    skill_card_to_training_pairs,
)


class TestEpisodeToTrainingPair:
    def test_basic_conversion(self, sample_episodes):
        ep = sample_episodes[0]  # successful, clean task
        pair = episode_to_training_pair(ep)
        assert pair["instruction"] == ep["task_description"]
        assert pair["response"] == ep["script"]
        assert pair["weight"] == ep["q_value"]
        assert pair["source_episode"] == ep["episode_id"]

    def test_prefers_episode_helpfulness_when_present(self, sample_episodes):
        ep = dict(sample_episodes[0])
        ep["episode_helpfulness"] = 0.61
        pair = episode_to_training_pair(ep)
        assert pair["weight"] == 0.61

    def test_uses_final_code_when_script_missing(self, sample_episodes):
        ep = dict(sample_episodes[0])
        ep["script"] = ""
        ep["final_code"] = "def solve():\n    return 1\n"
        pair = episode_to_training_pair(ep)
        assert pair["response"] == ep["final_code"]

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

    def test_filters_by_allowed_manifest_ids(self, sample_episodes):
        from cogmem.config import CogMemConfig

        episodes = []
        for idx, ep in enumerate(sample_episodes[:4]):
            clone = dict(ep)
            clone["manifest_id"] = "manifest_a" if idx < 2 else "manifest_b"
            clone["success"] = True
            episodes.append(clone)

        dataset = prepare_training_dataset(
            episodes,
            config=CogMemConfig(allowed_manifest_ids=["manifest_a"]),
        )

        assert all(pair["source_episode"] in {episodes[0]["episode_id"], episodes[1]["episode_id"]} for pair in dataset)


class TestPrepareSkillTrainingDataset:
    def test_builds_pairs_from_skill_card_evidence(self):
        episodes = [
            {
                "episode_id": "ep_a",
                "task_description": "sort a list safely",
                "task_type": "sorting",
                "script": "def solve(xs):\n    return sorted(xs)\n",
                "success": True,
                "q_value": 0.8,
                "episode_helpfulness": 0.8,
                "manifest_id": "manifest_a",
            },
            {
                "episode_id": "ep_b",
                "task_description": "sort numbers without mutation",
                "task_type": "sorting",
                "script": "def solve(xs):\n    return sorted(xs)\n",
                "success": True,
                "q_value": 0.7,
                "episode_helpfulness": 0.7,
                "manifest_id": "manifest_a",
            },
        ]
        card = {
            "skill_id": "skill_sorting",
            "manifest_ids": ["manifest_a"],
            "evidence_episode_ids": ["ep_a", "ep_b"],
            "confidence": 0.9,
            "transfer_gain": 0.6,
        }

        pairs = prepare_skill_training_dataset([card], episodes)

        assert [pair["source_episode"] for pair in pairs] == ["ep_a", "ep_b"]
        assert all(pair["source_skill_card"] == "skill_sorting" for pair in pairs)
        assert all(0.01 <= pair["weight"] <= 1.0 for pair in pairs)

    def test_filters_skill_cards_by_manifest(self):
        from cogmem.config import CogMemConfig

        episodes = [
            {
                "episode_id": "ep_a",
                "task_description": "parse json",
                "task_type": "parsing",
                "script": "import json\njson.loads(data)\n",
                "success": True,
                "q_value": 0.9,
                "episode_helpfulness": 0.9,
                "manifest_id": "manifest_a",
            },
        ]
        card = {
            "skill_id": "skill_parse",
            "manifest_ids": ["manifest_b"],
            "evidence_episode_ids": ["ep_a"],
            "confidence": 0.8,
            "transfer_gain": 0.5,
        }

        pairs = prepare_skill_training_dataset(
            [card],
            episodes,
            config=CogMemConfig(allowed_manifest_ids=["manifest_a"]),
        )

        assert pairs == []


class TestSkillCardToTrainingPairs:
    def test_merges_episode_and_card_signal_into_weight(self):
        episode = {
            "episode_id": "ep_1",
            "task_description": "read csv from disk",
            "task_type": "file_io",
            "script": "import pandas as pd\npd.read_csv(path)\n",
            "success": True,
            "q_value": 0.6,
            "episode_helpfulness": 0.6,
        }
        card = {
            "skill_id": "skill_csv",
            "evidence_episode_ids": ["ep_1"],
            "confidence": 0.9,
            "transfer_gain": 0.7,
        }

        pairs = skill_card_to_training_pairs(card, {"ep_1": episode})

        assert len(pairs) == 1
        assert pairs[0]["weight"] > 0.6
        assert pairs[0]["skill_confidence"] == 0.9


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

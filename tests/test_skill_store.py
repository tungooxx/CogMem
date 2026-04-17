from cogmem.memory.skill_store import SkillStore, normalize_skill_card


def test_normalize_skill_card_sets_required_fields():
    card = normalize_skill_card(
        {
            "task_type": "file_io",
            "domain": "file_io",
            "triggers": ["glob", "files", "glob"],
            "plan_steps": ["list files", "read file"],
            "evidence_episode_ids": ["ep_1", "ep_2"],
            "manifest_ids": ["manifest_a"],
            "transfer_gain": 0.25,
            "confidence": 0.8,
        },
        copy_card=True,
    )

    assert card["skill_id"].startswith("skill_")
    assert card["triggers"] == ["glob", "files"]
    assert card["plan_steps"] == ["list files", "read file"]
    assert card["status"] == "candidate"
    assert card["card_transfer_gain"] == 0.25
    assert card["retrieval_confidence"] == 0.8


def test_skill_store_roundtrip_and_filtering(tmp_path):
    cards = [
        {
            "skill_id": "skill_a",
            "task_type": "file_io",
            "domain": "file_io",
            "triggers": ["glob"],
            "plan_steps": ["list files"],
            "anti_patterns": ["avoid FileNotFoundError"],
            "evidence_episode_ids": ["ep_1"],
            "manifest_ids": ["manifest_a"],
            "transfer_gain": 0.2,
            "confidence": 0.7,
            "status": "promoted",
        },
        {
            "skill_id": "skill_b",
            "task_type": "networking",
            "domain": "general",
            "triggers": ["api"],
            "plan_steps": ["call api"],
            "anti_patterns": [],
            "evidence_episode_ids": ["ep_2"],
            "manifest_ids": ["manifest_b"],
            "transfer_gain": 0.01,
            "confidence": 0.4,
            "status": "candidate",
        },
    ]
    path = tmp_path / "skills.json"
    store = SkillStore(cards)
    store.save(str(path))
    loaded = SkillStore.load(str(path))

    assert len(loaded) == 2
    assert loaded.get("skill_a")["status"] == "promoted"
    assert loaded.filter(task_type="file_io")[0]["skill_id"] == "skill_a"
    assert loaded.filter(manifest_id="manifest_a")[0]["skill_id"] == "skill_a"
    assert loaded.filter(promoted=True)[0]["skill_id"] == "skill_a"
    summary = loaded.summary()
    assert summary["total"] == 2
    assert summary["promoted"] == 1

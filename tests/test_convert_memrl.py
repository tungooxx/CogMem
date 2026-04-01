import json
import pytest
from scripts.convert_memrl import convert_textual_memory_item, convert_cube_dump


@pytest.fixture
def sample_memrl_item():
    """Simulates a single item from MemRL's textual_memory.json."""
    return {
        "id": "mem-uuid-001",
        "memory": "put a clean mug in shelf 1",
        "metadata": {
            "type": "procedure",
            "source_benchmark": "alfworld",
            "full_content": "Task: put a clean mug in shelf 1\n\nProcedure:\n1. Find mug on countertop 1\n2. Take mug\n3. Go to sinkbasin 1\n4. Clean mug\n5. Go to shelf 1\n6. Put mug in shelf 1",
            "success": True,
            "q_value": 0.89,
            "q_visits": 5,
            "confidence": 0.85,
            "q_updated_at": "2026-04-01T10:30:00",
            "last_used_at": "2026-04-01T10:30:00",
        },
    }


def test_convert_single_item(sample_memrl_item):
    result = convert_textual_memory_item(sample_memrl_item, episode_num=1)
    assert result["episode_id"] == "ep_001"
    assert result["task_description"] == "put a clean mug in shelf 1"
    assert result["success"] is True
    assert result["q_value"] == 0.89
    assert result["q_visits"] == 5
    assert "script" in result
    assert result["script"].startswith("1.")


def test_convert_extracts_task_type(sample_memrl_item):
    result = convert_textual_memory_item(sample_memrl_item, episode_num=1)
    assert result["task_type"] == "clean"


def test_convert_cube_dump(tmp_path, sample_memrl_item):
    cube_dir = tmp_path / "cube"
    cube_dir.mkdir()
    (cube_dir / "textual_memory.json").write_text(json.dumps([sample_memrl_item]))

    output_path = str(tmp_path / "memory_bank.json")
    convert_cube_dump(str(cube_dir), output_path)

    with open(output_path) as f:
        bank = json.load(f)
    assert len(bank) == 1
    assert bank[0]["episode_id"] == "ep_001"

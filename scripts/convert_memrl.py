"""Convert MemRL cube dump to CogMem memory bank JSON format.

Usage:
    python scripts/convert_memrl.py <cube_dir> <output_path>

Example:
    python scripts/convert_memrl.py MemRL/results/snapshot/10/cube results/memory_bank.json
"""

import json
import re
import sys
from pathlib import Path

TASK_TYPE_PATTERNS = {
    "clean": r"clean",
    "heat": r"hot|heat",
    "cool": r"cool|cold",
    "examine": r"examine|look",
    "puttwo": r"put two|puttwo",
    "pick": r"put a|put an|place",
}


def _infer_task_type(task_description: str) -> str:
    desc = task_description.lower()
    for task_type, pattern in TASK_TYPE_PATTERNS.items():
        if re.search(pattern, desc):
            return task_type
    return "unknown"


def _extract_script(full_content: str) -> str:
    lines = full_content.split("\n")
    script_lines = []
    in_procedure = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("procedure:"):
            in_procedure = True
            continue
        if in_procedure and stripped:
            script_lines.append(stripped)
        elif re.match(r"^\d+\.", stripped):
            script_lines.append(stripped)
    return "\n".join(script_lines) if script_lines else full_content


def convert_textual_memory_item(item: dict, episode_num: int) -> dict:
    payload = item.get("payload", item)
    meta = payload.get("metadata", item.get("metadata", {}))
    full_content = meta.get("full_content", "")
    task_desc = payload.get("memory", item.get("memory", ""))

    return {
        "episode_id": f"ep_{episode_num:03d}",
        "task_description": task_desc,
        "task_type": _infer_task_type(task_desc),
        "script": _extract_script(full_content),
        "intent_embedding": [],  # re-embed with LocalEmbedder later
        "success": meta.get("success", False),
        "q_value": meta.get("q_value", 0.0),
        "q_visits": meta.get("q_visits", 0),
        "num_steps": 0,  # not available in cube dump
        "timestamp": meta.get("q_updated_at", ""),
    }


def convert_cube_dump(cube_dir: str, output_path: str) -> str:
    tm_path = Path(cube_dir) / "textual_memory.json"
    with open(tm_path, encoding="utf-8") as f:
        items = json.load(f)

    episodes = []
    for i, item in enumerate(items, 1):
        ep = convert_textual_memory_item(item, episode_num=i)
        episodes.append(ep)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2)

    print(f"Converted {len(episodes)} episodes to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert_cube_dump(sys.argv[1], sys.argv[2])

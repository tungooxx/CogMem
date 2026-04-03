"""Build Phase 2 training data by combining Phase 1 Q-values with expert trajectories.

For each memory from Phase 1:
1. Match task type to ALFWorld expert trajectory
2. Convert expert actions to ReAct format (Thought + Action)
3. Weight by Q-value (lower Q = more training copies)

Usage:
    python scripts/build_training_data.py results/memory_bank.json results/training.jsonl

Requires ALFWorld data in WSL at ~/.cache/alfworld/json_2.1.1/
"""

import json
import os
import re
import sys
from pathlib import Path

# ALFWorld task type mapping: our task_type -> folder prefix
TASK_TYPE_TO_FOLDER = {
    "examine": "look_at_obj_in_light",
    "clean": "pick_clean_then_place_in_recep",
    "heat": "pick_heat_then_place_in_recep",
    "cool": "pick_cool_then_place_in_recep",
    "pick": "pick_and_place_simple",
    "puttwo": "pick_two_obj_and_place",
}

# ALFWorld action -> ReAct action format
ACTION_MAP = {
    "GotoLocation": "go to {args}",
    "PickupObject": "take {args} from {loc}",
    "PutObject": "move {obj} to {args}",
    "CleanObject": "clean {args} with sinkbasin 1",
    "HeatObject": "heat {args} with microwave 1",
    "CoolObject": "cool {args} with fridge 1",
    "ToggleObject": "use {args}",
}

# Thought templates per action type
THOUGHT_MAP = {
    "GotoLocation": "I need to go to the {args} to continue the task.",
    "PickupObject": "I see the {args} here. Let me pick it up.",
    "PutObject": "Now I need to place the {obj} on the {args}.",
    "CleanObject": "I need to clean the {args} using the sink.",
    "HeatObject": "I need to heat the {args} using the microwave.",
    "CoolObject": "I need to cool the {args} using the fridge.",
    "ToggleObject": "I need to turn on the {args}.",
}


def expert_to_react(expert_actions: list[dict]) -> str:
    """Convert expert high-level actions to ReAct format."""
    lines = []
    for action in expert_actions:
        act_type = action["action"]
        args = " ".join(action.get("args", []))

        if act_type == "NoOp":
            continue

        thought = THOUGHT_MAP.get(act_type, f"I need to {act_type} {args}.")
        thought = thought.format(args=args, obj=args, loc=args)

        act_text = ACTION_MAP.get(act_type, f"{act_type} {args}")
        act_text = act_text.format(args=args, obj=args, loc=args)

        lines.append(f"Thought: {thought}")
        lines.append(f"Action: {act_text}")

    return "\n".join(lines)


def load_expert_trajectories(alfworld_data_dir: str) -> dict:
    """Load all expert trajectories grouped by folder prefix (task type)."""
    train_dir = Path(alfworld_data_dir) / "json_2.1.1" / "train"
    experts = {}

    for game_dir in train_dir.iterdir():
        if not game_dir.is_dir():
            continue
        # Get first trial
        trials = list(game_dir.iterdir())
        if not trials:
            continue
        traj_file = trials[0] / "traj_data.json"
        if not traj_file.exists():
            continue

        with open(traj_file, encoding="utf-8") as f:
            data = json.load(f)

        plan = data.get("plan", {}).get("high_pddl", [])
        actions = [p["discrete_action"] for p in plan]
        task_desc = (
            data.get("turk_annotations", {})
            .get("anns", [{}])[0]
            .get("task_desc", "")
        )

        # Extract folder prefix as task type
        folder_name = game_dir.name
        prefix = folder_name.rsplit("-", 1)[0]  # remove trailing number
        prefix = re.sub(r"-\d+$", "", folder_name)

        if prefix not in experts:
            experts[prefix] = []
        experts[prefix].append({
            "folder": folder_name,
            "task_desc": task_desc,
            "actions": actions,
            "react": expert_to_react(actions),
        })

    return experts


def q_to_copies(q_value: float, max_copies: int = 5) -> int:
    """Convert Q-value to training copies. Lower Q = more copies."""
    # Q range: [-0.88, 0.0]
    # Invert: Q=-0.88 -> 5 copies, Q=0.0 -> 1 copy
    normalized = abs(q_value)  # 0.0 to 0.88
    copies = 1 + int(normalized * (max_copies - 1) / 0.9)
    return min(copies, max_copies)


def build_training_data(
    memory_bank_path: str,
    alfworld_data_dir: str,
    output_path: str,
) -> str:
    # Load memories
    with open(memory_bank_path, encoding="utf-8") as f:
        memories = json.load(f)

    # Load expert trajectories
    print("Loading expert trajectories...")
    experts = load_expert_trajectories(alfworld_data_dir)
    print(f"Loaded {sum(len(v) for v in experts.values())} expert trajectories")
    print(f"Task types: {list(experts.keys())}")

    # Build training pairs
    training_data = []
    matched = 0
    unmatched = 0

    for mem in memories:
        task_type = mem["task_type"]
        folder_prefix = TASK_TYPE_TO_FOLDER.get(task_type)

        if not folder_prefix:
            unmatched += 1
            continue

        # Find matching expert trajectories
        matching = []
        for prefix, trajs in experts.items():
            if prefix.startswith(folder_prefix):
                matching.extend(trajs)

        if not matching:
            unmatched += 1
            continue

        # Pick a random expert trajectory of same type
        import hashlib
        idx = int(hashlib.md5(mem["episode_id"].encode()).hexdigest(), 16) % len(matching)
        expert = matching[idx]

        # Build training example
        q_value = mem["q_value"]
        copies = q_to_copies(q_value)

        example = {
            "messages": [
                {"role": "user", "content": mem["task_description"]},
                {"role": "assistant", "content": expert["react"]},
            ]
        }

        for _ in range(copies):
            training_data.append(example)

        matched += 1

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nMatched: {matched}/{len(memories)}")
    print(f"Unmatched: {unmatched}")
    print(f"Total training examples (Q-weighted): {len(training_data)}")
    print(f"Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/build_training_data.py <memory_bank.json> <alfworld_data_dir> <output.jsonl>")
        sys.exit(1)
    build_training_data(sys.argv[1], sys.argv[2], sys.argv[3])

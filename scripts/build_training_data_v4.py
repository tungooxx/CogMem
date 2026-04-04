"""Build Phase 2 training data v4 — Q-weighted only, higher copies.

Only uses the 248 Q-valued memories from Phase 1.
Max copies increased to 10 (harder tasks trained 10x more).
Dual format (MemRL + raw).

Usage:
    python scripts/build_training_data_v4.py results/memory_bank.json results/training_v4.jsonl
"""

import json
import hashlib
import re
import sys
from pathlib import Path

SYSTEM_PROMPT = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\nAction: your next action".

The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. move {obj} to {recep}
4. open {recep}
5. close {recep}
6. use {obj}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>"""

TASK_TYPE_TO_FOLDER = {
    "examine": "look_at_obj_in_light",
    "clean": "pick_clean_then_place_in_recep",
    "heat": "pick_heat_then_place_in_recep",
    "cool": "pick_cool_then_place_in_recep",
    "pick": "pick_and_place_simple",
    "puttwo": "pick_two_obj_and_place",
}

ACTION_MAP = {
    "GotoLocation": "go to {args} 1",
    "PickupObject": "take {args} 1 from {loc} 1",
    "PutObject": "move {obj} 1 to {args} 1",
    "CleanObject": "clean {args} 1 with sinkbasin 1",
    "HeatObject": "heat {args} 1 with microwave 1",
    "CoolObject": "cool {args} 1 with fridge 1",
    "ToggleObject": "use {args} 1",
}

THOUGHT_MAP = {
    "GotoLocation": "I need to go to the {args} to continue the task.",
    "PickupObject": "I found the {args}. Let me pick it up.",
    "PutObject": "Now I need to place the {obj} on the {args}.",
    "CleanObject": "I need to clean the {args} using the sink.",
    "HeatObject": "I need to heat the {args} using the microwave.",
    "CoolObject": "I need to cool the {args} using the fridge.",
    "ToggleObject": "I need to turn on the {args}.",
}


def expert_to_first_action(expert_actions):
    for action in expert_actions:
        act_type = action["action"]
        args = " ".join(action.get("args", [])).lower()
        if act_type == "NoOp":
            continue
        thought = THOUGHT_MAP.get(act_type, f"I need to {act_type} {args}.").format(args=args, obj=args, loc=args)
        act_text = ACTION_MAP.get(act_type, f"{act_type} {args}").format(args=args, obj=args, loc=args)
        return f"Thought: {thought}\nAction: {act_text}"
    return "Thought: I need to look around.\nAction: look"


def expert_to_full_plan(expert_actions):
    lines = []
    for action in expert_actions:
        act_type = action["action"]
        args = " ".join(action.get("args", [])).lower()
        if act_type == "NoOp":
            continue
        thought = THOUGHT_MAP.get(act_type, f"I need to {act_type} {args}.").format(args=args, obj=args, loc=args)
        act_text = ACTION_MAP.get(act_type, f"{act_type} {args}").format(args=args, obj=args, loc=args)
        lines.append(f"Thought: {thought}")
        lines.append(f"Action: {act_text}")
    return "\n".join(lines)


def load_expert_trajectories(alfworld_data_dir):
    train_dir = Path(alfworld_data_dir) / "json_2.1.1" / "train"
    experts = {}
    for game_dir in train_dir.iterdir():
        if not game_dir.is_dir():
            continue
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
        folder_name = game_dir.name
        prefix = re.sub(r"-\d+$", "", folder_name)
        if prefix not in experts:
            experts[prefix] = []
        experts[prefix].append({"folder": folder_name, "actions": actions})
    return experts


def load_few_shot_examples(path):
    with open(path, encoding="utf-8") as f:
        examples = json.load(f)
    return {ex["task"]: ex["example"] for ex in examples}


def q_to_copies(q_value, max_copies=10):
    """Lower Q = more copies. Max 10 copies for hardest tasks."""
    normalized = abs(q_value)  # 0.0 to 0.88
    copies = 1 + int(normalized * (max_copies - 1) / 0.9)
    return min(copies, max_copies)


def build_training_data(memory_bank_path, alfworld_data_dir, few_shot_path, output_path):
    with open(memory_bank_path, encoding="utf-8") as f:
        memories = json.load(f)

    print("Loading expert trajectories...")
    experts = load_expert_trajectories(alfworld_data_dir)
    print(f"Loaded {sum(len(v) for v in experts.values())} expert trajectories")

    few_shot = load_few_shot_examples(few_shot_path)
    training_data = []
    matched = 0

    for mem in memories:
        task_type = mem["task_type"]
        folder_prefix = TASK_TYPE_TO_FOLDER.get(task_type)
        if not folder_prefix:
            continue

        matching = []
        for prefix, trajs in experts.items():
            if prefix.startswith(folder_prefix):
                matching.extend(trajs)
        if not matching:
            continue

        idx = int(hashlib.md5(mem["episode_id"].encode()).hexdigest(), 16) % len(matching)
        expert = matching[idx]
        copies = q_to_copies(mem["q_value"])
        first_action = expert_to_first_action(expert["actions"])
        full_plan = expert_to_full_plan(expert["actions"])

        # MemRL format
        memrl_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if task_type in few_shot:
            example_copy = json.loads(json.dumps(few_shot[task_type]))
            example_copy[0]["content"] = "Here is an example of how to solve the task:\n" + example_copy[0]["content"]
            memrl_messages.extend(example_copy)
        memrl_messages.append({
            "role": "user",
            "content": f"Now, it's your turn to solve a new task.\n{mem['task_description']}"
        })
        memrl_messages.append({"role": "assistant", "content": first_action})

        # Raw format
        raw_messages = [
            {"role": "user", "content": mem["task_description"]},
            {"role": "assistant", "content": full_plan},
        ]

        for _ in range(copies):
            training_data.append({"messages": memrl_messages})
            training_data.append({"messages": raw_messages})

        matched += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats
    copy_counts = {}
    for mem in memories:
        c = q_to_copies(mem["q_value"])
        copy_counts[c] = copy_counts.get(c, 0) + 1

    print(f"\nMatched: {matched}/{len(memories)}")
    print(f"Total training examples: {len(training_data)}")
    print(f"  MemRL format: {len(training_data)//2}")
    print(f"  Raw format: {len(training_data)//2}")
    print(f"Copy distribution:")
    for copies, count in sorted(copy_counts.items()):
        print(f"  {copies} copies: {count} tasks")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python scripts/build_training_data_v4.py <memory_bank.json> <alfworld_data_dir> <few_shot.json> <output.jsonl>")
        sys.exit(1)
    build_training_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

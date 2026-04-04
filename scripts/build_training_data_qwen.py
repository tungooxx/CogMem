"""Build training data from Qwen2.5:3b's own meta-reflections + Q-weights.

No expert trajectories. Uses the model's own reflections from Phase 1.
Q-weighted: harder tasks (lower Q) get up to 10 copies.
Dual format: MemRL prompt + raw prompt.

Usage:
    python scripts/build_training_data_qwen.py results/memory_bank_qwen.json results/training_qwen.jsonl
"""

import json
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


def q_to_copies(q_value: float, max_copies: int = 10) -> int:
    """Lower Q = more copies. Hardest tasks get 10x training."""
    normalized = abs(q_value)
    copies = 1 + int(normalized * (max_copies - 1) / 0.9)
    return min(copies, max_copies)


def build_training_data(memory_bank_path: str, output_path: str) -> str:
    with open(memory_bank_path, encoding="utf-8") as f:
        memories = json.load(f)

    training_data = []
    skipped = 0

    for mem in memories:
        script = mem.get("script", "")
        task_desc = mem.get("task_description", "")

        # Skip if no useful script
        if not script or not task_desc:
            skipped += 1
            continue

        # Skip if script is garbage (no Thought/Action pattern)
        if "Thought:" not in script and "Action:" not in script:
            skipped += 1
            continue

        copies = q_to_copies(mem.get("q_value", 0.0))

        # Format 1: MemRL format (system + task)
        memrl_example = {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Now, it's your turn to solve a new task.\n{task_desc}"},
            {"role": "assistant", "content": script},
        ]}

        # Format 2: Raw format (just task)
        raw_example = {"messages": [
            {"role": "user", "content": task_desc},
            {"role": "assistant", "content": script},
        ]}

        for _ in range(copies):
            training_data.append(memrl_example)
            training_data.append(raw_example)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    used = len(memories) - skipped
    print(f"Memories: {len(memories)} total, {used} used, {skipped} skipped (no Thought/Action)")
    print(f"Training examples: {len(training_data)} (MemRL: {len(training_data)//2}, Raw: {len(training_data)//2})")
    print(f"Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/build_training_data_qwen.py <memory_bank.json> <output.jsonl>")
        sys.exit(1)
    build_training_data(sys.argv[1], sys.argv[2])

"""Build training data from BigCodeBench episodes with Q-weights.

Uses the model's own reasoning (Thought + Code) weighted by Q-values.
Dual format: system-prompt format + raw format.

Usage:
    python scripts/build_training_data_bigcode.py results/memory_bank_bigcode.json results/training_bigcode.jsonl
"""

import json
import sys
from pathlib import Path

from cogmem.benchmarks.bigcodebench.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_RAW


def q_to_copies(q_value: float, max_copies: int = 10) -> int:
    """Lower Q = more copies for failed tasks. Higher Q = more copies for passed tasks.

    For BigCodeBench (unlike ALFWorld), we weight SUCCESSFUL episodes more heavily,
    since the model needs to learn correct code patterns.
    Failed episodes still get included (1-3 copies) to learn what to avoid.
    """
    if q_value > 0:
        # Success: more copies for harder successes (lower positive Q)
        return max(3, max_copies)  # All successes get high weight
    else:
        # Failure: fewer copies, proportional to how badly it failed
        normalized = min(abs(q_value), 1.0)
        copies = 1 + int(normalized * 2)  # 1-3 copies for failures
        return min(copies, 3)


def build_training_data(memory_bank_path: str, output_path: str) -> str:
    with open(memory_bank_path, encoding="utf-8") as f:
        episodes = json.load(f)

    training_data = []
    skipped = 0
    successes = 0

    for ep in episodes:
        script = ep.get("script", "")
        task_desc = ep.get("task_description", "")

        # Skip if no useful response
        if not script or not task_desc:
            skipped += 1
            continue

        # Skip if no code was generated
        if "Code:" not in script and "```" not in script and "def " not in script:
            skipped += 1
            continue

        if ep.get("success"):
            successes += 1
        copies = q_to_copies(ep.get("q_value", 0.0))

        # Format 1: With system prompt (matches inference format)
        sys_example = {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_desc},
            {"role": "assistant", "content": script},
        ]}

        # Format 2: Raw format (task + response only)
        raw_example = {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT_RAW},
            {"role": "user", "content": task_desc},
            {"role": "assistant", "content": script},
        ]}

        for _ in range(copies):
            training_data.append(sys_example)
            training_data.append(raw_example)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    used = len(episodes) - skipped
    print(f"Episodes: {len(episodes)} total, {used} used, {skipped} skipped")
    print(f"Successes: {successes}, Failures: {used - successes}")
    print(f"Training examples: {len(training_data)} (System: {len(training_data)//2}, Raw: {len(training_data)//2})")
    print(f"Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/build_training_data_bigcode.py <memory_bank.json> <output.jsonl>")
        sys.exit(1)
    build_training_data(sys.argv[1], sys.argv[2])

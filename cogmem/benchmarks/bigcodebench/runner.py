"""Run Qwen2.5:3b on BigCodeBench tasks and collect episodes for CogMem.

This is the Phase 1 collection engine: the model attempts tasks, we record
success/failure + the model's reasoning as episodes in the memory bank.
"""

import json
import time
from pathlib import Path

from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution
from cogmem.benchmarks.bigcodebench.prompts import (
    SYSTEM_PROMPT,
    extract_code,
    format_messages,
)


def run_single_task(
    task: dict,
    llm_client,
    eval_mode: str = "subprocess",
    eval_timeout: int = 30,
    max_tokens: int = 2048,
    temperature: float = 0,
) -> dict:
    """Run the model on a single BigCodeBench task.

    Returns an episode dict compatible with CogMem memory bank:
        episode_id, task_id, task_description, script, success, q_value,
        generated_code, error, timestamp
    """
    task_id = task["task_id"]
    messages = format_messages(task, use_instruct=True)

    # Generate solution
    try:
        response = llm_client.chat(
            messages, max_tokens=max_tokens, temperature=temperature
        )
    except Exception as e:
        return _make_episode(task, response="", code="", passed=False, error=str(e))

    # Extract code from response
    code = extract_code(response, task)

    # Evaluate
    result = evaluate_solution(task, code, timeout=eval_timeout, mode=eval_mode)

    return _make_episode(
        task,
        response=response,
        code=code,
        passed=result["passed"],
        error=result.get("error"),
    )


def _make_episode(
    task: dict,
    response: str,
    code: str,
    passed: bool,
    error: str | None,
) -> dict:
    """Build a CogMem episode from a BigCodeBench task attempt."""
    return {
        "episode_id": f"bigcode_{task['task_id'].replace('/', '_')}_{int(time.time())}",
        "task_id": task["task_id"],
        "task_type": "bigcodebench",
        "task_description": task.get("instruct_prompt", task.get("complete_prompt", "")),
        "script": response,  # Full model output (reasoning + code)
        "generated_code": code,  # Extracted code only
        "success": passed,
        "q_value": 1.0 if passed else -1.0,  # Initial binary Q-value
        "error": error,
        "entry_point": task.get("entry_point", ""),
        "timestamp": time.time(),
    }


def run_batch(
    tasks: list[dict],
    llm_client,
    output_path: str | None = None,
    checkpoint_path: str | None = None,
    eval_mode: str = "subprocess",
    eval_timeout: int = 30,
    max_tokens: int = 2048,
    temperature: float = 0,
    progress_callback=None,
) -> list[dict]:
    """Run model on a batch of tasks with checkpoint/resume support.

    Args:
        tasks: List of BigCodeBench task dicts.
        llm_client: LLMClient instance (Ollama, Groq, etc.)
        output_path: Path to save episodes JSONL (append-per-task, crash-safe).
        checkpoint_path: Path to checkpoint file for resume support.
        eval_mode: "subprocess" or "docker".
        eval_timeout: Seconds per test.
        max_tokens: Max tokens for generation.
        temperature: Sampling temperature.
        progress_callback: Optional callable(i, total, episode) for progress.

    Returns:
        List of episode dicts.
    """
    # Resume from checkpoint
    completed_ids = set()
    episodes = []
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ep = json.loads(line)
                    completed_ids.add(ep["task_id"])
                    episodes.append(ep)
        print(f"Resumed: {len(completed_ids)} tasks already completed")

    # Ensure output directory exists
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    remaining = [t for t in tasks if t["task_id"] not in completed_ids]
    total = len(tasks)
    done = len(completed_ids)

    for i, task in enumerate(remaining):
        episode = run_single_task(
            task,
            llm_client,
            eval_mode=eval_mode,
            eval_timeout=eval_timeout,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        episodes.append(episode)
        done += 1

        # Append to output file (crash-safe)
        if output_path:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(episode, ensure_ascii=False) + "\n")

        # Also write checkpoint (same as output for simplicity)
        if checkpoint_path and checkpoint_path != output_path:
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(episode, ensure_ascii=False) + "\n")

        status = "PASS" if episode["success"] else "FAIL"
        print(f"[{done}/{total}] {task['task_id']} — {status}")

        if progress_callback:
            progress_callback(done, total, episode)

    return episodes


def episodes_to_memory_bank(episodes: list[dict], output_path: str) -> str:
    """Convert episodes list to CogMem memory bank JSON format."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
    total = len(episodes)
    passed = sum(1 for ep in episodes if ep["success"])
    rate = passed / total if total > 0 else 0.0
    print(f"Memory bank: {total} episodes, {passed} passed ({rate:.1%})")
    return output_path


def load_episodes(path: str) -> list[dict]:
    """Load episodes from JSONL checkpoint file."""
    episodes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes

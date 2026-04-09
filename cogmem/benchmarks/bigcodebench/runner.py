"""Run models on BigCodeBench tasks and collect episodes for CogMem.

Phase 1 collection engine: the model attempts tasks (with retry),
we record success/failure + trajectory as episodes in the memory bank.

Episodes include:
- trajectory: list of attempts with code + test result
- final_code: the successful code (if any)
"""

import json
import time
from pathlib import Path

from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution
from cogmem.benchmarks.bigcodebench.prompts import extract_code, format_messages


# -------------------------------------------------------------------------
# Single task execution with retry
# -------------------------------------------------------------------------

def run_single_task(
    task: dict,
    llm_client,
    eval_mode: str = "subprocess",
    eval_timeout: int = 30,
    max_tokens: int = 2048,
    temperature: float = 0,
    max_attempts: int = 1,
) -> dict:
    """Run the model on a single BigCodeBench task with optional retry.

    When max_attempts > 1, failed attempts get error feedback in the
    retry prompt. This builds a trajectory of attempts.

    Returns an episode dict with trajectory and domain fields.
    """
    trajectory = []

    for attempt in range(1, max_attempts + 1):
        # Build prompt (with error feedback for retries)
        if attempt == 1:
            messages = format_messages(task, use_instruct=True)
        else:
            prev_error = trajectory[-1].get("error", "Unknown error")
            retry_prompt = (
                f"Your previous attempt failed with this error:\n"
                f"{prev_error[:500]}\n\n"
                f"Please fix the code and try again."
            )
            messages = format_messages(task, use_instruct=True)
            messages.append({"role": "user", "content": retry_prompt})

        # Generate
        try:
            response = llm_client.chat(
                messages, max_tokens=max_tokens, temperature=temperature
            )
        except Exception as e:
            trajectory.append({
                "attempt": attempt,
                "code": "",
                "response": "",
                "test_result": "ERROR",
                "error": str(e),
            })
            continue

        code = extract_code(response, task)
        result = evaluate_solution(task, code, timeout=eval_timeout, mode=eval_mode)
        passed = result["passed"]

        trajectory.append({
            "attempt": attempt,
            "code": code,
            "response": response,
            "test_result": "PASS" if passed else "FAIL",
            "error": result.get("error") if not passed else None,
        })

        if passed:
            break

    # Build episode
    success = any(t["test_result"] == "PASS" for t in trajectory)
    final_code = None
    for t in trajectory:
        if t["test_result"] == "PASS":
            final_code = t["code"]
            break

    return _make_episode(
        task,
        trajectory=trajectory,
        success=success,
        final_code=final_code,
    )


def _make_episode(
    task: dict,
    trajectory: list[dict],
    success: bool,
    final_code: str | None,
) -> dict:
    """Build a CogMem episode from a BigCodeBench task attempt."""
    # Use last response as "script" for backward compat
    last_response = ""
    last_code = ""
    for t in reversed(trajectory):
        if t.get("response"):
            last_response = t["response"]
            last_code = t["code"]
            break

    return {
        "episode_id": f"bigcode_{task['task_id'].replace('/', '_')}_{int(time.time())}",
        "task_id": task["task_id"],
        "task_type": "bigcodebench",
        "task_description": task.get("instruct_prompt", task.get("complete_prompt", "")),
        "script": last_response,
        "generated_code": last_code,
        "final_code": final_code,
        "success": success,
        "q_value": 0.5,  # starts neutral, updated by retrieval feedback in collect_sequential
        "error": trajectory[-1].get("error") if trajectory else None,
        "entry_point": task.get("entry_point", ""),
        "trajectory": trajectory,
        "num_attempts": len(trajectory),
        "timestamp": time.time(),
    }


# -------------------------------------------------------------------------
# Batch execution
# -------------------------------------------------------------------------

def run_batch(
    tasks: list[dict],
    llm_client,
    output_path: str | None = None,
    checkpoint_path: str | None = None,
    eval_mode: str = "subprocess",
    eval_timeout: int = 30,
    max_tokens: int = 2048,
    temperature: float = 0,
    max_attempts: int = 1,
    progress_callback=None,
) -> list[dict]:
    """Run model on a batch of tasks with checkpoint/resume support.

    Args:
        tasks: List of BigCodeBench task dicts.
        llm_client: LLMClient instance.
        output_path: Path to save episodes JSONL.
        checkpoint_path: Path to checkpoint file for resume.
        eval_mode: "subprocess" or "docker".
        eval_timeout: Seconds per test.
        max_tokens: Max tokens for generation.
        temperature: Sampling temperature.
        max_attempts: Number of retry attempts per task.
        progress_callback: Optional callable(i, total, episode).

    Returns:
        List of episode dicts.
    """
    completed_ids = set()
    episodes = []
    if checkpoint_path and Path(checkpoint_path).exists():
        episodes = load_episodes(checkpoint_path)
        completed_ids = {ep["task_id"] for ep in episodes}
        print(f"Resumed: {len(completed_ids)} tasks already completed")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path and checkpoint_path != output_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    remaining = [t for t in tasks if t["task_id"] not in completed_ids]
    total = len(tasks)
    done = len(completed_ids)

    for task in remaining:
        episode = run_single_task(
            task,
            llm_client,
            eval_mode=eval_mode,
            eval_timeout=eval_timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
        )
        episodes.append(episode)
        done += 1

        if output_path:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(episode, ensure_ascii=False) + "\n")

        if checkpoint_path and checkpoint_path != output_path:
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(episode, ensure_ascii=False) + "\n")

        status = "PASS" if episode["success"] else "FAIL"
        attempts = episode["num_attempts"]
        print(f"[{done}/{total}] {task['task_id']} — {status} "
              f"(attempts={attempts})")

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
    """Load episodes from JSONL checkpoint file.

    Tolerates truncated final lines (e.g. from a crash mid-write).
    """
    episodes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed line in {path}")
                continue
    return episodes

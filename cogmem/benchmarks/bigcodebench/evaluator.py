"""Evaluate generated code against BigCodeBench test cases.

Two evaluation modes:
1. subprocess: Run tests in a subprocess (simple, no Docker)
2. docker: Run tests in Docker container (safe, isolated)

For Paperspace, subprocess mode is fine since we're in an isolated VM.
"""

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def evaluate_solution(
    task: dict,
    generated_code: str,
    timeout: int = 30,
    mode: str = "subprocess",
) -> dict:
    """Evaluate a single solution against its test cases.

    Args:
        task: BigCodeBench task dict with 'test', 'complete_prompt', 'entry_point'.
        generated_code: The model's generated code.
        timeout: Max seconds for test execution.
        mode: "subprocess" or "docker".

    Returns:
        {"passed": bool, "error": str | None, "output": str}
    """
    if mode == "docker":
        return _evaluate_docker(task, generated_code, timeout)
    return _evaluate_subprocess(task, generated_code, timeout)


def _build_test_script(task: dict, generated_code: str) -> str:
    """Build a complete test script combining imports, solution, and tests."""
    # The complete_prompt includes imports + function signature
    # The generated code is the function body/implementation
    # The test field contains unit tests

    complete_prompt = task.get("complete_prompt", "")
    test_code = task.get("test", "")
    entry_point = task.get("entry_point", "")

    # Build the full script:
    # 1. The complete prompt (has imports + function signature)
    # 2. The generated implementation
    # 3. The test code
    # 4. Run the test
    script = f"""{complete_prompt}
{generated_code}

{test_code}

# Run the test
import traceback
try:
    check({entry_point})
    print("ALL_TESTS_PASSED")
except Exception as e:
    traceback.print_exc()
    print(f"TEST_FAILED: {{e}}")
"""
    return script


def _evaluate_subprocess(task: dict, generated_code: str, timeout: int) -> dict:
    """Run tests in a subprocess."""
    script = _build_test_script(task, generated_code)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_safe_env(),
        )
        output = result.stdout + result.stderr
        passed = "ALL_TESTS_PASSED" in result.stdout and result.returncode == 0
        return {
            "passed": passed,
            "error": result.stderr if not passed else None,
            "output": output[:2000],  # Truncate long output
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"Timeout after {timeout}s", "output": ""}
    except Exception as e:
        return {"passed": False, "error": str(e), "output": ""}
    finally:
        Path(script_path).unlink(missing_ok=True)


def _evaluate_docker(task: dict, generated_code: str, timeout: int) -> dict:
    """Run tests in a Docker container for isolation."""
    script = _build_test_script(task, generated_code)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network=none",
                f"--memory=512m",
                f"--cpus=1",
                "-v", f"{script_path}:/tmp/test.py:ro",
                "python:3.10-slim",
                "python3", "/tmp/test.py",
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 10,  # Extra time for Docker overhead
        )
        output = result.stdout + result.stderr
        passed = "ALL_TESTS_PASSED" in result.stdout and result.returncode == 0
        return {
            "passed": passed,
            "error": result.stderr if not passed else None,
            "output": output[:2000],
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"Timeout after {timeout}s", "output": ""}
    except Exception as e:
        return {"passed": False, "error": str(e), "output": ""}
    finally:
        Path(script_path).unlink(missing_ok=True)


def _safe_env():
    """Build a safe environment for subprocess execution."""
    import os
    env = os.environ.copy()
    # Prevent the script from accessing network or sensitive env vars
    for key in ["TOGETHER_API_KEY", "GROQ_API_KEY", "HF_TOKEN", "OPENAI_API_KEY"]:
        env.pop(key, None)
    return env


def evaluate_batch(
    tasks: list[dict],
    solutions: dict[str, str],
    timeout: int = 30,
    mode: str = "subprocess",
) -> dict:
    """Evaluate a batch of solutions.

    Args:
        tasks: List of BigCodeBench task dicts.
        solutions: Dict mapping task_id -> generated_code.
        timeout: Max seconds per test.
        mode: "subprocess" or "docker".

    Returns:
        {"results": {task_id: result_dict}, "pass_rate": float, "total": int, "passed": int}
    """
    results = {}
    passed_count = 0

    for task in tasks:
        task_id = task["task_id"]
        code = solutions.get(task_id, "")
        if not code:
            results[task_id] = {"passed": False, "error": "No solution generated", "output": ""}
            continue

        result = evaluate_solution(task, code, timeout=timeout, mode=mode)
        results[task_id] = result
        if result["passed"]:
            passed_count += 1

    total = len(tasks)
    return {
        "results": results,
        "pass_rate": passed_count / total if total > 0 else 0,
        "total": total,
        "passed": passed_count,
    }

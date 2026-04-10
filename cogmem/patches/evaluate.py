"""Evaluation for cognitive patches.

Three modes:
- Cold: base model only (floor)
- Patched: base + dynamic patches per task (full system)
- Best-of-N: patched + N candidates per task (ceiling)
"""

import torch

from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution
from cogmem.benchmarks.bigcodebench.prompts import SYSTEM_PROMPT, extract_code
from cogmem.patches.compose import PatchedModel
from cogmem.patches.wake import generate_with_model


def evaluate_cold(
    base_model,
    tokenizer,
    tasks: list[dict],
    eval_timeout: int = 30,
) -> float:
    """Base model only, no patches. This is the floor."""
    passed = 0

    for i, task in enumerate(tasks):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.get("instruct_prompt", task.get("complete_prompt", ""))},
        ]

        try:
            response = generate_with_model(base_model, tokenizer, messages, temperature=0)
            code = extract_code(response)
            result = evaluate_solution(task, code, timeout=eval_timeout, mode="subprocess")
            if result["passed"]:
                passed += 1
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            print(f"    Cold eval: {passed}/{i+1} ({passed/(i+1):.1%})")

    rate = passed / max(len(tasks), 1)
    print(f"    Cold eval final: {passed}/{len(tasks)} ({rate:.1%})")
    return rate


def evaluate_patched(
    base_model,
    tokenizer,
    tasks: list[dict],
    patch_bank,
    embedder,
    eval_timeout: int = 30,
) -> float:
    """Base model + dynamically composed patches per task."""
    passed = 0

    for i, task in enumerate(tasks):
        prompt = task.get("instruct_prompt", task.get("complete_prompt", ""))
        task_embedding = embedder.encode(prompt).tolist()

        # Get patches for this task
        active_patches = patch_bank.get_active_patches(task_embedding, top_k=5)
        for p in active_patches:
            patch_bank.load_weights(p)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            with PatchedModel(base_model, active_patches):
                response = generate_with_model(base_model, tokenizer, messages, temperature=0)
            code = extract_code(response)
            result = evaluate_solution(task, code, timeout=eval_timeout, mode="subprocess")
            if result["passed"]:
                passed += 1
        except Exception:
            pass
        finally:
            for p in active_patches:
                p.unload_weights()

        if (i + 1) % 50 == 0:
            print(f"    Patched eval: {passed}/{i+1} ({passed/(i+1):.1%})")

    rate = passed / max(len(tasks), 1)
    print(f"    Patched eval final: {passed}/{len(tasks)} ({rate:.1%})")
    return rate


def evaluate_best_of_n(
    base_model,
    tokenizer,
    tasks: list[dict],
    patch_bank,
    embedder,
    n: int = 8,
    temperature: float = 0.8,
    eval_timeout: int = 30,
) -> float:
    """Patched model + N candidates per task. This is the ceiling."""
    passed = 0

    for i, task in enumerate(tasks):
        prompt = task.get("instruct_prompt", task.get("complete_prompt", ""))
        task_embedding = embedder.encode(prompt).tolist()

        active_patches = patch_bank.get_active_patches(task_embedding, top_k=5)
        for p in active_patches:
            patch_bank.load_weights(p)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        task_passed = False
        try:
            with PatchedModel(base_model, active_patches):
                for _ in range(n):
                    try:
                        response = generate_with_model(
                            base_model, tokenizer, messages, temperature=temperature
                        )
                        code = extract_code(response)
                        result = evaluate_solution(task, code, timeout=eval_timeout, mode="subprocess")
                        if result["passed"]:
                            task_passed = True
                            break
                    except Exception:
                        continue
        finally:
            for p in active_patches:
                p.unload_weights()

        if task_passed:
            passed += 1

        if (i + 1) % 50 == 0:
            print(f"    Best-of-{n} eval: {passed}/{i+1} ({passed/(i+1):.1%})")

    rate = passed / max(len(tasks), 1)
    print(f"    Best-of-{n} eval final: {passed}/{len(tasks)} ({rate:.1%})")
    return rate

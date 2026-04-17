"""Wake mode for the experimental patch-consolidation path.

Process tasks sequentially:
1. RETRIEVE relevant cluster memories
2. LOAD distilled patch artifacts and compose them onto the base model
3. GENERATE N candidates with patched model
4. TEST each candidate
5. LEARN: record a new episode from pass/fail contrast
6. UPDATE memory-object usage and utility signals
"""

import time
from concurrent.futures import Future
from difflib import SequenceMatcher

import torch

from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution
from cogmem.benchmarks.bigcodebench.prompts import SYSTEM_PROMPT, extract_code
from cogmem.patches.compose import PatchedModel
from cogmem.patches.memory_bank import (
    ClusterMemoryBank,
    DEFAULT_ACTIVE_MEMORY_TOP_K,
    DEFAULT_PATCH_SCALE,
)


def _tokenize_messages(model, tokenizer, messages):
    """Tokenize a chat prompt onto the model device."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)


def generate_with_model(model, tokenizer, messages, temperature=0.8, max_tokens=2048):
    """Generate a response using the model directly (not Ollama)."""
    inputs = _tokenize_messages(model, tokenizer, messages)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def generate_many_with_model(
    model,
    tokenizer,
    messages,
    n_candidates: int,
    temperature: float = 0.8,
    max_tokens: int = 2048,
):
    """Generate multiple sampled responses in one model.generate call when possible."""
    if n_candidates <= 1:
        return [generate_with_model(model, tokenizer, messages, temperature=temperature, max_tokens=max_tokens)]

    if temperature <= 0:
        response = generate_with_model(model, tokenizer, messages, temperature=temperature, max_tokens=max_tokens)
        return [response for _ in range(n_candidates)]

    inputs = _tokenize_messages(model, tokenizer, messages)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=n_candidates,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs.input_ids.shape[1]
    responses = []
    for row in outputs:
        gen_ids = row[prompt_len:]
        responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return responses


def find_best_contrast_pair(passes, fails, min_sim=0.3, max_sim=0.95):
    """Find the pass/fail pair with highest similarity (focused contrast)."""
    best_pair = None
    best_sim = 0

    for p in passes:
        for f in fails:
            sim = SequenceMatcher(None, p["code"], f["code"]).ratio()
            if min_sim < sim < max_sim and sim > best_sim:
                best_pair = {"pass": p, "fail": f}
                best_sim = sim

    # Fallback: any pair
    if best_pair is None and passes and fails:
        best_pair = {"pass": passes[0], "fail": fails[0]}
        best_sim = SequenceMatcher(None, passes[0]["code"], fails[0]["code"]).ratio()

    return best_pair, best_sim


def run_wake_cycle(
    tasks: list[dict],
    base_model,
    tokenizer,
    memory_bank: ClusterMemoryBank,
    embedder,
    n_candidates: int = 8,
    temperature: float = 0.8,
    eval_timeout: int = 30,
    save_every: int = 50,
    rebuild_every: int = 10,
    async_build: bool = False,
) -> dict:
    """Process tasks sequentially, creating patches from experience.

    Args:
        tasks: List of BigCodeBench task dicts.
        base_model: Frozen 4-bit base model on GPU.
        tokenizer: Tokenizer.
        memory_bank: ClusterMemoryBank to read from and write to.
        embedder: SentenceTransformer for task embeddings.
        n_candidates: Candidates per task for best-of-N.
        temperature: Sampling temperature.
        eval_timeout: Test timeout in seconds.
        save_every: Save bank every N tasks.
        rebuild_every: Rebuild cluster memories every N newly recorded episodes
            when value > 0. Units are episodes/tasks encountered in this wake loop.
        async_build: If True, schedule expensive ``build_memories`` refreshes on
            a background worker so the wake loop keeps moving. When False, rebuilds
            happen synchronously and can noticeably slow the loop.

    Returns:
        Stats dict.
    """
    total = len(tasks)
    episodes_created = 0
    tasks_passed = 0
    start_time = time.time()
    pending_build: Future | None = None

    def maybe_raise_pending_build() -> None:
        nonlocal pending_build
        if pending_build is not None and pending_build.done():
            pending_build.result()
            pending_build = None

    for i, task in enumerate(tasks):
        maybe_raise_pending_build()
        task_id = task["task_id"]
        prompt = task.get("instruct_prompt", task.get("complete_prompt", ""))

        # 1. Embed task
        task_embedding = embedder.encode(prompt).tolist()

        # 2. Get relevant memories and their distilled patches
        active_memories, active_patches = memory_bank.get_active_patches(
            task_embedding,
            prompt,
            top_k=DEFAULT_ACTIVE_MEMORY_TOP_K,
            return_memories=True,
        )

        # 3. Generate N candidates with patched model
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        candidates = []
        try:
            with PatchedModel(base_model, active_patches, scaling_factor=DEFAULT_PATCH_SCALE):
                responses = generate_many_with_model(
                    base_model,
                    tokenizer,
                    messages,
                    n_candidates=n_candidates,
                    temperature=temperature,
                )
                for response in responses:
                    try:
                        code = extract_code(response)
                        if code and len(code.strip()) > 20:
                            result = evaluate_solution(task, code, timeout=eval_timeout, mode="subprocess")
                            candidates.append({
                                "code": code,
                                "passed": result["passed"],
                                "response": response,
                            })
                    except Exception:
                        continue
        finally:
            for patch in active_patches:
                patch.unload_weights()

        passes = [c for c in candidates if c["passed"]]
        fails = [c for c in candidates if not c["passed"]]
        task_succeeded = len(passes) > 0

        if task_succeeded:
            tasks_passed += 1

        # 5. Record episode from contrast
        if passes and fails:
            best_pair, best_sim = find_best_contrast_pair(passes, fails)
            if best_pair:
                memory_bank.record_episode(
                    task_id=task_id,
                    prompt=prompt,
                    task_embedding=task_embedding,
                    failed_code=best_pair["fail"]["code"],
                    passed_code=best_pair["pass"]["code"],
                    pass_fail_similarity=best_sim,
                )
                episodes_created += 1

                if rebuild_every > 0 and episodes_created % rebuild_every == 0:
                    if async_build:
                        if pending_build is None:
                            pending_build = memory_bank.schedule_build_memories(
                                base_model,
                                tokenizer,
                            )
                    else:
                        memory_bank.build_memories(base_model, tokenizer)

        # 6. Update utility signals of active memories
        for memory in active_memories:
            memory_bank.update_memory_utility(
                memory.memory_id,
                task_succeeded,
                persist=False,
            )

        # 7. Progress
        if (i + 1) % 10 == 0 or i < 5:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 3600 if elapsed > 0 else 0
            eta = (total - i - 1) / (rate / 60) if rate > 0 else 0
            stats = memory_bank.stats()
            n_pass = len(passes)
            n_fail = len(fails)
            print(
                f"  [{i+1}/{total}] {task_id}: "
                f"{n_pass}P/{n_fail}F | "
                f"memories={stats.get('memories', 0)} | "
                f"pass_rate={tasks_passed}/{i+1} ({tasks_passed/(i+1):.1%}) | "
                f"active={len(active_memories)} | "
                f"ETA={eta:.0f}m"
            )

        # Save periodically
        if (i + 1) % save_every == 0:
            memory_bank.save()

    if pending_build is not None:
        pending_build.result()
        pending_build = None
    else:
        memory_bank.build_memories(base_model, tokenizer)
    memory_bank.save()

    return {
        "total_tasks": total,
        "tasks_passed": tasks_passed,
        "pass_rate": tasks_passed / max(total, 1),
        "episodes_created": episodes_created,
        "total_memories": len(memory_bank.memories),
    }

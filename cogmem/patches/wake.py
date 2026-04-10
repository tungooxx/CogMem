"""Wake mode — experience loop for cognitive patch creation.

Process tasks sequentially:
1. RETRIEVE relevant patches from bank
2. COMPOSE patches onto base model
3. GENERATE N candidates with patched model
4. TEST each candidate
5. LEARN: create new patch from pass/fail contrast
6. UPDATE Q-values of active patches
"""

import time
from difflib import SequenceMatcher

import torch

from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution
from cogmem.benchmarks.bigcodebench.prompts import SYSTEM_PROMPT, extract_code
from cogmem.patches.bank import PatchBank
from cogmem.patches.compose import PatchedModel
from cogmem.patches.create import create_patch_from_contrast


def generate_with_model(model, tokenizer, messages, temperature=0.8, max_tokens=2048):
    """Generate a response using the model directly (not Ollama)."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=2048).to(model.device)

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
    patch_bank: PatchBank,
    embedder,
    n_candidates: int = 8,
    temperature: float = 0.8,
    eval_timeout: int = 30,
    save_every: int = 50,
) -> dict:
    """Process tasks sequentially, creating patches from experience.

    Args:
        tasks: List of BigCodeBench task dicts.
        base_model: Frozen 4-bit base model on GPU.
        tokenizer: Tokenizer.
        patch_bank: PatchBank to read from and write to.
        embedder: SentenceTransformer for task embeddings.
        n_candidates: Candidates per task for best-of-N.
        temperature: Sampling temperature.
        eval_timeout: Test timeout in seconds.
        save_every: Save bank every N tasks.

    Returns:
        Stats dict.
    """
    total = len(tasks)
    patches_created = 0
    tasks_passed = 0
    start_time = time.time()

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        prompt = task.get("instruct_prompt", task.get("complete_prompt", ""))

        # 1. Embed task
        task_embedding = embedder.encode(prompt).tolist()

        # 2. Get relevant patches
        active_patches = patch_bank.get_active_patches(task_embedding, top_k=5)

        # Load weights for active patches (lazy loading)
        for p in active_patches:
            patch_bank.load_weights(p)

        # 3. Generate N candidates with patched model
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        candidates = []
        try:
            with PatchedModel(base_model, active_patches):
                for _ in range(n_candidates):
                    try:
                        response = generate_with_model(
                            base_model, tokenizer, messages,
                            temperature=temperature,
                        )
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

        # 5. Create patch from contrast
        if passes and fails:
            best_pair, _ = find_best_contrast_pair(passes, fails)
            if best_pair:
                new_patch = create_patch_from_contrast(
                    base_model, tokenizer,
                    prompt,
                    best_pair["fail"]["code"],
                    best_pair["pass"]["code"],
                    patch_id=f"patch_{task_id.replace('/', '_')}_{int(time.time())}",
                )
                new_patch.embedding = task_embedding
                new_patch.source_task_id = task_id
                patch_bank.add(new_patch)
                patches_created += 1

        # 6. Update Q-values of active patches
        for patch in active_patches:
            patch_bank.update_q(patch.patch_id, task_succeeded)

        # 7. Progress
        if (i + 1) % 10 == 0 or i < 5:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 3600 if elapsed > 0 else 0
            eta = (total - i - 1) / (rate / 60) if rate > 0 else 0
            stats = patch_bank.stats()
            n_pass = len(passes)
            n_fail = len(fails)
            print(
                f"  [{i+1}/{total}] {task_id}: "
                f"{n_pass}P/{n_fail}F | "
                f"patches={stats.get('total', 0)} | "
                f"pass_rate={tasks_passed}/{i+1} ({tasks_passed/(i+1):.1%}) | "
                f"active={len(active_patches)} | "
                f"ETA={eta:.0f}m"
            )

        # Save periodically
        if (i + 1) % save_every == 0:
            patch_bank.save()

    patch_bank.save()

    return {
        "total_tasks": total,
        "tasks_passed": tasks_passed,
        "pass_rate": tasks_passed / max(total, 1),
        "patches_created": patches_created,
        "total_patches": len(patch_bank.patches),
    }

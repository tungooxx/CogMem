"""Generate high-quality DPO training pairs from BigCodeBench attempts.

Three sources of pairs, all with high similarity (focused contrasts):
1. Best-of-N: generate 8 candidates, select highest-similarity pass/fail pair
2. Mutation: take passing code, mutate slightly, test if it breaks
3. Retry: from existing memory bank trajectories (first failure vs final success)

Usage:
    python scripts/generate_dpo_pairs.py --tasks tasks.jsonl --output pairs.jsonl
    python scripts/generate_dpo_pairs.py --tasks tasks.jsonl --bank bank.json --source all
    python scripts/generate_dpo_pairs.py --resume --output pairs.jsonl
"""

import argparse
import ast
import json
import os
import random
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution
from cogmem.benchmarks.bigcodebench.prompts import (
    SYSTEM_PROMPT,
    extract_code,
    format_messages,
)

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:3b"
N_CANDIDATES = 8
CANDIDATE_TEMPERATURE = 0.8
SIMILARITY_MIN = 0.3
SIMILARITY_MAX = 0.95
SIMILARITY_TARGET_MIN = 0.5
SIMILARITY_TARGET_MAX = 0.9
MIN_CODE_LENGTH = 50


# ════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════

def compute_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def make_pair(task, chosen, rejected, source, similarity):
    prompt = SYSTEM_PROMPT + "\n\n" + task.get(
        "instruct_prompt", task.get("complete_prompt", "")
    )
    filtered = similarity < SIMILARITY_MIN or similarity > SIMILARITY_MAX
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source,
        "task_id": task.get("task_id", ""),
        "similarity": round(similarity, 3),
        "filtered": filtered,
    }


def load_checkpoint(output_path):
    """Load already-processed task IDs from output file."""
    done = {}
    if Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        pair = json.loads(line)
                        tid = pair.get("task_id", "")
                        src = pair.get("source", "")
                        done.setdefault(src, set()).add(tid)
                    except json.JSONDecodeError:
                        continue
    return done


def save_pair(pair, output_path):
    """Append a single pair to the output file."""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")


# ════════════════════════════════════════════
# SOURCE 1: Best-of-N with similarity filtering
# ════════════════════════════════════════════

def generate_best_of_n_pairs(tasks, output_path, model, resume_done):
    """Generate N candidates per task, find highest-similarity pass/fail pair."""
    from openai import OpenAI

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    done_ids = resume_done.get("best_of_n", set())

    remaining = [t for t in tasks if t["task_id"] not in done_ids]
    total = len(tasks)
    processed = len(done_ids)
    pairs_found = 0
    start_time = time.time()

    print(f"\n[Best-of-N] {len(remaining)} tasks remaining, {N_CANDIDATES} candidates each")

    for i, task in enumerate(remaining):
        task_id = task["task_id"]
        messages = format_messages(task, use_instruct=True)

        # Generate N candidates
        passes = []
        fails = []
        for _ in range(N_CANDIDATES):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=2048,
                    temperature=CANDIDATE_TEMPERATURE,
                )
                response = resp.choices[0].message.content
                code = extract_code(response)

                if code and len(code.strip()) > MIN_CODE_LENGTH:
                    result = evaluate_solution(task, code, timeout=30, mode="subprocess")
                    if result["passed"]:
                        passes.append(code)
                    else:
                        fails.append(code)
            except Exception:
                continue

        # Find best pair (highest similarity in target range)
        if passes and fails:
            best_pair = None
            best_sim = 0
            for p in passes:
                for f in fails:
                    sim = compute_similarity(p, f)
                    if SIMILARITY_TARGET_MIN < sim < SIMILARITY_TARGET_MAX and sim > best_sim:
                        best_pair = (p, f)
                        best_sim = sim

            # Fallback: best pair in any range
            if best_pair is None:
                for p in passes:
                    for f in fails:
                        sim = compute_similarity(p, f)
                        if sim > best_sim:
                            best_pair = (p, f)
                            best_sim = sim

            if best_pair:
                pair = make_pair(task, best_pair[0], best_pair[1], "best_of_n", best_sim)
                save_pair(pair, output_path)
                pairs_found += 1

        processed += 1
        if (i + 1) % 10 == 0 or i < 5:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 3600 if elapsed > 0 else 0
            eta = (len(remaining) - i - 1) / (rate / 60) if rate > 0 else 0
            p_rate = len(passes) / N_CANDIDATES if passes or fails else 0
            print(
                f"  [{processed}/{total}] {task_id}: "
                f"{len(passes)}P/{len(fails)}F | "
                f"pairs={pairs_found} | "
                f"rate={rate:.0f}/hr | ETA={eta:.0f}m"
            )

    print(f"  [Best-of-N] Done: {pairs_found} pairs from {processed} tasks")
    return pairs_found


# ════════════════════════════════════════════
# SOURCE 2: Programmatic mutation pairs
# ════════════════════════════════════════════

MUTATIONS = [
    "remove_import",
    "remove_parameter",
    "swap_method",
    "remove_try_except",
    "remove_string_op",
    "swap_axis",
    "remove_encoding",
]


def mutate_remove_import(code):
    """Remove one import line."""
    lines = code.split("\n")
    import_lines = [i for i, l in enumerate(lines) if l.strip().startswith(("import ", "from "))]
    if not import_lines:
        return None
    idx = random.choice(import_lines)
    new_lines = lines[:idx] + lines[idx + 1:]
    return "\n".join(new_lines)


def mutate_remove_parameter(code):
    """Remove one keyword parameter from a function call."""
    # Find patterns like func(a, key=value) → func(a)
    match = re.search(r",\s*\w+=[^,)]+", code)
    if not match:
        return None
    return code[:match.start()] + code[match.end():]


def mutate_swap_method(code):
    """Swap common method pairs."""
    swaps = [
        (".sort_values(", ".sort("),
        (".apply(", ".map("),
        (".read_csv(", ".read_table("),
        (".items()", ".keys()"),
        (".strip()", ".lstrip()"),
        ("json.loads(", "json.load("),
        ("pd.read_csv(", "pd.read("),
    ]
    for old, new in swaps:
        if old in code:
            return code.replace(old, new, 1)
    return None


def mutate_remove_try_except(code):
    """Remove try/except wrapper, keep the try body."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                # Replace try block with its body
                return ast.unparse(ast.Module(body=node.body, type_ignores=[]))
    except SyntaxError:
        pass
    # Fallback: regex
    if "try:" in code and "except" in code:
        lines = code.split("\n")
        new_lines = []
        skip = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("try:"):
                continue
            if stripped.startswith("except"):
                skip = True
                continue
            if skip and (not stripped or stripped.startswith(" ") or stripped.startswith("\t")):
                continue
            skip = False
            # Dedent lines that were in try block
            if line.startswith("    ") and not stripped.startswith("def "):
                new_lines.append(line[4:] if line.startswith("    ") else line)
            else:
                new_lines.append(line)
        result = "\n".join(new_lines)
        if result.strip() != code.strip():
            return result
    return None


def mutate_remove_string_op(code):
    """Remove .strip(), .lower(), .upper() calls."""
    for op in [".strip()", ".lower()", ".upper()", ".rstrip()", ".lstrip()"]:
        if op in code:
            return code.replace(op, "", 1)
    return None


def mutate_swap_axis(code):
    """Change axis=0 to axis=1 or vice versa."""
    if "axis=0" in code:
        return code.replace("axis=0", "axis=1", 1)
    if "axis=1" in code:
        return code.replace("axis=1", "axis=0", 1)
    return None


def mutate_remove_encoding(code):
    """Remove encoding='utf-8' from file operations."""
    for pattern in ["encoding='utf-8'", 'encoding="utf-8"', ", encoding='utf-8'", ', encoding="utf-8"']:
        if pattern in code:
            return code.replace(pattern, "", 1)
    return None


MUTATION_FNS = {
    "remove_import": mutate_remove_import,
    "remove_parameter": mutate_remove_parameter,
    "swap_method": mutate_swap_method,
    "remove_try_except": mutate_remove_try_except,
    "remove_string_op": mutate_remove_string_op,
    "swap_axis": mutate_swap_axis,
    "remove_encoding": mutate_remove_encoding,
}


def generate_mutation_pairs(tasks, episodes, output_path, resume_done):
    """Take passing solutions, mutate, test if mutations break them."""
    done_ids = resume_done.get("mutation", set())
    tasks_by_id = {t["task_id"]: t for t in tasks}

    # Get passing episodes with final_code
    passing = [
        ep for ep in episodes
        if ep.get("success") and ep.get("final_code") and ep["task_id"] in tasks_by_id
    ]
    remaining = [ep for ep in passing if ep["task_id"] not in done_ids]

    pairs_found = 0
    total = len(passing)
    processed = len(done_ids)

    print(f"\n[Mutation] {len(remaining)} passing episodes to mutate")

    rng = random.Random(42)

    for i, ep in enumerate(remaining):
        task_id = ep["task_id"]
        task = tasks_by_id[task_id]
        original_code = ep["final_code"]

        # Try 3-5 random mutations
        mutations_to_try = rng.sample(MUTATIONS, min(5, len(MUTATIONS)))

        for mut_name in mutations_to_try:
            mut_fn = MUTATION_FNS[mut_name]
            mutated = mut_fn(original_code)

            if mutated is None:
                continue
            if mutated.strip() == original_code.strip():
                continue
            if len(mutated.strip()) < MIN_CODE_LENGTH:
                continue

            # Test if mutation breaks the code
            result = evaluate_solution(task, mutated, timeout=30, mode="subprocess")
            if not result["passed"]:
                sim = compute_similarity(original_code, mutated)
                pair = make_pair(task, original_code, mutated, "mutation", sim)
                pair["mutation_type"] = mut_name
                save_pair(pair, output_path)
                pairs_found += 1

        processed += 1
        if (i + 1) % 10 == 0 or i < 5:
            print(f"  [{processed}/{total}] {task_id}: pairs so far={pairs_found}")

    print(f"  [Mutation] Done: {pairs_found} pairs from {processed} episodes")
    return pairs_found


# ════════════════════════════════════════════
# SOURCE 3: Existing retry pairs
# ════════════════════════════════════════════

def generate_retry_pairs(episodes, output_path, resume_done):
    """Extract pairs from within-episode retries."""
    done_ids = resume_done.get("retry", set())
    pairs_found = 0

    print(f"\n[Retry] Scanning episodes for retry pairs")

    for ep in episodes:
        task_id = ep.get("task_id", "")
        if task_id in done_ids:
            continue

        trajectory = ep.get("trajectory", [])
        if len(trajectory) < 2 or not ep.get("success"):
            continue

        chosen = ep.get("final_code", "")
        if not chosen or len(chosen.strip()) < MIN_CODE_LENGTH:
            continue

        # Pick FIRST failure (natural mistake)
        rejected = ""
        for step in trajectory:
            if step.get("test_result") != "PASS" and step.get("code", "").strip():
                rejected = step["code"]
                break

        if not rejected or len(rejected.strip()) < MIN_CODE_LENGTH:
            continue
        if chosen.strip() == rejected.strip():
            continue

        sim = compute_similarity(chosen, rejected)

        # Build task-like dict for make_pair
        task_proxy = {
            "task_id": task_id,
            "instruct_prompt": ep.get("task_description", ""),
        }
        pair = make_pair(task_proxy, chosen, rejected, "retry", sim)
        save_pair(pair, output_path)
        pairs_found += 1

    print(f"  [Retry] Done: {pairs_found} pairs")
    return pairs_found


# ════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════

def print_summary(output_path):
    """Print pair statistics."""
    pairs = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not pairs:
        print("\nNo pairs generated.")
        return

    by_source = {}
    sims = []
    filtered_count = 0
    for p in pairs:
        src = p.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        sims.append(p.get("similarity", 0))
        if p.get("filtered"):
            filtered_count += 1

    print(f"\n{'='*60}")
    print("DPO PAIR SUMMARY")
    print(f"{'='*60}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Filtered (outside {SIMILARITY_MIN}-{SIMILARITY_MAX}): {filtered_count}")
    print(f"Usable: {len(pairs) - filtered_count}")
    print()

    print("By source:")
    for src in sorted(by_source):
        count = by_source[src]
        src_sims = [p["similarity"] for p in pairs if p["source"] == src]
        avg_sim = sum(src_sims) / len(src_sims) if src_sims else 0
        print(f"  {src}: {count} pairs (avg similarity: {avg_sim:.2f})")

    print()
    print("Similarity distribution:")
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for lo, hi in buckets:
        count = sum(1 for s in sims if lo <= s < hi)
        bar = "#" * min(count, 50)
        label = "***" if SIMILARITY_TARGET_MIN <= lo < SIMILARITY_TARGET_MAX else ""
        print(f"  {lo:.1f}-{hi:.1f}: {count:4d} {bar} {label}")

    # Show example from each source
    print()
    for src in sorted(by_source):
        examples = [p for p in pairs if p["source"] == src and not p.get("filtered")]
        if examples:
            ex = examples[0]
            print(f"Example ({src}, sim={ex['similarity']:.2f}):")
            print(f"  Task: {ex['task_id']}")
            print(f"  Chosen:   {ex['chosen'][:100]}...")
            print(f"  Rejected: {ex['rejected'][:100]}...")
            print()


# ════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPO pairs for CogMem")
    parser.add_argument("--tasks", required=True, help="Path to tasks JSONL")
    parser.add_argument("--bank", default=None, help="Path to memory bank JSON (for retry pairs)")
    parser.add_argument("--output", default="results/dpo_pairs.jsonl", help="Output JSONL path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES, help="Candidates for best-of-N")
    parser.add_argument("--source", default="all", choices=["best_of_n", "mutation", "retry", "all"])
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    N_CANDIDATES = args.n_candidates

    # Load tasks
    tasks = []
    with open(args.tasks) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    # Load memory bank if provided
    episodes = []
    if args.bank and Path(args.bank).exists():
        with open(args.bank) as f:
            episodes = json.load(f)
        print(f"Loaded {len(episodes)} episodes from {args.bank}")

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint for resume
    resume_done = load_checkpoint(args.output) if args.resume else {}
    if resume_done:
        total_done = sum(len(v) for v in resume_done.values())
        print(f"Resuming: {total_done} tasks already processed")

    # Run selected sources
    total_pairs = 0
    source = args.source

    if source in ("best_of_n", "all"):
        total_pairs += generate_best_of_n_pairs(tasks, args.output, args.model, resume_done)

    if source in ("mutation", "all"):
        if episodes:
            total_pairs += generate_mutation_pairs(tasks, episodes, args.output, resume_done)
        else:
            print("\n[Mutation] Skipped — no memory bank provided (use --bank)")

    if source in ("retry", "all"):
        if episodes:
            total_pairs += generate_retry_pairs(episodes, args.output, resume_done)
        else:
            print("\n[Retry] Skipped — no memory bank provided (use --bank)")

    print(f"\nTotal pairs generated: {total_pairs}")

    # Print summary
    print_summary(args.output)

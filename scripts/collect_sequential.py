"""Sequential BigCodeBench collection with retrieval-based Q-values.

Each task retrieves from past episodes, uses them as few-shot context,
generates code, tests it, then updates Q-values of retrieved episodes.

Q-values measure HOW USEFUL a memory is to future tasks — not just pass/fail.

Usage:
    python scripts/collect_sequential.py
    python scripts/collect_sequential.py --resume
    python scripts/collect_sequential.py --analyze-only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import openai
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Ensure cogmem is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution
from cogmem.benchmarks.bigcodebench.prompts import SYSTEM_PROMPT, extract_code

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "qwen2.5:3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Q-learning parameters
Q_INITIAL = 0.5
Q_ALPHA = 0.3
Q_REWARD_SUCCESS = 1.0
Q_REWARD_FAIL = 0.0

# Retrieval parameters
TOP_K_SEMANTIC = 10
TOP_K_FINAL = 3
MIN_SIMILARITY = 0.3

# Paths
MEMORY_BANK_PATH = os.environ.get(
    "COGMEM_BANK_PATH",
    "results/bigcodebench/memory_bank_sequential.json",
)
TASKS_PATH = None  # set from args or auto-detect


# ════════════════════════════════════════════
# MEMORY BANK
# ════════════════════════════════════════════

class SequentialMemoryBank:
    """Episodic memory with retrieval-based Q-value tracking."""

    def __init__(self, path):
        self.path = path
        self.episodes = []
        self.embeddings = []
        self._load()

    def _load(self):
        if Path(self.path).exists():
            try:
                with open(self.path) as f:
                    raw = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: corrupt memory bank at {self.path}, starting fresh")
                return
            valid_episodes = []
            valid_embeddings = []
            for ep in raw:
                try:
                    emb = np.array(ep["intent_embedding"])
                    valid_episodes.append(ep)
                    valid_embeddings.append(emb)
                except (KeyError, TypeError, ValueError):
                    print(f"Warning: skipping episode {ep.get('episode_id', '?')} (bad embedding)")
            self.episodes = valid_episodes
            self.embeddings = valid_embeddings
            print(f"Loaded {len(self.episodes)} episodes from {self.path}")

    def save(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.episodes, f, indent=2)

    def add(self, episode):
        self.episodes.append(episode)
        self.embeddings.append(np.array(episode["intent_embedding"]))

    def retrieve(self, query_embedding, top_k_semantic=10, top_k_final=3,
                 min_similarity=0.3):
        """Two-phase retrieval: semantic similarity then Q-value re-ranking."""
        if not self.episodes:
            return []

        query = np.array(query_embedding)

        # Phase 1: Cosine similarity
        candidates = []
        for i, emb in enumerate(self.embeddings):
            norm = np.linalg.norm(query) * np.linalg.norm(emb) + 1e-8
            cos_sim = float(np.dot(query, emb) / norm)
            if cos_sim >= min_similarity:
                candidates.append((i, cos_sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k_semantic]

        if not candidates:
            return []

        # Phase 2: Re-rank by Q-value (Q weighted more)
        reranked = []
        for i, sim in candidates:
            ep = self.episodes[i]
            score = sim * 0.4 + ep["q_value"] * 0.6
            reranked.append((i, score, sim))

        reranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score, sim in reranked[:top_k_final]:
            ep = self.episodes[i].copy()
            ep["_index"] = i
            ep["_similarity"] = sim
            ep["_score"] = score
            results.append(ep)

        return results

    def update_q(self, episode_index, task_succeeded):
        """Update Q-value based on whether retrieval helped."""
        ep = self.episodes[episode_index]
        reward = Q_REWARD_SUCCESS if task_succeeded else Q_REWARD_FAIL
        ep["q_value"] = ep["q_value"] + Q_ALPHA * (reward - ep["q_value"])
        ep["q_visits"] = ep.get("q_visits", 0) + 1
        if task_succeeded:
            ep["q_successes"] = ep.get("q_successes", 0) + 1
        else:
            ep["q_failures"] = ep.get("q_failures", 0) + 1

    def completed_task_ids(self):
        return {ep["task_id"] for ep in self.episodes}

    def get_q_stats(self):
        if not self.episodes:
            return {}
        q_vals = [ep["q_value"] for ep in self.episodes]
        return {
            "total": len(self.episodes),
            "mean_q": float(np.mean(q_vals)),
            "std_q": float(np.std(q_vals)),
            "min_q": float(min(q_vals)),
            "max_q": float(max(q_vals)),
            "high_q": sum(1 for q in q_vals if q >= 0.7),
            "mid_q": sum(1 for q in q_vals if 0.3 <= q < 0.7),
            "low_q": sum(1 for q in q_vals if q < 0.3),
            "ever_retrieved": sum(
                1 for ep in self.episodes if ep.get("q_visits", 0) > 0
            ),
        }


# ════════════════════════════════════════════
# PROMPT BUILDING
# ════════════════════════════════════════════

def build_prompt_with_retrieval(task_description, retrieved_episodes):
    """Build prompt with retrieved episodes as few-shot examples.

    Only shows SUCCESSFUL episodes as examples.
    Uses the same SYSTEM_PROMPT as evaluation.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    successful = [ep for ep in retrieved_episodes if ep.get("success")]

    if successful:
        parts = ["Here are similar tasks you solved before:\n"]
        for i, ep in enumerate(successful):
            code = ep.get("generated_code", "")
            desc = ep.get("task_description", "")[:200]
            parts.append(
                f"--- Example {i+1} (Q={ep['q_value']:.2f}) ---\n"
                f"Task: {desc}\n"
                f"Solution:\n```python\n{code}\n```\n"
            )
        parts.append(f"--- Current Task ---\n{task_description}")
        messages.append({"role": "user", "content": "\n".join(parts)})
    else:
        messages.append({"role": "user", "content": task_description})

    return messages


# ════════════════════════════════════════════
# MAIN COLLECTION
# ════════════════════════════════════════════

def collect_sequential(tasks_path, resume=False, shuffle=False, model=None, cycle=0):
    """Process tasks sequentially with retrieval + Q-value updates.

    Args:
        tasks_path: Path to tasks JSONL.
        resume: Resume from existing bank.
        shuffle: Randomize task order (use for cycle 1+ so late tasks
                 get retrieval exposure too).
    """
    # Load tasks
    tasks = []
    with open(tasks_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} tasks from {tasks_path}")

    if shuffle:
        import random
        local_rng = random.Random(42)
        local_rng.shuffle(tasks)
        print("Shuffled task order (for cycle 1+ retrieval exposure)")

    # Initialize
    active_model = model or MODEL_NAME
    # Bank path includes cycle number
    bank_path = MEMORY_BANK_PATH.replace(".json", f"_cycle{cycle}.json")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    bank = SequentialMemoryBank(bank_path)
    print(f"Model: {active_model}")
    print(f"Cycle: {cycle}")
    print(f"Bank: {bank_path}")

    # Resume support
    completed = set()
    if resume:
        completed = bank.completed_task_ids()
        if completed:
            print(f"Resuming: {len(completed)} tasks already done")
        remaining = [t for t in tasks if t["task_id"] not in completed]
        done = len(bank.episodes)  # use episode count for ID numbering
        successes = sum(1 for ep in bank.episodes if ep["success"])
    else:
        # Fresh run: discard any stale data from a previous partial run
        bank.episodes = []
        bank.embeddings = []
        remaining = tasks
        done = 0
        successes = 0
    total_tasks = len(tasks)

    start_time = time.time()

    for task in remaining:
        task_id = task["task_id"]
        instruction = task.get("instruct_prompt", task.get("complete_prompt", ""))

        # 1. Embed
        task_embedding = embedder.encode(instruction).tolist()

        # 2. Retrieve (exclude same task to avoid self-retrieval)
        retrieved_raw = bank.retrieve(
            task_embedding,
            top_k_semantic=TOP_K_SEMANTIC,
            top_k_final=TOP_K_FINAL,
            min_similarity=MIN_SIMILARITY,
        )
        retrieved = [ep for ep in retrieved_raw if ep.get("task_id") != task_id]

        # 3. Build prompt
        messages = build_prompt_with_retrieval(instruction, retrieved)

        # 4. Generate with retry (builds trajectory for DPO pairs)
        MAX_ATTEMPTS = 3
        trajectory = []
        success = False
        final_code = None
        last_response = ""
        last_code = ""
        error = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                if attempt > 1:
                    prev_err = trajectory[-1].get("error", "")[:500]
                    messages = build_prompt_with_retrieval(instruction, retrieved)
                    messages.append({"role": "user",
                                     "content": f"Previous attempt failed:\n{prev_err}\nPlease fix."})

                resp = client.chat.completions.create(
                    model=active_model, messages=messages,
                    max_tokens=2048, temperature=0,
                )
                response = resp.choices[0].message.content
                code = extract_code(response)
            except (openai.OpenAIError, ConnectionError, TimeoutError) as e:
                trajectory.append({
                    "attempt": attempt, "code": "", "response": "",
                    "test_result": "ERROR", "error": str(e),
                })
                continue

            if code:
                result = evaluate_solution(task, code, timeout=30, mode="subprocess")
                passed = result["passed"]
                err = result.get("error")
            else:
                passed = False
                err = "Empty code"

            trajectory.append({
                "attempt": attempt, "code": code, "response": response,
                "test_result": "PASS" if passed else "FAIL",
                "error": err if not passed else None,
            })

            last_response = response
            last_code = code

            if passed:
                success = True
                final_code = code
                error = None
                break

        if not success and trajectory:
            error = trajectory[-1].get("error")

        # 5. Store episode
        episode = {
            "episode_id": f"bcb_c{cycle}_{done:04d}",
            "task_id": task_id,
            "task_description": instruction,
            "generated_code": last_code,
            "final_code": final_code,
            "script": last_response,
            "trajectory": trajectory,
            "num_attempts": len(trajectory),
            "success": success,
            "q_value": Q_INITIAL,
            "q_visits": 0,
            "q_successes": 0,
            "q_failures": 0,
            "retrieved_from": [ep["episode_id"] for ep in retrieved if ep.get("success")],
            "retrieved_by": [],
            "intent_embedding": task_embedding,
            "error": error,
            "entry_point": task.get("entry_point", ""),
            "model": active_model,
            "cycle": cycle,
            "timestamp": time.time(),
        }
        bank.add(episode)

        # 7. Update Q-values of retrieved episodes that were actually shown
        #    Only successful episodes are included in the prompt (see build_prompt_with_retrieval),
        #    so only credit those — others weren't used and shouldn't get Q-updates.
        shown_in_prompt = [ep for ep in retrieved if ep.get("success")]
        for ret_ep in shown_in_prompt:
            idx = ret_ep["_index"]
            bank.update_q(idx, success)
            bank.episodes[idx].setdefault("retrieved_by", []).append(
                episode["episode_id"]
            )

        # 8. Track + print
        done += 1
        if success:
            successes += 1

        if done % 10 == 0 or done <= 20 or done == total_tasks:
            elapsed = time.time() - start_time
            rate = (done - len(completed)) / elapsed * 3600 if elapsed > 0 else 0
            eta = (total_tasks - done) / (rate / 60) if rate > 0 else 0
            q = bank.get_q_stats()
            print(
                f"[{done}/{total_tasks}] {task_id}: "
                f"{'PASS' if success else 'FAIL'} | "
                f"Pass: {successes}/{done} ({successes/done:.1%}) | "
                f"Ret: {len(retrieved)} | "
                f"Q: {q.get('mean_q', 0):.2f} "
                f"[{q.get('low_q', 0)}L/{q.get('mid_q', 0)}M/{q.get('high_q', 0)}H] | "
                f"ETA: {eta:.0f}m"
            )

        # Save every 100
        if done % 100 == 0:
            bank.save()

    bank.save()

    q = bank.get_q_stats()
    print(f"\n{'='*60}")
    print("COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {done}, Passed: {successes} ({successes/max(done,1):.1%})")
    if q:
        print(f"Q-values: mean={q['mean_q']:.3f} std={q['std_q']:.3f}")
        print(f"  High (>=0.7): {q['high_q']}")
        print(f"  Mid (0.3-0.7): {q['mid_q']}")
        print(f"  Low (<0.3): {q['low_q']}")
        print(f"  Ever retrieved: {q['ever_retrieved']}/{q['total']}")
    else:
        print("No episodes collected!")

    return bank


# ════════════════════════════════════════════
# ANALYSIS
# ════════════════════════════════════════════

def analyze_q_values(bank_path):
    """Analyze Q-value distribution and validate Q predicts helpfulness."""
    with open(bank_path) as f:
        episodes = json.load(f)

    q_vals = [ep["q_value"] for ep in episodes]

    print(f"\nQ-VALUE ANALYSIS ({len(episodes)} episodes)")
    print("=" * 60)

    # Distribution histogram
    print("\nDistribution:")
    for start in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        end = start + 0.1
        count = sum(1 for q in q_vals if start <= q < end)
        bar = "#" * min(count, 80)
        print(f"  {start:.1f}-{end:.1f}: {count:4d} {bar}")

    # Most retrieved
    visited = sorted(episodes, key=lambda x: x.get("q_visits", 0), reverse=True)
    print("\nMost retrieved episodes:")
    for ep in visited[:10]:
        v = ep.get("q_visits", 0)
        s = ep.get("q_successes", 0)
        f = ep.get("q_failures", 0)
        if v > 0:
            print(
                f"  {ep['task_id']}: Q={ep['q_value']:.2f} "
                f"visits={v} helped={s} hurt={f} "
                f"help_rate={s/v:.0%} "
                f"{'PASS' if ep['success'] else 'FAIL'}"
            )

    # KEY: Does Q predict helpfulness?
    print("\nKEY VALIDATION: Does Q predict helpfulness?")
    for q_min, q_max, label in [
        (0.0, 0.3, "Low Q"),
        (0.3, 0.7, "Mid Q"),
        (0.7, 1.0, "High Q"),
    ]:
        bucket = [
            ep for ep in episodes
            if q_min <= ep["q_value"] < q_max and ep.get("q_visits", 0) > 0
        ]
        if bucket:
            avg_help = np.mean([
                ep.get("q_successes", 0) / max(ep.get("q_visits", 1), 1)
                for ep in bucket
            ])
            print(f"  {label} ({q_min:.1f}-{q_max:.1f}): "
                  f"{len(bucket)} episodes, avg help rate = {avg_help:.0%}")

    print("\nIf High Q help rate > Low Q help rate -> Q-values work!")


# ════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential BigCodeBench collection")
    parser.add_argument("--tasks", default=None, help="Path to tasks JSONL")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--shuffle", action="store_true", help="Randomize task order (cycle 1+)")
    parser.add_argument("--cycle", type=int, default=0, help="Cycle number (0, 1, 2, ...)")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing data")
    parser.add_argument("--model", default=MODEL_NAME, help="Ollama model name")
    parser.add_argument("--bank", default=MEMORY_BANK_PATH, help="Memory bank path")
    args = parser.parse_args()

    MODEL_NAME = args.model
    MEMORY_BANK_PATH = args.bank

    if args.analyze_only:
        bank_path = MEMORY_BANK_PATH.replace(".json", f"_cycle{args.cycle}.json")
        analyze_q_values(bank_path)
    else:
        tasks_path = args.tasks
        if tasks_path is None:
            for p in [
                "/notebooks/bigcodebench_tasks.jsonl",
                "results/bigcodebench/tasks.jsonl",
            ]:
                if Path(p).exists():
                    tasks_path = p
                    break
        if tasks_path is None:
            print("ERROR: No tasks file found. Use --tasks <path>")
            sys.exit(1)

        print(f"Model: {MODEL_NAME}")
        print(f"Tasks: {tasks_path}")
        print(f"Memory bank: {MEMORY_BANK_PATH}")
        print()

        bank = collect_sequential(
            tasks_path, resume=args.resume, shuffle=args.shuffle,
            model=args.model, cycle=args.cycle,
        )
        bank_path = MEMORY_BANK_PATH.replace(".json", f"_cycle{args.cycle}.json")
        print("\nRunning Q-value analysis...")
        analyze_q_values(bank_path)

"""Full wake/build cycle runner for episode-first cluster memories."""

import json
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cogmem.patches.evaluate import evaluate_cold, evaluate_patched
from cogmem.patches.memory_bank import ClusterMemoryBank
from cogmem.patches.wake import run_wake_cycle


def run_cogmem_cycle(
    tasks_path: str,
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    patch_bank_dir: str = "results/cluster_memories",
    n_cycles: int = 5,
    n_candidates: int = 8,
    eval_size: int = 200,
):
    """Run full wake/sleep cycles.

    Args:
        tasks_path: Path to BigCodeBench tasks JSONL.
        base_model_name: HuggingFace model name.
        patch_bank_dir: Directory for cluster-memory storage.
        n_cycles: Number of wake/sleep cycles.
        n_candidates: Candidates per task during wake.
        eval_size: Number of tasks for evaluation (subset for speed).
    """
    # Load tasks
    tasks = []
    with open(tasks_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} tasks")

    # Load base model (4-bit, stays on GPU)
    print("Loading base model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on {base_model.device}")

    # Load embedder (CPU)
    print("Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # Initialize or load cluster memory bank
    patch_bank = ClusterMemoryBank(patch_bank_dir)
    if Path(patch_bank_dir).exists():
        patch_bank.load()

    learning_curve = []

    for cycle in range(n_cycles):
        print(f"\n{'#' * 60}")
        print(f"# CYCLE {cycle}")
        print(f"{'#' * 60}")

        # ═══ WAKE ═══
        print(f"\n--- WAKE (cycle {cycle}) ---")
        wake_stats = run_wake_cycle(
            tasks, base_model, tokenizer, patch_bank, embedder,
            n_candidates=n_candidates,
        )
        print(f"Wake results: {wake_stats['tasks_passed']}/{wake_stats['total_tasks']} "
              f"({wake_stats['pass_rate']:.1%}), "
              f"{wake_stats['episodes_created']} new episodes")

        # ═══ EVALUATE ═══
        print(f"\n--- EVALUATE (cycle {cycle}) ---")
        eval_tasks = tasks[:eval_size]

        cold_rate = evaluate_cold(
            base_model, tokenizer, eval_tasks,
        )
        patched_rate = evaluate_patched(
            base_model, tokenizer, eval_tasks, patch_bank, embedder,
        )

        point = {
            "cycle": cycle,
            "cold_pass_rate": cold_rate,
            "patched_pass_rate": patched_rate,
            "improvement": patched_rate - cold_rate,
            "total_memories": len(patch_bank.memories),
            "bank_stats": patch_bank.stats(),
            "wake_stats": wake_stats,
        }
        learning_curve.append(point)

        print(f"\nCycle {cycle} results:")
        print(f"  Cold (no patches): {cold_rate:.1%}")
        print(f"  Patched:           {patched_rate:.1%}")
        print(f"  Improvement:       {patched_rate - cold_rate:+.1%}")
        print(f"  Episodes stored:   {len(patch_bank.episodes)}")
        print(f"  Memories in bank:  {len(patch_bank.memories)}")

        # Save learning curve
        results_dir = Path(patch_bank_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "learning_curve.json", "w") as f:
            json.dump(learning_curve, f, indent=2)

    # Final summary
    print(f"\n{'=' * 60}")
    print("COGMEM CYCLES COMPLETE")
    print(f"{'=' * 60}")
    print(f"{'Cycle':<8} {'Cold':>8} {'Patched':>10} {'Delta':>8} {'Memories':>10}")
    print("-" * 46)
    for p in learning_curve:
        print(
            f"{p['cycle']:<8} "
            f"{p['cold_pass_rate']:>7.1%} "
            f"{p['patched_pass_rate']:>9.1%} "
            f"{p['improvement']:>+7.1%} "
            f"{p['total_memories']:>10}"
        )

    return learning_curve

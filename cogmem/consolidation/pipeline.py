"""Consolidation pipeline — orchestrates the full "sleep" phase.

Two pipeline modes:

1. **Experiment 1 (legacy)**: Compare 5 selection policies for SFT-only.
   Uses: select.py -> abstract.py -> train_lora*.py -> verify.py

2. **Sleep phase (new)**: Q-value triage -> SFT + GRPO + merge.
   Uses: triage.py -> sft_train.py + grpo_train.py -> merge.py -> verify.py
   This is the core CogMem novelty.
"""

import json
from datetime import datetime
from pathlib import Path
from statistics import mean

from cogmem.config import CogMemConfig, ConsolidationConfig
from cogmem.consolidation.abstract import prepare_training_dataset, save_as_jsonl
from cogmem.consolidation.grpo_train import (
    create_bigcodebench_reward_fn,
    prepare_grpo_dataset,
    train_grpo,
)
from cogmem.consolidation.merge import merge_adapters
from cogmem.consolidation.prune import tag_consolidated
from cogmem.consolidation.select import POLICIES
from cogmem.consolidation.sft_train import train_sft_dora
from cogmem.consolidation.triage import select_anchors, split_holdout, triage_episodes
from cogmem.consolidation.verify import (
    aggregate_seed_results,
    run_verification_single_seed,
    verification_passed,
)
from cogmem.memory.memory_bank import MemoryBank
from cogmem.utils.logging import save_results


# -------------------------------------------------------------------------
# Sleep phase — the core CogMem consolidation pipeline
# -------------------------------------------------------------------------

def run_sleep_phase(
    memory_bank_path: str,
    config: ConsolidationConfig,
    benchmark: str = "bigcodebench",
    run_task_fn=None,
) -> dict:
    """Run the full Q-value triage consolidation pipeline.

    Steps:
        1. Load memory bank
        2. Triage episodes by Q-value zone
        3. Split holdout from high-Q zone
        4. Step A: SFT DoRA on high-Q episodes (memorize patterns)
        5. Step B: GRPO DoRA on middle-Q episodes (practice reasoning)
        6. Step C: Merge both adapters
        7. Verify merged adapter on holdout
        8. Tag consolidated episodes

    Args:
        memory_bank_path: Path to memory bank JSON.
        config: ConsolidationConfig.
        benchmark: "bigcodebench" or "alfworld".
        run_task_fn: Optional callable(task_description) -> {"success": bool}
            for holdout verification. If None, verification is skipped.

    Returns:
        Dict with triage stats, adapter paths, verification results.
    """
    print("=" * 60)
    print(f"COGMEM SLEEP PHASE — {benchmark.upper()}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # 1. Load
    bank = MemoryBank.load(memory_bank_path)
    episodes = list(bank)
    print(f"\n1. Loaded memory bank: {len(episodes)} episodes")

    # 2. Triage
    print(f"\n2. Q-value triage:")
    zones = triage_episodes(episodes, config)

    # 3. Split holdout from high-Q
    print(f"\n3. Holdout split:")
    high_train, holdout = split_holdout(
        zones["high"], fraction=config.holdout_fraction, seed=config.seed
    )
    print(f"  High-Q for SFT: {len(high_train)}")
    print(f"  Holdout for verification: {len(holdout)}")

    anchors = select_anchors(zones["high"], config)

    # 4. Step A: SFT on high-Q
    print(f"\n{'=' * 60}")
    print("STEP A: SFT DoRA on high-Q episodes")
    print("=" * 60)

    sft_path, sft_loss = train_sft_dora(
        high_train, config, benchmark=benchmark, output_name="sft_dora"
    )

    # 5. Step B: GRPO on middle-Q
    print(f"\n{'=' * 60}")
    print("STEP B: GRPO DoRA on middle-Q episodes")
    print("=" * 60)

    grpo_path = None
    merged_path = sft_path  # default: SFT-only if GRPO skipped

    if len(zones["middle"]) < config.grpo_min_episodes:
        print(f"  Only {len(zones['middle'])} middle-Q episodes "
              f"(need >= {config.grpo_min_episodes})")
        print("  Skipping GRPO — using SFT adapter only.")
    else:
        grpo_dataset = prepare_grpo_dataset(zones["middle"], anchors)

        if benchmark == "bigcodebench":
            reward_fn = create_bigcodebench_reward_fn(grpo_dataset)
        else:
            # ALFWorld: placeholder — use heuristic scoring
            def _alfworld_placeholder_reward(completions, **kw):
                return [0.5] * len(completions)

            reward_fn = _alfworld_placeholder_reward

        grpo_path = train_grpo(
            grpo_dataset, reward_fn, config, output_name="grpo_dora"
        )

        # 6. Step C: Merge
        print(f"\n{'=' * 60}")
        print("STEP C: Merge SFT + GRPO adapters")
        print("=" * 60)

        merged_path = merge_adapters(sft_path, grpo_path, config)

    # 7. Verify
    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    print("=" * 60)

    verification = {}
    if run_task_fn is not None and holdout:
        seed_results = []
        for seed in config.eval_seeds:
            r = run_verification_single_seed(holdout, run_task_fn, seed)
            seed_results.append(r)
        verification = aggregate_seed_results(seed_results)
        print(f"  Merged adapter: {verification['mean']:.1%} "
              f"(+/- {verification['std']:.1%})")
    else:
        print("  Skipped (no run_task_fn or no holdout)")

    # 8. Tag consolidated episodes (exclude holdout — those weren't trained on)
    holdout_ids = {ep["episode_id"] for ep in holdout}
    consolidated_ids = {ep["episode_id"] for ep in zones["high"]} - holdout_ids
    tag_consolidated(episodes, consolidated_ids, domain=benchmark)
    bank.save(memory_bank_path)
    print(f"\n8. Tagged {len(consolidated_ids)} high-Q episodes as consolidated")

    # Save results
    results = {
        "benchmark": benchmark,
        "timestamp": datetime.now().isoformat(),
        "triage": {
            "high": len(zones["high"]),
            "middle": len(zones["middle"]),
            "low": len(zones["low"]),
        },
        "sft": {
            "path": sft_path,
            "training_episodes": len(high_train),
            "final_loss": sft_loss,
        },
        "grpo": {
            "path": grpo_path,
            "practice_problems": len(zones["middle"]) if grpo_path else 0,
        },
        "merged": {"path": merged_path},
        "verification": verification,
    }

    results_dir = Path(config.experiments_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = str(results_dir / f"sleep_phase_{benchmark}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("SLEEP PHASE COMPLETE")
    print(f"  SFT adapter:    {sft_path}")
    print(f"  GRPO adapter:   {grpo_path}")
    print(f"  Merged adapter: {merged_path}")
    if verification:
        print(f"  Holdout rate:   {verification['mean']:.1%}")
    print("=" * 60)

    return results


def run_iterative_consolidation(
    memory_bank_path: str,
    config: ConsolidationConfig,
    benchmark: str = "bigcodebench",
    collect_fn=None,
    run_task_fn=None,
    max_cycles: int = 8,
) -> list[dict]:
    """Run multiple sleep cycles. Each cycle:

    1. Run sleep phase (SFT + GRPO + merge)
    2. Optionally collect new episodes with improved model
    3. Track learning curve
    4. Stop on plateau

    Args:
        memory_bank_path: Path to memory bank JSON.
        config: ConsolidationConfig.
        benchmark: "bigcodebench" or "alfworld".
        collect_fn: Optional callable(adapter_path) -> list[dict].
            Collects new episodes using the improved model.
        run_task_fn: Optional callable for verification.
        max_cycles: Maximum number of consolidation cycles.

    Returns:
        Learning curve: list of dicts with per-cycle metrics.
    """
    learning_curve = []

    for cycle in range(1, max_cycles + 1):
        print(f"\n{'#' * 60}")
        print(f"# CYCLE {cycle} / {max_cycles}")
        print(f"{'#' * 60}")

        # Run sleep phase
        results = run_sleep_phase(
            memory_bank_path, config, benchmark, run_task_fn
        )

        # Record learning curve
        point = {
            "cycle": cycle,
            "triage": results["triage"],
            "sft_loss": results["sft"].get("final_loss"),
            "merged_path": results["merged"]["path"],
            "verification": results.get("verification", {}),
        }
        learning_curve.append(point)

        # Optionally collect new episodes with improved model
        if collect_fn is not None:
            print(f"\n  Collecting new episodes with improved model...")
            new_episodes = collect_fn(results["merged"]["path"])
            if new_episodes:
                bank = MemoryBank.load(memory_bank_path)
                current = list(bank)
                current.extend(new_episodes)
                bank_new = MemoryBank(current)
                bank_new.save(memory_bank_path)
                print(f"  Added {len(new_episodes)} new episodes "
                      f"(total: {len(current)})")

        # Check for plateau (only when verification was actually run)
        if len(learning_curve) >= 3:
            recent_verif = [
                p.get("verification", {}).get("mean")
                for p in learning_curve[-3:]
            ]
            valid = [v for v in recent_verif if v is not None and v > 0]
            if len(valid) >= 3:
                gain = valid[-1] - valid[0]
                if gain < 0.01:
                    print("\n  Plateau detected (gain < 1% over 3 cycles).")
                    break

    # Save learning curve
    results_dir = Path(config.experiments_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    curve_path = str(results_dir / f"learning_curve_{benchmark}.json")
    with open(curve_path, "w") as f:
        json.dump(learning_curve, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("ITERATIVE CONSOLIDATION COMPLETE")
    print("=" * 60)
    print(f"{'Cycle':<8} {'SFT Loss':<12} {'High-Q':<10} {'Middle-Q':<10}")
    print("-" * 40)
    for p in learning_curve:
        loss = p.get("sft_loss")
        loss_str = f"{loss:.3f}" if loss else "N/A"
        print(f"{p['cycle']:<8} {loss_str:<12} "
              f"{p['triage']['high']:<10} {p['triage']['middle']:<10}")

    return learning_curve


# -------------------------------------------------------------------------
# Legacy pipelines (Experiment 1 & 2) — preserved for backward compat
# -------------------------------------------------------------------------

def _load_replay_buffer(path: str) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    with open(path) as f:
        return json.load(f)


def run_consolidation(
    policy_name: str,
    available_episodes: list[dict],
    holdout_episodes: list[dict],
    replay_buffer: list[dict],
    config: CogMemConfig,
    run_task_fn=None,
) -> dict:
    """Legacy: single-policy SFT consolidation."""
    from cogmem.consolidation.train_lora import train_lora_together
    from cogmem.consolidation.train_lora_local import train_lora_local

    policy_fn = POLICIES[policy_name]
    selected = policy_fn(available_episodes, config)

    training_pairs = prepare_training_dataset(selected, replay_buffer=replay_buffer)
    jsonl_dir = f"{config.logs_dir}/jsonl"
    jsonl_path = save_as_jsonl(training_pairs, f"{jsonl_dir}/{policy_name}.jsonl")

    if config.lora_provider == "local":
        train_result = train_lora_local(jsonl_path, config, policy_name=policy_name)
    else:
        train_result = train_lora_together(jsonl_path, config, policy_name=policy_name)

    seed_results = []
    if run_task_fn is not None:
        for seed in config.eval_seeds:
            r = run_verification_single_seed(holdout_episodes, run_task_fn, seed)
            seed_results.append(r)

    verification = aggregate_seed_results(seed_results) if seed_results else {}

    result = {
        "policy": policy_name,
        "episodes_selected": len(selected),
        "training_pairs": len(training_pairs),
        "q_value_mean": mean([ep["q_value"] for ep in selected]) if selected else 0,
        "verification": verification,
        "train_result": train_result,
        "jsonl_path": jsonl_path,
    }

    save_results(result, config.logs_dir, f"consolidation_{policy_name}")
    return result


def run_experiment_1(config: CogMemConfig, run_task_fn=None) -> dict:
    """Legacy: compare 5 selection policies on same holdout."""
    from cogmem.evaluation.compare import (
        best_policy,
        format_comparison_table,
        save_comparison,
    )
    from cogmem.utils.logging import save_config_snapshot

    bank = MemoryBank.load(config.memory_bank_path)
    bank_hash = bank.sha256()

    n_success = len(bank.successful())
    holdout_n = 30 if n_success > 200 else config.verification_holdout
    holdout, available = bank.stratified_holdout(n=holdout_n, seed=config.seed)

    replay_buffer = _load_replay_buffer(config.replay_buffer_path)
    save_config_snapshot(config, config.logs_dir)

    all_results = {
        "memory_bank_hash": bank_hash,
        "holdout_ids": [ep["episode_id"] for ep in holdout],
    }

    for policy_name in ["q_top_k", "recency", "frequency", "random", "all"]:
        print(f"\n{'=' * 60}")
        print(f"Running consolidation: {policy_name}")
        print("=" * 60)
        result = run_consolidation(
            policy_name, available, holdout, replay_buffer, config, run_task_fn
        )
        all_results[policy_name] = result

    policy_results = {k: v for k, v in all_results.items() if k in POLICIES}
    print("\n" + format_comparison_table(policy_results))
    print(f"\nBest policy: {best_policy(policy_results)}")

    save_comparison(all_results, f"{config.logs_dir}/experiment_1_results.json")
    return all_results


def run_experiment_2(
    config: CogMemConfig, exp1_results: dict, run_task_fns: dict
) -> dict:
    """Legacy: full system ablation."""
    from cogmem.evaluation.compare import (
        format_comparison_table,
        save_comparison,
    )

    bank = MemoryBank.load(config.memory_bank_path)
    holdout_n = 30 if len(bank.successful()) > 200 else config.verification_holdout
    holdout, _ = bank.stratified_holdout(n=holdout_n, seed=config.seed)

    all_results = {}
    for variant_name, run_fn in run_task_fns.items():
        print(f"\nEvaluating variant: {variant_name}")
        seed_results = []
        for seed in config.eval_seeds:
            r = run_verification_single_seed(holdout, run_fn, seed)
            seed_results.append(r)
        all_results[variant_name] = aggregate_seed_results(seed_results)

    print(
        "\n"
        + format_comparison_table(
            {k: {"verification": v, "episodes_selected": "-"} for k, v in all_results.items()}
        )
    )

    save_comparison(all_results, f"{config.logs_dir}/experiment_2_results.json")
    return all_results

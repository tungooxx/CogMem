import json
from pathlib import Path
from statistics import mean

from cogmem.config import CogMemConfig
from cogmem.consolidation.abstract import prepare_training_dataset, save_as_jsonl
from cogmem.consolidation.select import POLICIES
from cogmem.consolidation.train_lora import train_lora_together
from cogmem.consolidation.verify import (
    aggregate_seed_results,
    run_verification_single_seed,
    verification_passed,
)
from cogmem.evaluation.compare import best_policy, format_comparison_table, save_comparison
from cogmem.memory.memory_bank import MemoryBank
from cogmem.utils.logging import file_sha256, save_config_snapshot, save_results


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
    # Select
    policy_fn = POLICIES[policy_name]
    selected = policy_fn(available_episodes, config)

    # Abstract + save JSONL
    training_pairs = prepare_training_dataset(selected, replay_buffer=replay_buffer)
    jsonl_dir = f"{config.logs_dir}/jsonl"
    jsonl_path = save_as_jsonl(training_pairs, f"{jsonl_dir}/{policy_name}.jsonl")

    # Train LoRA
    train_result = train_lora_together(jsonl_path, config, policy_name=policy_name)

    # Verify with multiple seeds
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
    # Load and hash memory bank
    bank = MemoryBank.load(config.memory_bank_path)
    bank_hash = bank.sha256()

    # Determine holdout size
    n_success = len(bank.successful())
    holdout_n = 30 if n_success > 200 else config.verification_holdout

    # Split
    holdout, available = bank.stratified_holdout(n=holdout_n, seed=config.seed)

    # Replay buffer
    replay_buffer = _load_replay_buffer(config.replay_buffer_path)

    # Save reproducibility artifacts
    save_config_snapshot(config, config.logs_dir)

    all_results = {"memory_bank_hash": bank_hash, "holdout_ids": [ep["episode_id"] for ep in holdout]}

    for policy_name in ["q_top_k", "recency", "frequency", "random", "all"]:
        print(f"\n{'=' * 60}")
        print(f"Running consolidation: {policy_name}")
        print(f"{'=' * 60}")
        result = run_consolidation(
            policy_name, available, holdout, replay_buffer, config, run_task_fn
        )
        all_results[policy_name] = result

    # Print comparison
    policy_results = {k: v for k, v in all_results.items() if k in POLICIES}
    print("\n" + format_comparison_table(policy_results))
    print(f"\nBest policy: {best_policy(policy_results)}")

    save_comparison(all_results, f"{config.logs_dir}/experiment_1_results.json")
    return all_results


def run_experiment_2(config: CogMemConfig, exp1_results: dict, run_task_fns: dict) -> dict:
    """Full system ablation. run_task_fns maps variant name to a callable.

    Expected keys in run_task_fns:
    - "cold_3b": base 3B, no adapter, no memory
    - "memrl_3b": base 3B with full episodic retrieval
    - "consolidated_3b": base 3B + best LoRA from Exp 1, no memory
    - "cogmem_3b": base 3B + best LoRA + router (consolidated + episodic fallback)
    - "cold_8b": Groq 8B, no memory (optional)
    - "cold_70b": Groq 70B, no memory (optional)
    """
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

    print("\n" + format_comparison_table(
        {k: {"verification": v, "episodes_selected": "-"} for k, v in all_results.items()}
    ))

    save_comparison(all_results, f"{config.logs_dir}/experiment_2_results.json")
    return all_results

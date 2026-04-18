"""Q-STaR consolidation pipeline.

Orchestrates the full "sleep" phase:
1. Select high-Q episodes
2. Build preference pairs from Q-value differences
3. Train generator DoRA (SFT on high-Q, then DPO on Q-value pairs)
4. Train verifier DoRA (DPO on same Q-value pairs)
5. Evaluate

Also preserves legacy Experiment 1/2 pipelines for backward compat.
"""

import json
from datetime import datetime
from pathlib import Path
from statistics import mean

from datasets import Dataset

from cogmem.config import CogMemConfig
from cogmem.consolidation.abstract import (
    prepare_preference_dataset,
    prepare_skill_training_dataset,
    prepare_training_dataset,
    save_as_jsonl,
)
from cogmem.consolidation.adapter_registry import AdapterRegistry
from cogmem.consolidation.proceduralize import build_skill_cards
from cogmem.consolidation.select import POLICIES, filter_manifest_eligible
from cogmem.memory.memory_bank import MemoryBank
from cogmem.memory.schema import get_episode_helpfulness
from cogmem.memory.skill_store import SkillStore, skill_family_key
from cogmem.utils.logging import save_results


# -------------------------------------------------------------------------
# Q-STaR cycle — the core CogMem pipeline
# -------------------------------------------------------------------------

def run_qstar_cycle(
    memory_bank_path: str,
    config: CogMemConfig,
    cycle: int = 0,
    run_task_fn=None,
    existing_skill_cards_path: str | None = None,
) -> dict:
    """Run one Q-STaR consolidation cycle.

    Steps:
        1. Load memory bank, select high-Q episodes
        2. Build preference pairs from Q-value differences
        3. Train generator DoRA (SFT then DPO)
        4. Train verifier DoRA (DPO)
        5. Evaluate

    Args:
        memory_bank_path: Path to memory bank JSON.
        config: CogMemConfig.
        cycle: Current cycle number (0-indexed).
        run_task_fn: Optional callable for evaluation.
        existing_skill_cards_path: Optional prevalidated skill-card store to reuse.

    Returns:
        Dict with adapter paths, training stats, evaluation results.
    """
    from cogmem.consolidation.train_generator import train_generator_full
    from cogmem.consolidation.train_verifier import train_verifier
    from cogmem.consolidation.verify import (
        aggregate_seed_results,
        run_verification_single_seed,
        verification_passed,
    )

    print("=" * 60)
    print(f"Q-STaR CYCLE {cycle}")
    print(f"Model: {config.active_model_hf}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # 1. Load and split holdout BEFORE training
    bank = MemoryBank.load(memory_bank_path)
    all_episodes = filter_manifest_eligible(list(bank), config)
    print(f"\n1. Memory bank: {len(all_episodes)} episodes")

    holdout_n = min(config.min_holdout, len(all_episodes))
    holdout_episodes = all_episodes[:holdout_n]
    holdout_task_ids = {ep.get("task_id") for ep in holdout_episodes if ep.get("task_id")}
    available_episodes = [ep for ep in all_episodes if ep.get("task_id") not in holdout_task_ids]
    print(f"  Holdout: {len(holdout_episodes)} ({len(holdout_task_ids)} tasks), "
          f"Available: {len(available_episodes)}")

    select_fn = POLICIES["q_top_k"]
    selected = select_fn(available_episodes, config)
    successes = [ep for ep in available_episodes if ep.get("success")]
    print(f"  Successes: {len(successes)}")
    print(f"  High-Q selected: {len(selected)} (Q >= {config.q_threshold})")

    default_skill_cards_path = str(Path(config.skills_dir) / f"skill_cards_cycle_{cycle}.json")
    skill_cards_path = existing_skill_cards_path or default_skill_cards_path
    if existing_skill_cards_path and Path(existing_skill_cards_path).exists():
        skill_store = SkillStore.load(existing_skill_cards_path)
        print(f"  Skill cards: loaded existing store from {existing_skill_cards_path}")
    else:
        skill_store = build_skill_cards(
            available_episodes,
            holdout_episodes,
            config=config,
            output_path=skill_cards_path,
        )
    skill_summary = skill_store.summary()
    promoted_cards = list(skill_store.filter(promoted=True))
    promoted_skill_ids = [card["skill_id"] for card in promoted_cards]
    promoted_manifest_ids = sorted(
        {
            manifest_id
            for card in promoted_cards
            for manifest_id in card.get("manifest_ids", [])
        }
    )
    promoted_families = sorted(
        {
            family_key
            for card in promoted_cards
            for family_key in [skill_family_key(card)]
            if family_key
        }
    )
    print(
        f"  Skill cards: {skill_summary['total']} candidates, "
        f"{skill_summary['promoted']} promoted"
    )

    # 2. Build preference pairs (from available episodes only)
    print("\n2. Building preference pairs from Q-values...")
    pref_pairs = prepare_preference_dataset(available_episodes, config)

    pref_dataset = None
    if pref_pairs:
        pref_dataset = Dataset.from_list(pref_pairs)

    skill_training_pairs = prepare_skill_training_dataset(
        promoted_cards,
        available_episodes,
        config=config,
    )
    training_source = "skill_cards" if skill_training_pairs else "episodes"
    training_manifest_ids = (
        promoted_manifest_ids
        if skill_training_pairs else
        sorted({ep.get("manifest_id") for ep in selected if ep.get("manifest_id")})
    )
    training_families = (
        promoted_families
        if skill_training_pairs else
        sorted({ep.get("task_type") for ep in selected if ep.get("task_type")})
    )

    # 3. Train generator (SFT then DPO)
    print(f"\n{'=' * 60}")
    print("STEP 3: Train Generator DoRA (SFT -> DPO)")
    print("=" * 60)

    if not selected and not skill_training_pairs:
        print("  No high-Q episodes available. Skipping training.")
        return {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "model": config.active_model_hf,
            "total_episodes": len(all_episodes),
            "high_q_episodes": 0,
            "preference_pairs": len(pref_pairs),
            "generator_path": None,
            "verifier_path": None,
            "skill_cards_path": skill_cards_path,
            "skill_cards_total": skill_summary["total"],
            "skill_cards_promoted": skill_summary["promoted"],
            "training_source": training_source,
            "training_examples": 0,
            "verification": {},
            "status": "skipped_no_data",
        }

    min_promoted_skills = int(getattr(config, "min_promoted_skills_for_adapter", 1))
    min_skill_families = int(getattr(config, "min_skill_families_for_adapter", 1))
    min_skill_pairs = int(getattr(config, "min_skill_training_pairs_for_adapter", 1))
    if (
        training_source != "skill_cards"
        or len(promoted_cards) < min_promoted_skills
        or len(promoted_families) < min_skill_families
        or len(skill_training_pairs) < min_skill_pairs
    ):
        print(
            "  Skipping adapter training: "
            f"source={training_source}, "
            f"promoted_skills={len(promoted_cards)}/{min_promoted_skills}, "
            f"families={len(promoted_families)}/{min_skill_families}, "
            f"skill_pairs={len(skill_training_pairs)}/{min_skill_pairs}"
        )
        return {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "model": config.active_model_hf,
            "total_episodes": len(all_episodes),
            "high_q_episodes": len(selected),
            "preference_pairs": len(pref_pairs),
            "generator_path": None,
            "verifier_path": None,
            "skill_cards_path": skill_cards_path,
            "skill_cards_total": skill_summary["total"],
            "skill_cards_promoted": skill_summary["promoted"],
            "training_source": training_source,
            "training_examples": len(skill_training_pairs),
            "adapter_registry_path": config.adapter_registry_path,
            "verification": {},
            "status": "skipped_skill_gate",
        }

    generator_path = train_generator_full(
        selected, pref_dataset, config, cycle=cycle,
        source_skill_card_ids=promoted_skill_ids,
        training_manifest_ids=training_manifest_ids,
        compatible_families=training_families,
        registry_path=config.adapter_registry_path,
        adapter_role="global",
        sft_pairs=skill_training_pairs or None,
    )

    # 4. Train verifier (DPO)
    print(f"\n{'=' * 60}")
    print("STEP 4: Train Verifier DoRA (DPO)")
    print("=" * 60)

    verifier_path = None
    if pref_dataset is not None and len(pref_dataset) >= config.min_dpo_pairs:
        verifier_path = train_verifier(pref_dataset, config, cycle=cycle)
    else:
        n = len(pref_dataset) if pref_dataset else 0
        print(f"  Skipping verifier: {n} pairs (need >= {config.min_dpo_pairs})")

    # 5. Evaluate
    print(f"\n{'=' * 60}")
    print("STEP 5: Evaluate")
    print("=" * 60)

    verification = {}
    if run_task_fn is not None:
        seed_results = []
        for seed in config.eval_seeds:
            r = run_verification_single_seed(
                holdout_episodes, run_task_fn, seed
            )
            seed_results.append(r)
        verification = aggregate_seed_results(seed_results)
        print(f"  Pass rate: {verification['mean']:.1%}")

    if generator_path:
        registry = AdapterRegistry.load(config.adapter_registry_path)
        record = registry.find_by_path(generator_path)
        if record is not None:
            registry.update(
                record.adapter_id,
                dev_gain=verification.get("mean", 0.0) if verification else 0.0,
                metadata={
                    **record.metadata,
                    "verification": verification,
                },
            )
            registry.save(config.adapter_registry_path)

    # Save results
    results = {
        "cycle": cycle,
        "timestamp": datetime.now().isoformat(),
        "model": config.active_model_hf,
        "total_episodes": len(all_episodes),
        "high_q_episodes": len(selected),
        "preference_pairs": len(pref_pairs),
        "generator_path": generator_path,
        "verifier_path": verifier_path,
        "skill_cards_path": skill_cards_path,
        "skill_cards_total": skill_summary["total"],
        "skill_cards_promoted": skill_summary["promoted"],
        "training_source": training_source,
        "training_examples": len(skill_training_pairs) if skill_training_pairs else len(selected),
        "adapter_registry_path": config.adapter_registry_path,
        "verification": verification,
    }

    results_dir = Path(config.experiments_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(str(results_dir / f"cycle_{cycle}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"CYCLE {cycle} COMPLETE")
    print(f"  Generator: {generator_path}")
    print(f"  Verifier:  {verifier_path}")
    print(f"  Skill cards: {skill_summary['promoted']}/{skill_summary['total']} promoted")
    print(f"  Training source: {training_source} ({results['training_examples']} examples)")
    if verification:
        print(f"  Pass rate: {verification['mean']:.1%}")
    print("=" * 60)

    return results


def run_iterative_qstar(
    memory_bank_path: str,
    config: CogMemConfig,
    collect_fn=None,
    run_task_fn=None,
) -> list[dict]:
    """Run multiple Q-STaR cycles until plateau.

    Args:
        memory_bank_path: Path to memory bank JSON.
        config: CogMemConfig.
        collect_fn: Optional callable(generator_path, verifier_path) -> None.
            Collects new episodes using improved model.
        run_task_fn: Optional callable for evaluation.

    Returns:
        Learning curve: list of per-cycle results.
    """
    learning_curve = []

    for cycle in range(config.max_cycles):
        results = run_qstar_cycle(
            memory_bank_path, config, cycle=cycle, run_task_fn=run_task_fn
        )
        learning_curve.append(results)

        # Collect new episodes with improved model
        if collect_fn is not None and results.get("generator_path"):
            print(f"\n  Collecting new episodes with cycle-{cycle} model...")
            collect_fn(results["generator_path"], results.get("verifier_path"))

        # Check plateau
        if len(learning_curve) >= 3:
            recent = [
                r.get("verification", {}).get("mean")
                for r in learning_curve[-3:]
            ]
            valid = [v for v in recent if v is not None]
            if len(valid) >= 3 and valid[-1] - valid[0] < config.plateau_threshold:
                print(f"\n  Plateau detected (< {config.plateau_threshold:.0%} "
                      f"improvement over 3 cycles).")
                break

    # Save learning curve
    results_dir = Path(config.experiments_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(str(results_dir / "iterative_qstar_results.json"), "w") as f:
        json.dump(learning_curve, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Q-STaR ITERATIVE RESULTS")
    print("=" * 60)
    print(f"{'Cycle':<8} {'Episodes':<12} {'Pairs':<10} {'Generator':<15}")
    print("-" * 45)
    for r in learning_curve:
        print(f"{r['cycle']:<8} {r['total_episodes']:<12} "
              f"{r['preference_pairs']:<10} {r['generator_path']}")

    return learning_curve


# -------------------------------------------------------------------------
# Legacy pipelines (Experiment 1 & 2)
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

    training_pairs = prepare_training_dataset(
        selected,
        replay_buffer=replay_buffer,
        config=config,
    )
    jsonl_dir = f"{config.logs_dir}/jsonl"
    jsonl_path = save_as_jsonl(training_pairs, f"{jsonl_dir}/{policy_name}.jsonl")

    if config.lora_provider == "local":
        train_result = train_lora_local(
            jsonl_path,
            config,
            policy_name=policy_name,
            training_manifest_ids=sorted(
                {ep.get("manifest_id") for ep in selected if ep.get("manifest_id")}
            ),
            compatible_families=sorted(
                {ep.get("task_type") for ep in selected if ep.get("task_type")}
            ),
            registry_path=config.adapter_registry_path,
            adapter_role="legacy_policy",
            dev_gain=0.0,
        )
    else:
        train_result = train_lora_together(
            jsonl_path,
            config,
            policy_name=policy_name,
            training_manifest_ids=sorted(
                {ep.get("manifest_id") for ep in selected if ep.get("manifest_id")}
            ),
            compatible_families=sorted(
                {ep.get("task_type") for ep in selected if ep.get("task_type")}
            ),
            registry_path=config.adapter_registry_path,
            adapter_role="legacy_policy",
            dev_gain=0.0,
        )

    seed_results = []
    if run_task_fn is not None:
        for seed in config.eval_seeds:
            r = run_verification_single_seed(holdout_episodes, run_task_fn, seed)
            seed_results.append(r)

    verification = aggregate_seed_results(seed_results) if seed_results else {}
    adapter_dir = train_result.get("adapter_dir")
    if adapter_dir:
        registry = AdapterRegistry.load(config.adapter_registry_path)
        record = registry.find_by_path(adapter_dir)
        if record is not None:
            registry.update(
                record.adapter_id,
                dev_gain=verification.get("mean", 0.0) if verification else 0.0,
                metadata={
                    **record.metadata,
                    "verification": verification,
                },
            )
            registry.save(config.adapter_registry_path)

    result = {
        "policy": policy_name,
        "episodes_selected": len(selected),
        "training_pairs": len(training_pairs),
        "episode_helpfulness_mean": mean([get_episode_helpfulness(ep) for ep in selected]) if selected else 0,
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

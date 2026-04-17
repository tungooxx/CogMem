"""CLI runner for the research-only patch experiment."""

from __future__ import annotations

import argparse
import json

from cogmem.patches.experiment import (
    PatchExperimentConfig,
    load_patch_runtime,
    load_patch_tasks,
    prepare_patch_task_split,
    run_patch_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CogMem patch experiment without the notebook.")
    parser.add_argument("--tasks-jsonl", default="", help="Optional local BigCodeBench JSONL path.")
    parser.add_argument("--dataset-version", default="v0.1.4")
    parser.add_argument("--hard-only", action="store_true")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--memory-dir", default="results/cluster_memories")
    parser.add_argument("--train-task-count", type=int, default=500)
    parser.add_argument("--reset-progress", action="store_true")
    parser.add_argument("--force-rerun-eval", action="store_true")
    args = parser.parse_args()

    tasks = load_patch_tasks(
        task_jsonl_path=args.tasks_jsonl or None,
        version=args.dataset_version,
        hard_only=args.hard_only,
    )
    train_tasks, eval_tasks = prepare_patch_task_split(tasks, args.train_task_count)

    config = PatchExperimentConfig(
        memory_dir=args.memory_dir,
        model_name=args.model_name,
        train_task_count=args.train_task_count,
    )
    base_model, tokenizer, embedder = load_patch_runtime(model_name=args.model_name)
    results = run_patch_experiment(
        train_tasks,
        eval_tasks,
        base_model,
        tokenizer,
        embedder,
        config=config,
        reset_progress=args.reset_progress,
        force_rerun_eval=args.force_rerun_eval,
    )
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()

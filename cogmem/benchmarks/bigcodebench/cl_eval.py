"""Split-aware evaluation helpers for BigCodeBench continual-learning runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cogmem.benchmarks.bigcodebench.dataset import filter_by_split


OFFICIAL_BASELINE_LABEL = "Official BigCodeBench baseline"
CL_LABEL = "BigCodeBench-CL"


@dataclass
class EvalCase:
    task_id: str
    split_name: str | None = None
    cold_pass: bool = False
    routed_pass: bool = False
    pass_at_3: bool | None = None
    retrieval_used: bool = False
    abstained: bool = False
    route_kind: str = "cold"

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_eval_label(label: str | None, *, official: bool = False) -> str:
    if official:
        return OFFICIAL_BASELINE_LABEL
    if not label or label.lower() == "bigcodebench_cl":
        return CL_LABEL
    return label


def summarize_eval_cases(
    cases: list[EvalCase | dict],
    *,
    benchmark_label: str = CL_LABEL,
) -> dict:
    normalized_cases = [case.to_dict() if isinstance(case, EvalCase) else dict(case) for case in cases]
    total = len(normalized_cases)
    if total == 0:
        return {
            "benchmark_label": normalize_eval_label(benchmark_label),
            "total": 0,
            "cold_pass_at_1": 0.0,
            "routed_pass_at_1": 0.0,
            "pass_at_3": 0.0,
            "abstention_rate": 0.0,
            "negative_transfer_rate": 0.0,
            "retrieval_helpfulness_rate": 0.0,
            "consolidation_gain": 0.0,
        }

    cold_successes = sum(1 for case in normalized_cases if case.get("cold_pass"))
    routed_successes = sum(1 for case in normalized_cases if case.get("routed_pass"))
    pass_at_3_hits = sum(
        1 for case in normalized_cases
        if case.get("pass_at_3", case.get("routed_pass"))
    )
    abstentions = sum(1 for case in normalized_cases if case.get("abstained"))
    retrieval_used = [case for case in normalized_cases if case.get("retrieval_used")]
    helpful_retrievals = [
        case for case in retrieval_used
        if case.get("routed_pass") and not case.get("cold_pass")
    ]
    harmful_retrievals = [
        case for case in normalized_cases
        if case.get("cold_pass") and not case.get("routed_pass")
    ]

    return {
        "benchmark_label": normalize_eval_label(benchmark_label),
        "total": total,
        "cold_pass_at_1": cold_successes / total,
        "routed_pass_at_1": routed_successes / total,
        "pass_at_3": pass_at_3_hits / total,
        "abstention_rate": abstentions / total,
        "negative_transfer_rate": len(harmful_retrievals) / total,
        "retrieval_helpfulness_rate": (
            len(helpful_retrievals) / len(retrieval_used)
            if retrieval_used else 0.0
        ),
        "consolidation_gain": (routed_successes - cold_successes) / total,
    }


def run_cl_eval(
    tasks: list[dict],
    evaluate_case_fn,
    *,
    split_name: str | None = None,
    benchmark_label: str = CL_LABEL,
) -> dict:
    eval_tasks = filter_by_split(tasks, split_name) if split_name else list(tasks)
    cases = []
    for task in eval_tasks:
        case = evaluate_case_fn(task)
        if isinstance(case, EvalCase):
            cases.append(case)
        else:
            payload = dict(case)
            payload.setdefault("task_id", task.get("task_id", ""))
            payload.setdefault("split_name", task.get("split_name"))
            cases.append(EvalCase(**payload))
    summary = summarize_eval_cases(cases, benchmark_label=benchmark_label)
    summary["split_name"] = split_name
    summary["cases"] = [case.to_dict() for case in cases]
    return summary

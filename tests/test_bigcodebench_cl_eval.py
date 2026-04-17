from cogmem.benchmarks.bigcodebench.cl_eval import (
    CL_LABEL,
    OFFICIAL_BASELINE_LABEL,
    EvalCase,
    normalize_eval_label,
    run_cl_eval,
    summarize_eval_cases,
)


def test_normalize_eval_label():
    assert normalize_eval_label(None) == CL_LABEL
    assert normalize_eval_label("bigcodebench_cl") == CL_LABEL
    assert normalize_eval_label("custom") == "custom"
    assert normalize_eval_label("anything", official=True) == OFFICIAL_BASELINE_LABEL


def test_summarize_eval_cases_computes_core_metrics():
    cases = [
        EvalCase(task_id="t1", cold_pass=False, routed_pass=True, retrieval_used=True, abstained=False),
        EvalCase(task_id="t2", cold_pass=True, routed_pass=False, retrieval_used=True, abstained=False),
        EvalCase(task_id="t3", cold_pass=False, routed_pass=False, retrieval_used=False, abstained=True),
        EvalCase(task_id="t4", cold_pass=True, routed_pass=True, pass_at_3=True, retrieval_used=True, abstained=False),
    ]

    summary = summarize_eval_cases(cases)

    assert summary["benchmark_label"] == CL_LABEL
    assert summary["cold_pass_at_1"] == 0.5
    assert summary["routed_pass_at_1"] == 0.5
    assert summary["abstention_rate"] == 0.25
    assert summary["negative_transfer_rate"] == 0.25
    assert round(summary["retrieval_helpfulness_rate"], 3) == round(1 / 3, 3)
    assert summary["consolidation_gain"] == 0.0


def test_run_cl_eval_filters_by_split():
    tasks = [
        {"task_id": "t_train", "split_name": "train"},
        {"task_id": "t_test", "split_name": "test"},
    ]

    summary = run_cl_eval(
        tasks,
        lambda task: {"task_id": task["task_id"], "cold_pass": False, "routed_pass": task["task_id"] == "t_test"},
        split_name="test",
    )

    assert summary["split_name"] == "test"
    assert summary["total"] == 1
    assert summary["cases"][0]["task_id"] == "t_test"

from cogmem.evaluation.compare import format_comparison_table, best_policy


def test_format_comparison_table():
    results = {
        "q_top_k": {"verification": {"mean": 0.65, "std": 0.05}, "episodes_selected": 5},
        "recency": {"verification": {"mean": 0.50, "std": 0.08}, "episodes_selected": 5},
        "random": {"verification": {"mean": 0.45, "std": 0.10}, "episodes_selected": 5},
    }
    table = format_comparison_table(results)
    assert "q_top_k" in table
    assert "0.65" in table


def test_best_policy():
    results = {
        "q_top_k": {"verification": {"mean": 0.65}},
        "recency": {"verification": {"mean": 0.50}},
        "random": {"verification": {"mean": 0.45}},
    }
    assert best_policy(results) == "q_top_k"

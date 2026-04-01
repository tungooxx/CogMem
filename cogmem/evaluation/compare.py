import json
from pathlib import Path


def format_comparison_table(results: dict) -> str:
    header = f"{'Policy':<15} {'Success Rate':<15} {'Std':<10} {'Episodes':<10}"
    sep = "-" * 50
    lines = [header, sep]
    for policy, data in sorted(results.items()):
        v = data.get("verification", {})
        rate = v.get("mean", 0)
        std = v.get("std", 0)
        n_eps = data.get("episodes_selected", 0)
        lines.append(f"{policy:<15} {rate:<15.2f} {std:<10.3f} {n_eps:<10}")
    return "\n".join(lines)


def best_policy(results: dict) -> str:
    return max(
        results,
        key=lambda p: results[p].get("verification", {}).get("mean", 0),
    )


def save_comparison(results: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

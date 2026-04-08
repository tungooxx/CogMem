"""Generate experience summaries from episodic memory.

Instead of SFT on raw solutions, extract RULES from experience:
- What patterns work?
- What mistakes to avoid?
- What imports/functions are useful?

The model reflects on its own experiences and produces summaries
that become SFT training data. This is "memory becoming knowledge."
"""

import json
import re
from collections import Counter


def categorize_domain(code: str, task_desc: str) -> str:
    """Categorize a task by its primary library/domain."""
    text = (code + " " + task_desc).lower()

    domains = {
        "pandas": ["pandas", "pd.", "dataframe", "read_csv", "groupby", "merge"],
        "numpy": ["numpy", "np.", "ndarray", "array(", "linspace", "reshape"],
        "matplotlib": ["matplotlib", "plt.", "plot(", "figure(", "subplot"],
        "file_io": ["open(", "os.path", "pathlib", "shutil", "glob"],
        "regex": ["re.", "re.search", "re.match", "re.sub", "re.findall"],
        "datetime": ["datetime", "timedelta", "strftime", "strptime"],
        "json_xml": ["json.", "xml.", "yaml.", "csv."],
        "math": ["math.", "statistics", "scipy", "random."],
        "collections": ["counter", "defaultdict", "ordereddict", "deque"],
        "itertools": ["itertools", "permutations", "combinations", "chain"],
        "sklearn": ["sklearn", "train_test_split", "model.fit", "predict"],
        "subprocess": ["subprocess", "os.system", "popen"],
        "string": ["string.", ".split(", ".join(", ".replace(", ".strip("],
        "crypto": ["hashlib", "hmac", "base64"],
    }

    scores = {}
    for domain, keywords in domains.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[domain] = score

    return max(scores, key=scores.get) if scores else "general"


def extract_import_patterns(episodes: list[dict]) -> list[str]:
    """Find the most common import statements from successful episodes."""
    imports = Counter()
    for ep in episodes:
        code = ep.get("final_code") or ep.get("generated_code") or ""
        for line in code.split("\n"):
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                imports[line] += 1

    return [imp for imp, count in imports.most_common(10) if count >= 2]


def extract_common_errors(episodes: list[dict]) -> list[str]:
    """Find the most common error types from failed episodes."""
    errors = Counter()
    for ep in episodes:
        err = ep.get("error") or ""
        # Extract error type
        match = re.search(r"(\w+Error)", err)
        if match:
            errors[match.group(1)] += 1
        # Extract specific message
        match = re.search(r"(\w+Error: .{10,60})", err)
        if match:
            errors[match.group(1)] += 1

    return [err for err, count in errors.most_common(5) if count >= 2]


def extract_function_patterns(episodes: list[dict]) -> list[str]:
    """Find commonly used function calls in successful code."""
    calls = Counter()
    for ep in episodes:
        code = ep.get("final_code") or ep.get("generated_code") or ""
        # Find method calls like .sort_values(, pd.read_csv(, etc.
        for match in re.findall(r"[\w.]+\(", code):
            if len(match) > 3 and not match.startswith("def "):
                calls[match] += 1

    return [call for call, count in calls.most_common(10) if count >= 3]


def build_experience_summary(
    domain: str,
    success_episodes: list[dict],
    failure_episodes: list[dict],
) -> str:
    """Build a structured experience summary for a domain.

    This is what the model "learned" from its experiences.
    """
    total = len(success_episodes) + len(failure_episodes)
    success_rate = len(success_episodes) / max(total, 1)

    imports = extract_import_patterns(success_episodes)
    errors = extract_common_errors(failure_episodes)
    patterns = extract_function_patterns(success_episodes)

    lines = []
    lines.append(f"Based on your experience with {domain} tasks:")
    lines.append(f"You attempted {total} tasks, succeeded on {len(success_episodes)} ({success_rate:.0%}).")
    lines.append("")

    if imports:
        lines.append("IMPORTS THAT WORK:")
        for imp in imports[:5]:
            lines.append(f"  {imp}")
        lines.append("")

    if patterns:
        lines.append("USEFUL FUNCTIONS:")
        for pat in patterns[:5]:
            lines.append(f"  {pat}")
        lines.append("")

    if errors:
        lines.append("COMMON MISTAKES TO AVOID:")
        for err in errors[:5]:
            lines.append(f"  - {err}")
        lines.append("")

    # Add 1-2 short successful code examples
    if success_episodes:
        lines.append("EXAMPLE OF GOOD CODE:")
        example = success_episodes[0]
        code = example.get("final_code") or example.get("generated_code") or ""
        # Truncate to ~15 lines
        code_lines = code.split("\n")[:15]
        lines.append("```python")
        lines.extend(code_lines)
        lines.append("```")

    return "\n".join(lines)


def generate_llm_summary(
    domain: str,
    success_episodes: list[dict],
    failure_episodes: list[dict],
    llm_client,
    model_name: str = "qwen2.5:3b",
) -> str:
    """Use the LLM itself to reflect on its own experience.

    The model generates a summary of what it learned from
    its successes and failures in a domain.
    """
    successes_text = ""
    for ep in success_episodes[:3]:
        code = ep.get("final_code") or ep.get("generated_code") or ""
        desc = ep.get("task_description", "")[:100]
        successes_text += f"\nTask: {desc}\nCode:\n{code[:300]}\n"

    failures_text = ""
    for ep in failure_episodes[:3]:
        code = ep.get("generated_code") or ""
        desc = ep.get("task_description", "")[:100]
        err = (ep.get("error") or "")[:200]
        failures_text += f"\nTask: {desc}\nCode:\n{code[:200]}\nError: {err}\n"

    prompt = f"""You attempted {len(success_episodes) + len(failure_episodes)} {domain} coding tasks.
You succeeded on {len(success_episodes)} and failed on {len(failure_episodes)}.

Here are some of your successes:
{successes_text}

Here are some of your failures with their errors:
{failures_text}

Based on this experience, write a brief guide for yourself:
1. What patterns work for {domain} tasks?
2. What mistakes should you avoid?
3. What libraries and functions are most useful?

Write this as practical rules you can follow. Be specific."""

    resp = llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are reflecting on your coding experience to extract useful rules."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0,
    )

    return resp.choices[0].message.content


def build_sft_data_with_summaries(
    episodes: list[dict],
    system_prompt: str,
    use_llm: bool = False,
    llm_client=None,
    model_name: str = "qwen2.5:3b",
) -> list[dict]:
    """Build SFT training data using experience summaries.

    Format: (system + experience_summary + task → solution)

    The model learns to produce correct code GIVEN its own
    experiential wisdom as context.
    """
    # Group episodes by domain
    by_domain: dict[str, dict[str, list]] = {}
    for ep in episodes:
        code = ep.get("final_code") or ep.get("generated_code") or ""
        desc = ep.get("task_description", "")
        domain = categorize_domain(code, desc)

        if domain not in by_domain:
            by_domain[domain] = {"success": [], "failure": []}

        if ep.get("success"):
            by_domain[domain]["success"].append(ep)
        else:
            by_domain[domain]["failure"].append(ep)

    # Generate summaries per domain
    summaries = {}
    for domain, eps in by_domain.items():
        if not eps["success"]:
            continue

        if use_llm and llm_client:
            summary = generate_llm_summary(
                domain, eps["success"], eps["failure"],
                llm_client, model_name,
            )
        else:
            summary = build_experience_summary(
                domain, eps["success"], eps["failure"],
            )
        summaries[domain] = summary
        print(f"  {domain}: {len(eps['success'])} successes, "
              f"{len(eps['failure'])} failures, summary={len(summary)} chars")

    # Build SFT data: (summary + task → solution)
    sft_data = []
    for ep in episodes:
        if not ep.get("success"):
            continue

        code = ep.get("final_code") or ep.get("generated_code") or ""
        desc = ep.get("task_description", "")
        if not code or not desc:
            continue

        domain = categorize_domain(code, desc)
        summary = summaries.get(domain, "")

        if summary:
            user_content = f"{summary}\n\nNow solve this task:\n{desc}"
        else:
            user_content = desc

        sft_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": code},
            ]
        })

    return sft_data

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

from cogmem.benchmarks.bigcodebench.prompts import format_for_training


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

    Q-values weight which experiences matter most:
    - High-Q episodes (proven useful to other tasks) → trusted patterns
    - Low-Q episodes (retrieved but didn't help) → less trusted
    - Episodes sorted by Q so high-Q patterns appear first
    """
    total = len(success_episodes) + len(failure_episodes)
    success_rate = len(success_episodes) / max(total, 1)

    # Sort by Q-value: high-Q episodes first (most trustworthy)
    sorted_success = sorted(
        success_episodes,
        key=lambda ep: ep.get("q_value", 0.5),
        reverse=True,
    )
    sorted_failures = sorted(
        failure_episodes,
        key=lambda ep: ep.get("q_visits", 0),
        reverse=True,
    )

    # High-Q successes are most reliable for pattern extraction
    high_q = [ep for ep in sorted_success if ep.get("q_value", 0.5) >= 0.6]
    imports = extract_import_patterns(high_q or sorted_success)
    errors = extract_common_errors(sorted_failures)
    patterns = extract_function_patterns(high_q or sorted_success)

    # Q-value stats
    q_vals = [ep.get("q_value", 0.5) for ep in success_episodes]
    avg_q = sum(q_vals) / len(q_vals) if q_vals else 0.5
    n_high_q = len(high_q)

    lines = []
    lines.append(f"Based on your experience with {domain} tasks:")
    lines.append(f"You attempted {total} tasks, succeeded on {len(success_episodes)} ({success_rate:.0%}).")
    if n_high_q > 0:
        lines.append(f"Of these, {n_high_q} solutions were proven helpful to other tasks (high Q-value).")
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

    # Add 1-2 short successful code examples (highest Q first)
    if sorted_success:
        lines.append("EXAMPLE OF GOOD CODE (highest Q):")
        example = sorted_success[0]
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

    Q-values guide the reflection:
    - High-Q successes → "proven reliable patterns"
    - Low-Q successes → "unreliable, warn about these"
    - Failures → "mistakes to avoid"
    """
    # Split by Q-value credibility
    high_q = sorted(
        [ep for ep in success_episodes if ep.get("q_value", 0.5) >= 0.6],
        key=lambda ep: ep.get("q_value", 0), reverse=True,
    )
    low_q = [ep for ep in success_episodes if ep.get("q_value", 0.5) < 0.4]

    def format_eps(eps, n=3, show_q=True):
        text = ""
        for ep in eps[:n]:
            code = ep.get("final_code") or ep.get("generated_code") or ""
            desc = ep.get("task_description", "")[:100]
            q = ep.get("q_value", 0.5)
            q_str = f" (Q={q:.2f})" if show_q else ""
            text += f"\nTask: {desc}{q_str}\nCode:\n{code[:300]}\n"
        return text or "\n(none)\n"

    def format_fails(eps, n=3):
        text = ""
        for ep in eps[:n]:
            code = ep.get("generated_code") or ""
            desc = ep.get("task_description", "")[:100]
            err = (ep.get("error") or "")[:200]
            text += f"\nTask: {desc}\nCode:\n{code[:200]}\nError: {err}\n"
        return text or "\n(none)\n"

    prompt = f"""You are reflecting on your coding experience with {domain} tasks.

YOUR MOST RELIABLE PATTERNS (high Q-value — these helped other tasks too):
{format_eps(high_q)}

PATTERNS THAT LOOK GOOD BUT DON'T GENERALIZE (low Q — didn't help others):
{format_eps(low_q)}

YOUR COMMON MISTAKES (failures with error messages):
{format_fails(failure_episodes)}

Based on this experience, write practical rules:
1. What patterns RELIABLY work? (from high-Q experiences)
2. What approaches should you AVOID? (from low-Q and failures)
3. What common errors must you watch for?

Focus on the HIGH-Q patterns — those are proven across multiple tasks.
Be skeptical of low-Q patterns — they may be one-time lucky fixes.
Write concise, actionable rules."""

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

        code = ep.get("final_code") or ep.get("generated_code") or ep.get("script") or ""
        desc = ep.get("task_description", "")
        if not code or not desc:
            continue

        domain = categorize_domain(code, desc)
        summary = summaries.get(domain, "")

        if summary:
            user_content = f"{summary}\n\nNow solve this task:\n{desc}"
        else:
            user_content = desc

        # Q-weighted copies: high-Q episodes get more training weight
        q = max(ep.get("q_value", 0.5), 0.0)
        copies = max(1, round(q * 5))

        pair = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": code},
            ]
        }
        for _ in range(copies):
            sft_data.append(pair)

    return sft_data

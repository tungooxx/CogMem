"""Build and validate procedural skill cards from episodic memory."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from cogmem.consolidation.experience_summary import categorize_domain, extract_common_errors, extract_function_patterns
from cogmem.consolidation.select import filter_manifest_eligible
from cogmem.memory.episodic_store import infer_error_family
from cogmem.memory.memory_bank import MemoryBank
from cogmem.memory.schema import get_episode_helpfulness
from cogmem.memory.skill_store import SkillStore, normalize_skill_card


TRIGGER_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "write", "function", "return",
    "using", "into", "your", "task", "python", "code", "given", "input", "output", "list",
    "you", "starting", "task_func", "import", "contained", "complete", "prompt", "answer",
    "solution", "implement", "generate", "generated", "following", "provided", "script",
    "assistant", "please", "should", "after", "before",
}

GENERIC_TASK_TYPES = {"general", "bigcodebench"}

FEATURE_KEYWORDS = {
    "pandas": ["groupby", "merge", "pivot", "read_csv", "columns", "fillna", "sort_values"],
    "numpy": ["reshape", "linspace", "ndarray", "array", "mean", "std"],
    "matplotlib": ["plot", "scatter", "hist", "bar", "subplot", "figure", "seaborn"],
    "file_io": ["glob", "pathlib", "shutil", "json", "yaml", "csv", "open"],
    "regex": ["regex", "findall", "search", "match", "sub"],
    "datetime": ["strftime", "strptime", "timedelta", "timezone"],
    "json_xml": ["json", "xml", "yaml", "csv"],
    "math": ["random", "statistics", "mean", "median", "sample", "normal"],
    "collections": ["counter", "defaultdict", "deque", "ordereddict"],
    "itertools": ["permutations", "combinations", "chain", "product"],
    "subprocess": ["subprocess", "popen", "system"],
    "string": ["split", "join", "replace", "strip", "format"],
    "crypto": ["hashlib", "hmac", "base64"],
}

PLAN_STEP_STOPWORDS = {
    "code:",
    "python:",
    "```",
    "```python",
    "response:",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _description_tokens(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-zA-Z_]{3,}", (text or "").lower())
        if token not in TRIGGER_STOPWORDS
    ]


def _episode_domain(episode: dict) -> str:
    code = episode.get("final_code") or episode.get("generated_code") or episode.get("script") or ""
    desc = episode.get("task_description", "")
    return categorize_domain(code, desc)


def _feature_bucket(episode: dict, domain: str | None = None) -> str:
    target_domain = domain or _episode_domain(episode)
    text = " ".join(
        [
            episode.get("task_description", ""),
            episode.get("final_code", "") or episode.get("generated_code", "") or episode.get("script", ""),
            " ".join(episode.get("libs", []) or []),
            episode.get("entry_point", "") or "",
        ]
    ).lower()
    for keyword in FEATURE_KEYWORDS.get(target_domain, []):
        if keyword in text:
            return keyword.replace(".", "_").replace("(", "")

    tokens = [token for token in _description_tokens(episode.get("task_description", "")) if token != target_domain]
    return tokens[0] if tokens else target_domain


def _group_key(episode: dict) -> str:
    base_task_type = str(episode.get("task_type", "general") or "general")
    domain = _episode_domain(episode)
    feature = _feature_bucket(episode, domain)

    if base_task_type in GENERIC_TASK_TYPES:
        if domain != "general" and feature and feature != domain:
            return f"{domain}:{feature}"
        return domain if domain != "general" else base_task_type

    return base_task_type


def _derive_triggers(episodes: list[dict], limit: int) -> list[str]:
    counts = Counter()
    libs = Counter()
    for ep in episodes:
        counts.update(set(_description_tokens(ep.get("task_description", ""))))
        libs.update(ep.get("libs", []) or [])
        if ep.get("task_type") and ep.get("task_type") not in GENERIC_TASK_TYPES:
            counts[ep["task_type"]] += 2
    ordered = [token for token, _ in counts.most_common(limit)]
    for lib, _ in libs.most_common(limit):
        if lib and lib not in TRIGGER_STOPWORDS and lib not in ordered:
            ordered.append(lib)
        if len(ordered) >= limit:
            break
    return ordered[:limit]


def _derive_plan_steps(success_episodes: list[dict], limit: int) -> list[str]:
    steps = Counter()
    for ep in success_episodes:
        script = ep.get("script") or ""
        for raw_line in script.splitlines():
            line = re.sub(r"^\d+\.\s*", "", raw_line.strip())
            lowered = line.lower()
            if (
                4 <= len(line) <= 100
                and lowered not in PLAN_STEP_STOPWORDS
                and not lowered.startswith("```")
                and not lowered.startswith("here")
            ):
                steps[line] += 1
    if steps:
        return [step for step, _ in steps.most_common(limit)]
    return extract_function_patterns(success_episodes)[:limit]


def _derive_anti_patterns(failure_episodes: list[dict], limit: int) -> list[str]:
    patterns = Counter()
    for ep in failure_episodes:
        error_family = ep.get("error_family") or infer_error_family(ep.get("error"))
        if error_family:
            patterns[f"avoid {error_family} failure modes"] += 2
    for error in extract_common_errors(failure_episodes)[:limit]:
        patterns[error] += 1
    return [pattern for pattern, _ in patterns.most_common(limit)]


def proceduralize_episodes(episodes: list[dict], config=None) -> list[dict]:
    """Turn repeated successful episodes into candidate procedural skill cards."""
    eligible = filter_manifest_eligible(episodes, config) if config is not None else list(episodes)
    min_evidence = int(getattr(config, "skill_min_evidence", 3))
    trigger_limit = int(getattr(config, "skill_trigger_limit", 6))
    plan_limit = int(getattr(config, "skill_plan_limit", 5))

    grouped_success: dict[str, list[dict]] = defaultdict(list)
    grouped_all: dict[str, list[dict]] = defaultdict(list)
    for ep in eligible:
        key = _group_key(ep)
        grouped_all[key].append(ep)
        if ep.get("success"):
            grouped_success[key].append(ep)

    cards = []
    for key, successes in grouped_success.items():
        if len(successes) < min_evidence:
            continue
        task_type = key
        all_group = grouped_all[key]
        failures = [ep for ep in all_group if not ep.get("success")]
        domain_counts = Counter(_episode_domain(ep) for ep in successes)
        domain = domain_counts.most_common(1)[0][0] if domain_counts else "general"
        manifests = sorted({ep.get("manifest_id") for ep in all_group if ep.get("manifest_id")})
        error_family = None
        if failures:
            error_counts = Counter(ep.get("error_family") for ep in failures if ep.get("error_family"))
            if error_counts:
                error_family = error_counts.most_common(1)[0][0]
        card = {
            "task_type": task_type,
            "domain": domain,
            "error_family": error_family,
            "triggers": _derive_triggers(successes, trigger_limit),
            "plan_steps": _derive_plan_steps(successes, plan_limit),
            "anti_patterns": _derive_anti_patterns(failures, plan_limit),
            "validation": {
                "status": "candidate",
                "recipe_kinds": sorted(
                    {
                        (ep.get("validation_recipe") or {}).get("kind", "episode_outcome")
                        for ep in all_group
                    }
                ),
            },
            "evidence_episode_ids": [ep["episode_id"] for ep in successes],
            "manifest_ids": manifests,
            "transfer_gain": mean(get_episode_helpfulness(ep, 0.0) for ep in successes),
            "confidence": _clamp01(mean(get_episode_helpfulness(ep, 0.0) for ep in successes)),
            "source_episode_count": len(successes),
            "status": "candidate",
        }
        cards.append(normalize_skill_card(card, copy_card=True))

    return cards


def _card_matches_episode(card: dict, episode: dict) -> bool:
    if episode.get("episode_id") in set(card.get("evidence_episode_ids", [])):
        return False
    if card.get("task_type") and _group_key(episode) == card["task_type"]:
        return True
    code = episode.get("final_code") or episode.get("generated_code") or episode.get("script") or ""
    domain = categorize_domain(code, episode.get("task_description", ""))
    episode_tokens = set(_description_tokens(episode.get("task_description", "")))
    trigger_tokens = {token.lower() for token in card.get("triggers", [])}
    if card.get("domain") and domain == card["domain"] and trigger_tokens and (episode_tokens & trigger_tokens):
        return True
    return bool(episode_tokens & trigger_tokens)


def validate_skill_card_record(card: dict, dev_episodes: list[dict], config=None) -> dict:
    """Validate one candidate skill card on held-out or out-of-cluster episodes."""
    eligible = filter_manifest_eligible(dev_episodes, config) if config is not None else list(dev_episodes)
    min_matches = int(getattr(config, "skill_validation_min_matches", 2))
    min_transfer_gain = float(getattr(config, "skill_min_transfer_gain", 0.05))
    min_confidence = float(getattr(config, "skill_confidence_threshold", 0.6))

    matches = [ep for ep in eligible if _card_matches_episode(card, ep)]
    baseline_pool = [ep for ep in eligible if ep.get("episode_id") not in set(card.get("evidence_episode_ids", []))]
    match_helpfulness = [get_episode_helpfulness(ep, 0.0) for ep in matches]
    baseline_helpfulness = [get_episode_helpfulness(ep, 0.0) for ep in baseline_pool]
    success_rate = mean(1.0 if ep.get("success") else 0.0 for ep in matches) if matches else 0.0
    negative_transfer_rate = mean(0.0 if ep.get("success") else 1.0 for ep in matches) if matches else 0.0
    transfer_gain = (
        (mean(match_helpfulness) - mean(baseline_helpfulness))
        if match_helpfulness and baseline_helpfulness
        else 0.0
    )
    support_factor = min(1.0, len(matches) / max(min_matches, 1))
    confidence = _clamp01(
        0.35 * support_factor
        + 0.35 * success_rate
        + 0.20 * max(transfer_gain, 0.0)
        + 0.10 * card.get("confidence", 0.0)
        - 0.25 * negative_transfer_rate
    )
    status = "promoted" if (
        len(matches) >= min_matches
        and transfer_gain >= min_transfer_gain
        and confidence >= min_confidence
    ) else "candidate"

    updated = dict(card)
    updated["transfer_gain"] = transfer_gain
    updated["confidence"] = confidence
    updated["negative_transfer_rate"] = negative_transfer_rate
    updated["status"] = status
    updated["validation"] = {
        **dict(card.get("validation", {})),
        "status": status,
        "matched_episodes": len(matches),
        "success_rate": success_rate,
        "transfer_gain": transfer_gain,
        "negative_transfer_rate": negative_transfer_rate,
        "validated_task_ids": sorted({ep.get("task_id") for ep in matches if ep.get("task_id")})[:20],
    }
    return normalize_skill_card(updated, copy_card=True)


def validate_skill_cards(cards: list[dict], dev_episodes: list[dict], config=None) -> list[dict]:
    return [validate_skill_card_record(card, dev_episodes, config=config) for card in cards]


def build_skill_cards(
    episodes: list[dict],
    dev_episodes: list[dict],
    config=None,
    output_path: str | None = None,
) -> SkillStore:
    cards = proceduralize_episodes(episodes, config=config)
    validated = validate_skill_cards(cards, dev_episodes, config=config)
    store = SkillStore(validated)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        store.save(output_path)
    return store


def validate_skill_card(
    card_id: str,
    dev_task_ids: list[str],
    skill_store_path: str,
    memory_bank_path: str,
    config=None,
) -> dict:
    """Validate one saved skill card against a list of dev task ids."""
    store = SkillStore.load(skill_store_path)
    card = store.get(card_id)
    if card is None:
        raise KeyError(f"Unknown skill card: {card_id}")
    bank = MemoryBank.load(memory_bank_path)
    dev_id_set = set(dev_task_ids)
    dev_episodes = [ep for ep in bank if ep.get("task_id") in dev_id_set]
    validated = validate_skill_card_record(card, dev_episodes, config=config)
    store.update(card_id, **{k: v for k, v in validated.items() if k != "skill_id"})
    store.save(skill_store_path)
    return validated

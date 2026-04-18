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
    "self", "def", "data", "each", "contains",
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
    "thought:",
}


def _add_unique(items: list[str], value: str | None) -> None:
    text = str(value or "").strip()
    if text and text not in items:
        items.append(text)


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


def task_to_skill_record(task: dict) -> dict:
    return {
        "task_description": task.get("instruct_prompt") or task.get("complete_prompt") or "",
        "libs": list(task.get("libs", []) or []),
        "entry_point": task.get("entry_point", "") or "",
        "task_type": task.get("task_type", "bigcodebench") or "bigcodebench",
        "error": task.get("error"),
    }


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


def _card_feature(card: dict) -> str | None:
    task_type = str(card.get("task_type", "") or "")
    if ":" in task_type:
        return task_type.split(":", 1)[1]
    if task_type and task_type not in GENERIC_TASK_TYPES and task_type != card.get("domain"):
        return task_type
    return None


def _episode_hint_tokens(episode: dict) -> set[str]:
    tokens = set(_description_tokens(episode.get("task_description", "")))
    for lib in episode.get("libs", []) or []:
        for token in re.findall(r"[a-zA-Z_]{3,}", str(lib).lower()):
            if token not in TRIGGER_STOPWORDS:
                tokens.add(token)
    entry_point = str(episode.get("entry_point", "") or "").lower()
    for token in re.findall(r"[a-zA-Z_]{3,}", entry_point):
        if token not in TRIGGER_STOPWORDS:
            tokens.add(token)
    return tokens


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


def _derive_plan_steps(
    success_episodes: list[dict],
    failure_episodes: list[dict],
    *,
    domain: str,
    feature: str,
    error_family: str | None,
    limit: int,
) -> list[str]:
    steps: list[str] = []
    if failure_episodes or error_family:
        _add_unique(steps, "inspect the traceback or failing assertion before rewriting logic")
    if error_family == "ModuleNotFoundError":
        _add_unique(steps, "check import paths and dependency availability before changing the algorithm")
    if error_family == "FileNotFoundError":
        _add_unique(steps, "verify file paths and existence before debugging downstream logic")
    if error_family == "KeyError":
        _add_unique(steps, "validate required keys or column names before indexing into data structures")
    if error_family == "AssertionError":
        _add_unique(steps, "reproduce the expected output and compare intermediate values before broad rewrites")

    domain_defaults = {
        "pandas": [
            "validate dataframe columns and dtypes before aggregation or joins",
            "check groupby or merge keys before transforming the dataframe",
        ],
        "numpy": [
            "check array shape and dtype before vectorized operations",
            "verify broadcasting assumptions before changing numeric logic",
        ],
        "matplotlib": [
            "verify figure and axes construction before styling or saving plots",
            "check that plotted series align in length before adjusting formatting",
        ],
        "file_io": [
            "validate paths, permissions, and file existence before reading or writing",
            "confirm serialization format before parsing persisted data",
        ],
        "math": [
            "check numeric bounds and randomization assumptions before tuning formulas",
            "verify edge cases for empty inputs and rounding before optimizing logic",
        ],
        "string": [
            "normalize delimiters and whitespace before joining or splitting text",
            "check encoding and character assumptions before rewriting parsing logic",
        ],
        "crypto": [
            "verify byte or string encoding before hashing, signing, or decoding",
        ],
    }
    for item in domain_defaults.get(domain, ["reproduce the failing case with minimal assumptions before broad rewrites"]):
        _add_unique(steps, item)

    feature_hints = {
        "columns": "validate referenced dataframe columns before aggregation",
        "groupby": "verify grouping keys and aggregation targets before refactoring",
        "merge": "confirm join keys and expected cardinality before changing merge logic",
        "plot": "check axes state and plotted series before changing rendering code",
        "read_csv": "verify CSV parsing options and required columns before changing transformation logic",
        "glob": "inspect path patterns and matched files before changing traversal logic",
        "json": "validate expected JSON schema before rewriting parsing code",
        "reshape": "check array dimensions before reshaping or flattening data",
        "ndarray": "validate array shape and indexing assumptions before numeric changes",
        "random": "fix the expected random seed or bounds before changing generation logic",
    }
    if feature in feature_hints:
        _add_unique(steps, feature_hints[feature])

    if any("import " in (ep.get("final_code") or ep.get("generated_code") or ep.get("script") or "") for ep in success_episodes):
        _add_unique(steps, "check imports and setup steps before debugging the core function body")

    return steps[:limit]


def _derive_activation_conditions(
    success_episodes: list[dict],
    *,
    task_type: str,
    domain: str,
    feature: str,
    error_family: str | None,
    triggers: list[str],
    limit: int,
) -> list[str]:
    conditions: list[str] = []
    named_triggers = [token for token in triggers if token not in {domain, feature}][:4]
    if domain != "general":
        trigger_text = ", ".join([domain] + ([feature] if feature and feature != domain else []) + named_triggers[:2])
        _add_unique(conditions, f"the task prompt mentions {trigger_text}")
    if error_family:
        _add_unique(conditions, f"the failure mode or traceback points to {error_family}")
    if task_type and task_type not in GENERIC_TASK_TYPES:
        _add_unique(conditions, f"the task belongs to the {task_type} pattern")
    _add_unique(conditions, "the same high-level bug pattern appears across multiple similar tasks")
    return conditions[:limit]


def _derive_stop_conditions(
    *,
    domain: str,
    feature: str,
    error_family: str | None,
    anti_patterns: list[str],
    limit: int,
) -> list[str]:
    conditions: list[str] = []
    if error_family == "ModuleNotFoundError":
        _add_unique(conditions, "stop and reassess if the required library is unavailable in the environment")
    if error_family == "AssertionError":
        _add_unique(conditions, "stop and reassess if intermediate values already match the intended logic")
    if feature == "columns":
        _add_unique(conditions, "stop and reassess if the required dataframe columns are genuinely absent")
    if domain == "file_io":
        _add_unique(conditions, "stop and reassess if the issue is missing input data rather than broken logic")
    if domain == "matplotlib":
        _add_unique(conditions, "stop and reassess if the mismatch comes from task expectations rather than plot construction")
    for pattern in anti_patterns[: max(limit - len(conditions), 0)]:
        suffix = pattern.replace("avoid ", "")
        _add_unique(conditions, f"stop and reassess if you are still triggering {suffix}")
    return conditions[:limit]


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
        feature = _feature_bucket(successes[0], domain) if successes else domain
        manifests = sorted({ep.get("manifest_id") for ep in all_group if ep.get("manifest_id")})
        evidence_task_ids = sorted({ep.get("task_id") for ep in successes if ep.get("task_id")})
        error_family = None
        if failures:
            error_counts = Counter(ep.get("error_family") for ep in failures if ep.get("error_family"))
            if error_counts:
                error_family = error_counts.most_common(1)[0][0]
        triggers = _derive_triggers(successes, trigger_limit)
        anti_patterns = _derive_anti_patterns(failures, plan_limit)
        card = {
            "task_type": task_type,
            "domain": domain,
            "error_family": error_family,
            "triggers": triggers,
            "activation_conditions": _derive_activation_conditions(
                successes,
                task_type=task_type,
                domain=domain,
                feature=feature,
                error_family=error_family,
                triggers=triggers,
                limit=plan_limit,
            ),
            "plan_steps": _derive_plan_steps(
                successes,
                failures,
                domain=domain,
                feature=feature,
                error_family=error_family,
                limit=plan_limit,
            ),
            "stop_conditions": _derive_stop_conditions(
                domain=domain,
                feature=feature,
                error_family=error_family,
                anti_patterns=anti_patterns,
                limit=plan_limit,
            ),
            "anti_patterns": anti_patterns,
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
            "evidence_task_ids": evidence_task_ids,
            "manifest_ids": manifests,
            "transfer_gain": mean(get_episode_helpfulness(ep, 0.0) for ep in successes),
            "confidence": _clamp01(mean(get_episode_helpfulness(ep, 0.0) for ep in successes)),
            "source_episode_count": len(successes),
            "distinct_task_count": len(evidence_task_ids),
            "status": "candidate",
        }
        cards.append(normalize_skill_card(card, copy_card=True))

    return cards


def _card_matches_episode(card: dict, episode: dict) -> bool:
    if episode.get("episode_id") in set(card.get("evidence_episode_ids", [])):
        return False
    episode_group = _group_key(episode)
    if card.get("task_type") and episode_group == card["task_type"]:
        return True
    code = episode.get("final_code") or episode.get("generated_code") or episode.get("script") or ""
    domain = categorize_domain(code, episode.get("task_description", ""))
    if card.get("domain") and card["domain"] != "general" and domain != card["domain"]:
        return False

    card_feature = _card_feature(card)
    episode_feature = _feature_bucket(episode, domain)
    if card_feature and episode_feature != card_feature:
        return False

    episode_tokens = _episode_hint_tokens(episode)
    trigger_tokens = {token.lower() for token in card.get("triggers", []) if token.lower() not in TRIGGER_STOPWORDS}
    overlap = episode_tokens & trigger_tokens

    if card_feature and card_feature in episode_tokens:
        return len(overlap) >= 1
    return len(overlap) >= 2


def validate_skill_card_record(card: dict, dev_episodes: list[dict], config=None) -> dict:
    """Validate one candidate skill card on held-out or out-of-cluster episodes."""
    eligible = filter_manifest_eligible(dev_episodes, config) if config is not None else list(dev_episodes)
    min_matches = int(getattr(config, "skill_validation_min_matches", 2))
    min_transfer_gain = float(getattr(config, "skill_min_transfer_gain", 0.05))
    min_confidence = float(getattr(config, "skill_confidence_threshold", 0.6))
    min_distinct_tasks = int(getattr(config, "skill_min_distinct_tasks", 2))

    matches = [ep for ep in eligible if _card_matches_episode(card, ep)]
    matched_task_ids = sorted({ep.get("task_id") for ep in matches if ep.get("task_id")})
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
    matched_task_count = len(matched_task_ids)
    support_factor = min(1.0, matched_task_count / max(min_matches, 1))
    evidence_task_count = int(card.get("distinct_task_count") or len(card.get("evidence_task_ids", []) or []))
    confidence = _clamp01(
        0.35 * support_factor
        + 0.35 * success_rate
        + 0.20 * max(transfer_gain, 0.0)
        + 0.10 * card.get("confidence", 0.0)
        - 0.25 * negative_transfer_rate
    )
    status = "promoted" if (
        matched_task_count >= min_matches
        and evidence_task_count >= min_distinct_tasks
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
        "matched_tasks": len(matched_task_ids),
        "evidence_task_count": evidence_task_count,
        "success_rate": success_rate,
        "transfer_gain": transfer_gain,
        "negative_transfer_rate": negative_transfer_rate,
        "validated_task_ids": matched_task_ids[:20],
    }
    return normalize_skill_card(updated, copy_card=True)


def _stop_condition_risk(card: dict, record_tokens: set[str], error_family: str) -> float:
    stop_text = " ".join(card.get("stop_conditions", []) or []).lower()
    if not stop_text:
        return 0.0
    risk = 0.0
    stop_tokens = {
        token
        for token in re.findall(r"[a-zA-Z_]{4,}", stop_text)
        if token not in TRIGGER_STOPWORDS
    }
    if error_family and error_family.lower() in stop_text:
        risk += 1.0
    risk += 0.15 * len(record_tokens & stop_tokens)
    return min(risk, 1.5)


def _runtime_route_disabled(card: dict, config=None) -> bool:
    runtime_stats = dict(card.get("runtime_stats", {}) or {})
    retrieved = int(runtime_stats.get("retrieved", 0) or 0)
    hurt = int(runtime_stats.get("hurt", 0) or 0)
    helped = int(runtime_stats.get("helped", 0) or 0)
    min_retrieved = int(getattr(config, "skill_runtime_disable_min_retrieved", 8))
    min_hurt = int(getattr(config, "skill_runtime_disable_min_hurt", 3))
    hurt_rate_threshold = float(getattr(config, "skill_runtime_disable_hurt_rate", 0.10))
    if retrieved < min_retrieved:
        return False
    hurt_rate = hurt / max(retrieved, 1)
    return helped == 0 and hurt >= min_hurt and hurt_rate >= hurt_rate_threshold


def _match_context(card: dict, record: dict) -> dict:
    record_group = _group_key(record)
    record_domain = _episode_domain(record)
    record_feature = _feature_bucket(record, record_domain)
    record_tokens = _episode_hint_tokens(record)
    error_family = str(record.get("error_family") or infer_error_family(record.get("error")) or "")
    card_feature = _card_feature(card)
    trigger_tokens = {token.lower() for token in card.get("triggers", []) if token.lower() not in TRIGGER_STOPWORDS}
    trigger_overlap = record_tokens & trigger_tokens
    exact_task_match = bool(card.get("task_type") == record_group)
    domain_match = bool(card.get("domain") == record_domain and record_domain != "general")
    feature_match = bool(card_feature and card_feature == record_feature)
    error_match = bool(card.get("error_family") and error_family and error_family == card.get("error_family"))
    strong_domain_match = domain_match and (feature_match or len(trigger_overlap) >= 2)
    strong_error_match = error_match and (domain_match or exact_task_match or len(trigger_overlap) >= 2)
    return {
        "record_group": record_group,
        "record_domain": record_domain,
        "record_feature": record_feature,
        "record_tokens": record_tokens,
        "error_family": error_family,
        "card_feature": card_feature,
        "trigger_overlap": trigger_overlap,
        "exact_task_match": exact_task_match,
        "domain_match": domain_match,
        "feature_match": feature_match,
        "error_match": error_match,
        "strong_domain_match": strong_domain_match,
        "strong_error_match": strong_error_match,
    }


def _skill_match_score(card: dict, record: dict) -> float:
    match = _match_context(card, record)
    score = 0.0

    if match["exact_task_match"]:
        score += 4.0
    if match["domain_match"]:
        score += 2.0

    if match["feature_match"]:
        score += 2.0

    score += 0.6 * len(match["trigger_overlap"])

    activation_text = " ".join(card.get("activation_conditions", []) or []).lower()
    if activation_text:
        score += 0.25 * sum(1 for token in match["record_tokens"] if token in activation_text)

    if match["error_match"]:
        score += 1.5

    score += 1.5 * max(float(card.get("transfer_gain", 0.0) or 0.0), 0.0)
    score += 0.75 * float(card.get("confidence", 0.0) or 0.0)
    score -= 1.75 * float(card.get("negative_transfer_rate", 0.0) or 0.0)

    runtime_stats = dict(card.get("runtime_stats", {}) or {})
    retrieved = int(runtime_stats.get("retrieved", 0) or 0)
    if retrieved > 0:
        helped = int(runtime_stats.get("helped", 0) or 0)
        hurt = int(runtime_stats.get("hurt", 0) or 0)
        score += 3.0 * (helped / max(retrieved, 1))
        score -= 4.5 * (hurt / max(retrieved, 1))

    score -= _stop_condition_risk(card, match["record_tokens"], match["error_family"])
    return score


def rank_skill_cards_for_record(
    store: SkillStore | list[dict],
    record: dict,
    *,
    limit: int = 1,
    promoted_only: bool = True,
    config=None,
) -> list[dict]:
    if isinstance(store, SkillStore):
        cards = list(store.filter(promoted=True)) if promoted_only else list(store)
        promoted_families = int(store.summary().get("promoted_families", 0) or 0)
    else:
        cards = [normalize_skill_card(card, copy_card=True) for card in store]
        if promoted_only:
            cards = [card for card in cards if card.get("status") == "promoted"]
        promoted_families = len({card.get("family_key") or card.get("task_type") or card.get("domain") for card in cards})
    min_score = float(getattr(config, "skill_retrieval_min_score", 4.5))
    strict_min_score = float(getattr(config, "skill_retrieval_strict_min_score", 6.0))
    min_families_for_broad_match = int(getattr(config, "skill_retrieval_min_promoted_families_for_broad_match", 3))
    low_coverage = promoted_families < min_families_for_broad_match
    scored = []
    for card in cards:
        if _runtime_route_disabled(card, config):
            continue
        match = _match_context(card, record)
        if low_coverage and not (
            match["exact_task_match"] or match["strong_domain_match"] or match["strong_error_match"]
        ):
            continue
        score = _skill_match_score(card, record)
        required_score = strict_min_score if low_coverage else min_score
        if score >= required_score:
            scored.append((score, card))
    scored.sort(key=lambda item: (-item[0], -float(item[1].get("confidence", 0.0) or 0.0), item[1]["skill_id"]))
    return [card for _, card in scored[:limit]]


def rank_skill_cards_for_task(
    store: SkillStore | list[dict],
    task: dict,
    *,
    limit: int = 1,
    promoted_only: bool = True,
    config=None,
) -> list[dict]:
    return rank_skill_cards_for_record(
        store,
        task_to_skill_record(task),
        limit=limit,
        promoted_only=promoted_only,
        config=config,
    )


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

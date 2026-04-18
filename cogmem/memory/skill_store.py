"""Typed storage for procedural skill cards."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from cogmem.memory.schema import CARD_TRANSFER_GAIN_KEY, NEGATIVE_TRANSFER_RATE_KEY, RETRIEVAL_CONFIDENCE_KEY


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _dedupe_str_list(values) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalize_runtime_stats(stats: dict | None) -> dict:
    target = dict(stats or {})
    target["retrieved"] = int(target.get("retrieved", 0) or 0)
    target["helped"] = int(target.get("helped", 0) or 0)
    target["hurt"] = int(target.get("hurt", 0) or 0)
    target["preserved_success"] = int(target.get("preserved_success", 0) or 0)
    target["preserved_failure"] = int(target.get("preserved_failure", 0) or 0)
    target["passed"] = int(target.get("passed", 0) or 0)
    target["failed"] = int(target.get("failed", 0) or 0)
    target["domains"] = {str(k): int(v) for k, v in dict(target.get("domains", {}) or {}).items()}
    target["error_families"] = {str(k): int(v) for k, v in dict(target.get("error_families", {}) or {}).items()}
    target["task_ids"] = _dedupe_str_list(target.get("task_ids", []))[:20]
    target["route_breakdown"] = {
        str(route): _normalize_runtime_stats(route_stats)
        for route, route_stats in dict(target.get("route_breakdown", {}) or {}).items()
    }
    return target


def merge_runtime_stats(existing: dict | None, update: dict | None) -> dict:
    base = _normalize_runtime_stats(existing)
    incoming = _normalize_runtime_stats(update)
    merged = dict(base)
    for key in ["retrieved", "helped", "hurt", "preserved_success", "preserved_failure", "passed", "failed"]:
        merged[key] = int(base.get(key, 0)) + int(incoming.get(key, 0))
    domains = dict(base.get("domains", {}) or {})
    for key, value in dict(incoming.get("domains", {}) or {}).items():
        domains[str(key)] = int(domains.get(str(key), 0)) + int(value)
    merged["domains"] = domains
    error_families = dict(base.get("error_families", {}) or {})
    for key, value in dict(incoming.get("error_families", {}) or {}).items():
        error_families[str(key)] = int(error_families.get(str(key), 0)) + int(value)
    merged["error_families"] = error_families
    merged["task_ids"] = _dedupe_str_list(list(base.get("task_ids", [])) + list(incoming.get("task_ids", [])))[:20]
    route_breakdown = dict(base.get("route_breakdown", {}) or {})
    for route, route_stats in dict(incoming.get("route_breakdown", {}) or {}).items():
        route_breakdown[str(route)] = merge_runtime_stats(route_breakdown.get(str(route), {}), route_stats)
    merged["route_breakdown"] = route_breakdown
    return merged


def _skill_id(card: dict) -> str:
    if card.get("skill_id"):
        return str(card["skill_id"])
    seed = json.dumps(
        {
            "task_type": card.get("task_type", "general"),
            "domain": card.get("domain", "general"),
            "error_family": card.get("error_family"),
            "triggers": _dedupe_str_list(card.get("triggers", [])),
            "evidence_episode_ids": _dedupe_str_list(card.get("evidence_episode_ids", [])),
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return f"skill_{hashlib.sha256(seed).hexdigest()[:16]}"


def skill_family_key(card: dict) -> str:
    task_type = str(card.get("task_type", "general") or "general")
    domain = str(card.get("domain", "general") or "general")
    if task_type and task_type != "general":
        return task_type
    return domain


def normalize_skill_card(card: dict, *, copy_card: bool = False) -> dict:
    target = deepcopy(card) if copy_card else card
    target["skill_id"] = _skill_id(target)
    target["triggers"] = _dedupe_str_list(target.get("triggers", []))
    target["plan_steps"] = _dedupe_str_list(target.get("plan_steps", []))
    target["activation_conditions"] = _dedupe_str_list(target.get("activation_conditions", []))
    target["stop_conditions"] = _dedupe_str_list(target.get("stop_conditions", []))
    target["anti_patterns"] = _dedupe_str_list(target.get("anti_patterns", []))
    target["evidence_episode_ids"] = _dedupe_str_list(target.get("evidence_episode_ids", []))
    target["evidence_task_ids"] = _dedupe_str_list(target.get("evidence_task_ids", []))
    target["manifest_ids"] = sorted(set(_dedupe_str_list(target.get("manifest_ids", []))))
    target["validation"] = dict(target.get("validation", {}) or {})
    target["task_type"] = str(target.get("task_type", "general") or "general")
    target["domain"] = str(target.get("domain", "general") or "general")
    target["error_family"] = target.get("error_family")
    target["source_episode_count"] = int(target.get("source_episode_count") or len(target["evidence_episode_ids"]))
    target["distinct_task_count"] = int(target.get("distinct_task_count") or len(target["evidence_task_ids"]))
    target["family_key"] = str(target.get("family_key") or skill_family_key(target))
    target["runtime_stats"] = _normalize_runtime_stats(target.get("runtime_stats", {}))
    transfer_gain = float(target.get("transfer_gain", target.get(CARD_TRANSFER_GAIN_KEY, 0.0)) or 0.0)
    confidence = _clamp01(target.get("confidence", target.get(RETRIEVAL_CONFIDENCE_KEY, 0.0)) or 0.0)
    negative_transfer_rate = _clamp01(
        target.get("negative_transfer_rate", target.get(NEGATIVE_TRANSFER_RATE_KEY, 0.0)) or 0.0
    )
    target["transfer_gain"] = transfer_gain
    target[CARD_TRANSFER_GAIN_KEY] = transfer_gain
    target["confidence"] = confidence
    target[RETRIEVAL_CONFIDENCE_KEY] = confidence
    target["negative_transfer_rate"] = negative_transfer_rate
    target[NEGATIVE_TRANSFER_RATE_KEY] = negative_transfer_rate
    status = str(target.get("status", "candidate") or "candidate")
    if status not in {"candidate", "validated", "promoted"}:
        status = "candidate"
    target["status"] = status
    return target


def render_skill_card_context(card: dict, *, include_header: bool = True) -> str:
    normalized = normalize_skill_card(card, copy_card=True)
    lines: list[str] = []

    if include_header:
        lines.append(
            "Retrieved skill: {} ({})".format(
                normalized.get("task_type", "general"),
                normalized.get("domain", "general"),
            )
        )
    if normalized.get("activation_conditions"):
        lines.append("Use this when:")
        for item in normalized["activation_conditions"][:4]:
            lines.append(f"- {item}")
    if normalized.get("plan_steps"):
        lines.append("Procedure:")
        for idx, item in enumerate(normalized["plan_steps"][:5], start=1):
            lines.append(f"{idx}. {item}")
    if normalized.get("stop_conditions"):
        lines.append("Stop and reconsider if:")
        for item in normalized["stop_conditions"][:4]:
            lines.append(f"- {item}")
    if normalized.get("anti_patterns"):
        lines.append("Avoid:")
        for item in normalized["anti_patterns"][:4]:
            lines.append(f"- {item}")
    return "\n".join(lines)


class SkillStore:
    """Persisted collection of validated procedural skill cards."""

    def __init__(self, cards: list[dict] | None = None):
        self._cards = [normalize_skill_card(card, copy_card=True) for card in (cards or [])]
        self._index = {card["skill_id"]: card for card in self._cards}

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self):
        return iter(self._cards)

    @property
    def cards(self) -> tuple[dict, ...]:
        return tuple(self._cards)

    @classmethod
    def load(cls, path: str) -> "SkillStore":
        p = Path(path)
        if not p.exists():
            return cls([])
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._cards, f, indent=2, ensure_ascii=False)

    def get(self, skill_id: str) -> dict | None:
        return self._index.get(skill_id)

    def add(self, card: dict) -> dict:
        normalized = normalize_skill_card(card, copy_card=True)
        existing = self._index.get(normalized["skill_id"])
        if existing is None:
            self._cards.append(normalized)
        else:
            idx = next(i for i, current in enumerate(self._cards) if current["skill_id"] == normalized["skill_id"])
            self._cards[idx] = normalized
        self._index[normalized["skill_id"]] = normalized
        return normalized

    def update(self, skill_id: str, **fields) -> dict | None:
        card = self._index.get(skill_id)
        if card is None:
            return None
        card.update(fields)
        normalized = normalize_skill_card(card, copy_card=False)
        self._index[skill_id] = normalized
        return normalized

    def apply_runtime_utility(self, utility_by_skill: dict[str, dict], *, route_name: str | None = None) -> int:
        updated = 0
        for skill_id, utility in dict(utility_by_skill or {}).items():
            card = self._index.get(skill_id)
            if card is None:
                continue
            utility_stats = _normalize_runtime_stats(utility)
            if route_name:
                route_stats = dict(utility_stats)
                route_stats["route_breakdown"] = {}
                utility_stats = dict(utility_stats)
                utility_stats["route_breakdown"] = {str(route_name): route_stats}
            merged = merge_runtime_stats(card.get("runtime_stats", {}), utility_stats)
            self.update(skill_id, runtime_stats=merged)
            updated += 1
        return updated

    def filter(
        self,
        *,
        task_type: str | None = None,
        domain: str | None = None,
        manifest_id: str | None = None,
        status: str | None = None,
        promoted: bool | None = None,
    ) -> list[dict]:
        results = list(self._cards)
        if task_type is not None:
            results = [card for card in results if card.get("task_type") == task_type]
        if domain is not None:
            results = [card for card in results if card.get("domain") == domain]
        if manifest_id is not None:
            results = [card for card in results if manifest_id in (card.get("manifest_ids") or [])]
        if status is not None:
            results = [card for card in results if card.get("status") == status]
        if promoted is not None:
            if promoted:
                results = [card for card in results if card.get("status") == "promoted"]
            else:
                results = [card for card in results if card.get("status") != "promoted"]
        return results

    def summary(self) -> dict:
        if not self._cards:
            return {
                "total": 0,
                "candidate": 0,
                "validated": 0,
                "promoted": 0,
                "mean_confidence": 0.0,
                "mean_transfer_gain": 0.0,
                "promoted_families": 0,
            }
        promoted = self.filter(promoted=True)
        candidate = self.filter(status="candidate")
        validated = self.filter(status="validated")
        return {
            "total": len(self._cards),
            "candidate": len(candidate),
            "validated": len(validated),
            "promoted": len(promoted),
            "mean_confidence": sum(card["confidence"] for card in self._cards) / len(self._cards),
            "mean_transfer_gain": sum(card["transfer_gain"] for card in self._cards) / len(self._cards),
            "promoted_families": len(
                {
                    skill_family_key(card)
                    for card in promoted
                    if skill_family_key(card)
                }
            ),
        }

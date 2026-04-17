"""Registry for routed adapter artifacts and their lineage metadata."""

from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _dedupe_str_list(values) -> list[str]:
    seen: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return seen


@dataclass
class AdapterArtifact:
    adapter_id: str
    adapter_path: str
    base_model: str
    adapter_role: str = "family"
    training_manifest_ids: list[str] = field(default_factory=list)
    source_skill_card_ids: list[str] = field(default_factory=list)
    dev_gain: float = 0.0
    compatible_families: list[str] = field(default_factory=list)
    rollback_status: str = "active"
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AdapterArtifact":
        return cls(**raw)


def normalize_adapter_record(record: dict[str, Any] | AdapterArtifact, *, copy_record: bool = False) -> dict[str, Any]:
    target = deepcopy(record.to_dict() if isinstance(record, AdapterArtifact) else record) if copy_record else (
        record.to_dict() if isinstance(record, AdapterArtifact) else record
    )
    target["adapter_id"] = str(target.get("adapter_id") or f"adapter_{uuid.uuid4().hex[:12]}")
    target["adapter_path"] = str(target.get("adapter_path") or "")
    target["base_model"] = str(target.get("base_model") or "")
    target["adapter_role"] = str(target.get("adapter_role") or "family")
    target["training_manifest_ids"] = sorted(set(_dedupe_str_list(target.get("training_manifest_ids", []))))
    target["source_skill_card_ids"] = _dedupe_str_list(target.get("source_skill_card_ids", []))
    target["compatible_families"] = _dedupe_str_list(target.get("compatible_families", []))
    target["rollback_status"] = str(target.get("rollback_status") or "active")
    target["dev_gain"] = float(target.get("dev_gain", 0.0) or 0.0)
    target["created_at"] = float(target.get("created_at", time.time()) or time.time())
    target["metadata"] = dict(target.get("metadata", {}) or {})
    return target


class AdapterRegistry:
    """Persisted registry of adapter artifacts for routed inference."""

    def __init__(self, records: list[dict[str, Any]] | list[AdapterArtifact] | None = None):
        self._records = [AdapterArtifact.from_dict(normalize_adapter_record(record, copy_record=True)) for record in (records or [])]
        self._index = {record.adapter_id: record for record in self._records}

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    @property
    def records(self) -> tuple[AdapterArtifact, ...]:
        return tuple(self._records)

    @classmethod
    def load(cls, path: str) -> "AdapterRegistry":
        p = Path(path)
        if not p.exists():
            return cls([])
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls(raw)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([record.to_dict() for record in self._records], f, indent=2, ensure_ascii=False)

    def get(self, adapter_id: str) -> AdapterArtifact | None:
        return self._index.get(adapter_id)

    def find_by_path(self, adapter_path: str) -> AdapterArtifact | None:
        for record in self._records:
            if record.adapter_path == adapter_path:
                return record
        return None

    def add(self, record: dict[str, Any] | AdapterArtifact) -> AdapterArtifact:
        normalized = AdapterArtifact.from_dict(normalize_adapter_record(record, copy_record=True))
        existing = self._index.get(normalized.adapter_id)
        if existing is None:
            self._records.append(normalized)
        else:
            idx = next(i for i, current in enumerate(self._records) if current.adapter_id == normalized.adapter_id)
            self._records[idx] = normalized
        self._index[normalized.adapter_id] = normalized
        return normalized

    def update(self, adapter_id: str, **fields) -> AdapterArtifact | None:
        record = self._index.get(adapter_id)
        if record is None:
            return None
        payload = record.to_dict()
        payload.update(fields)
        normalized = AdapterArtifact.from_dict(normalize_adapter_record(payload, copy_record=False))
        idx = next(i for i, current in enumerate(self._records) if current.adapter_id == adapter_id)
        self._records[idx] = normalized
        self._index[adapter_id] = normalized
        return normalized

    def filter(
        self,
        *,
        adapter_role: str | None = None,
        manifest_id: str | None = None,
        family: str | None = None,
        rollback_status: str | None = None,
    ) -> list[AdapterArtifact]:
        results = list(self._records)
        if adapter_role is not None:
            results = [record for record in results if record.adapter_role == adapter_role]
        if manifest_id is not None:
            results = [record for record in results if manifest_id in record.training_manifest_ids]
        if family is not None:
            results = [record for record in results if family in record.compatible_families]
        if rollback_status is not None:
            results = [record for record in results if record.rollback_status == rollback_status]
        return results

    def select_routed_adapters(
        self,
        *,
        task_family: str | None,
        min_dev_gain: float = 0.0,
        allow_global: bool = True,
        allow_family: bool = True,
    ) -> dict[str, AdapterArtifact | None]:
        active = [record for record in self._records if record.rollback_status == "active" and record.dev_gain >= min_dev_gain]
        selected_global = None
        selected_family = None
        if allow_global:
            globals_ = [record for record in active if record.adapter_role == "global"]
            if globals_:
                selected_global = max(globals_, key=lambda record: (record.dev_gain, record.created_at))
        if allow_family and task_family:
            family_matches = [
                record for record in active
                if record.adapter_role == "family" and task_family in record.compatible_families
            ]
            if family_matches:
                selected_family = max(family_matches, key=lambda record: (record.dev_gain, record.created_at))
        return {"global": selected_global, "family": selected_family}


def write_adapter_metadata(
    adapter_dir: str,
    *,
    base_model: str,
    adapter_role: str = "family",
    training_manifest_ids: list[str] | None = None,
    source_skill_card_ids: list[str] | None = None,
    compatible_families: list[str] | None = None,
    dev_gain: float = 0.0,
    rollback_status: str = "active",
    metadata: dict[str, Any] | None = None,
    adapter_id: str | None = None,
) -> dict[str, Any]:
    payload = normalize_adapter_record(
        {
            "adapter_id": adapter_id,
            "adapter_path": adapter_dir,
            "base_model": base_model,
            "adapter_role": adapter_role,
            "training_manifest_ids": training_manifest_ids or [],
            "source_skill_card_ids": source_skill_card_ids or [],
            "dev_gain": dev_gain,
            "compatible_families": compatible_families or [],
            "rollback_status": rollback_status,
            "metadata": metadata or {},
        },
        copy_record=True,
    )
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(adapter_dir) / "adapter_metadata.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def register_adapter_artifact(
    registry_path: str,
    adapter_dir: str,
    *,
    base_model: str,
    adapter_role: str = "family",
    training_manifest_ids: list[str] | None = None,
    source_skill_card_ids: list[str] | None = None,
    compatible_families: list[str] | None = None,
    dev_gain: float = 0.0,
    rollback_status: str = "active",
    metadata: dict[str, Any] | None = None,
    adapter_id: str | None = None,
) -> AdapterArtifact:
    payload = write_adapter_metadata(
        adapter_dir,
        base_model=base_model,
        adapter_role=adapter_role,
        training_manifest_ids=training_manifest_ids,
        source_skill_card_ids=source_skill_card_ids,
        compatible_families=compatible_families,
        dev_gain=dev_gain,
        rollback_status=rollback_status,
        metadata=metadata,
        adapter_id=adapter_id,
    )
    registry = AdapterRegistry.load(registry_path)
    record = registry.add(payload)
    registry.save(registry_path)
    return record

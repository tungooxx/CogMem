"""CognitivePatch — a micro-LoRA adapter capturing one lesson.

Each patch is a tiny rank-2 LoRA (A, B matrices) for attention projections.
For Qwen2.5:3b (hidden=2048, 36 layers, 4 projections):
  One rank-2 patch ≈ 2048*2*2 * 4 projections * 36 layers * 4 bytes ≈ 4.7MB

Patches are stored as .pt (weights) + .json (metadata) files.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class CognitivePatch:
    patch_id: str
    embedding: list[float]  # 384-dim task embedding (MiniLM)
    lora_weights: dict  # {layer_name: {"A": tensor, "B": tensor}}
    rank: int = 2
    q_value: float = 0.5
    q_visits: int = 0
    q_successes: int = 0
    q_failures: int = 0
    source_task_id: str = ""
    source_type: str = ""  # "fail_to_pass" | "mutation" | "cluster"
    description: str = ""
    created_at: float = field(default_factory=time.time)

    def save(self, directory: str) -> None:
        """Save patch as .pt (weights) + .json (metadata)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        # Save weights
        torch.save(self.lora_weights, str(d / f"{self.patch_id}.pt"))

        # Save metadata (everything except weights)
        meta = {
            "patch_id": self.patch_id,
            "embedding": self.embedding,
            "rank": self.rank,
            "q_value": self.q_value,
            "q_visits": self.q_visits,
            "q_successes": self.q_successes,
            "q_failures": self.q_failures,
            "source_task_id": self.source_task_id,
            "source_type": self.source_type,
            "description": self.description,
            "created_at": self.created_at,
        }
        with open(str(d / f"{self.patch_id}.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str, patch_id: str, load_weights: bool = True) -> "CognitivePatch":
        """Load patch from directory. Weights loaded lazily if load_weights=False."""
        d = Path(directory)

        with open(str(d / f"{patch_id}.json")) as f:
            meta = json.load(f)

        weights = {}
        if load_weights:
            weights = torch.load(str(d / f"{patch_id}.pt"), map_location="cpu",
                                 weights_only=True)

        return cls(
            patch_id=meta["patch_id"],
            embedding=meta["embedding"],
            lora_weights=weights,
            rank=meta.get("rank", 2),
            q_value=meta.get("q_value", 0.5),
            q_visits=meta.get("q_visits", 0),
            q_successes=meta.get("q_successes", 0),
            q_failures=meta.get("q_failures", 0),
            source_task_id=meta.get("source_task_id", ""),
            source_type=meta.get("source_type", ""),
            description=meta.get("description", ""),
            created_at=meta.get("created_at", 0),
        )

    def unload_weights(self) -> None:
        """Clear lora_weights dict to free GPU/CPU memory."""
        self.lora_weights = {}

    def memory_bytes(self) -> int:
        """Estimate memory usage of this patch's weights."""
        total = 0
        for layer_weights in self.lora_weights.values():
            for tensor in layer_weights.values():
                if isinstance(tensor, torch.Tensor):
                    total += tensor.nelement() * tensor.element_size()
        return total

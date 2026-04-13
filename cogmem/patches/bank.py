"""PatchBank — storage, retrieval, and Q-value tracking for cognitive patches.

Patches stored as individual files in a directory.
Retrieval: two-phase (semantic similarity → Q-value re-ranking).
Q-values updated based on whether patch activation helped the task.
"""

import json
from pathlib import Path

import numpy as np

from cogmem.patches.patch import CognitivePatch

Q_ALPHA = 0.3


class PatchBank:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.patches: list[CognitivePatch] = []
        self.promoted: set[str] = set()  # patch IDs always included in active set
        self._embeddings: np.ndarray | None = None  # Nx384, lazy-built
        self._index: dict[str, int] = {}  # patch_id → index

    def add(self, patch: CognitivePatch) -> None:
        """Add a patch and save it to disk."""
        idx = self._index.get(patch.patch_id)
        if idx is None:
            idx = len(self.patches)
            self.patches.append(patch)
            self._index[patch.patch_id] = idx
        else:
            self.patches[idx] = patch
        self._embeddings = None  # invalidate cache

        # Save immediately
        patch.save(self.save_dir)
        self._save_index()

    def retrieve(
        self, query_embedding, top_k: int = 10, min_similarity: float = 0.3
    ) -> list[CognitivePatch]:
        """Find patches most relevant to the query task."""
        if not self.patches:
            return []

        embs = self._get_embeddings()
        query = np.array(query_embedding)

        # Cosine similarity
        norms = np.linalg.norm(embs, axis=1) * np.linalg.norm(query) + 1e-8
        sims = embs @ query / norms

        # Filter and sort
        candidates = [
            (i, float(sims[i]))
            for i in range(len(self.patches))
            if sims[i] >= min_similarity
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k]

        return [self.patches[i] for i, _ in candidates]

    def get_active_patches(
        self, query_embedding, top_k: int = 5
    ) -> list[CognitivePatch]:
        """Retrieve patches, re-rank by Q-value, return top-k for composition.

        Score = similarity * 0.4 + q_value * 0.6
        Promoted patches are always included in the result regardless of score.
        """
        candidates = self.retrieve(query_embedding, top_k=top_k * 3)
        if not candidates and not self.promoted:
            return []

        query = np.array(query_embedding)
        scored = []
        for patch in candidates:
            emb = np.array(patch.embedding)
            norm = np.linalg.norm(emb) * np.linalg.norm(query) + 1e-8
            sim = float(emb @ query / norm)
            score = sim * 0.4 + patch.q_value * 0.6
            scored.append((score, patch))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [patch for _, patch in scored[:top_k]]

        # Always include promoted patches (they've proven universally useful)
        result_ids = {p.patch_id for p in result}
        for pid in self.promoted:
            if pid not in result_ids:
                idx = self._index.get(pid)
                if idx is not None:
                    result.append(self.patches[idx])

        return result

    def update_q(self, patch_id: str, task_succeeded: bool) -> None:
        """Update Q-value based on whether activation helped."""
        idx = self._index.get(patch_id)
        if idx is None:
            return

        patch = self.patches[idx]
        reward = 1.0 if task_succeeded else 0.0
        patch.q_value = patch.q_value + Q_ALPHA * (reward - patch.q_value)
        patch.q_visits += 1
        if task_succeeded:
            patch.q_successes += 1
        else:
            patch.q_failures += 1

        # Save updated metadata
        patch.save(self.save_dir)

    def save(self) -> None:
        """Save all patches and index."""
        for patch in self.patches:
            patch.save(self.save_dir)
        self._save_index()

    def load(self) -> None:
        """Load patches from directory."""
        d = Path(self.save_dir)
        index_path = d / "index.json"
        if not index_path.exists():
            return

        with open(index_path) as f:
            raw = json.load(f)

        # Support both old format (list of IDs) and new format (dict)
        if isinstance(raw, list):
            patch_ids = raw
            self.promoted = set()
        else:
            patch_ids = raw.get("patch_ids", [])
            self.promoted = set(raw.get("promoted", []))

        self.patches = []
        self._index = {}
        for i, pid in enumerate(patch_ids):
            try:
                patch = CognitivePatch.load(self.save_dir, pid, load_weights=False)
                self.patches.append(patch)
                self._index[pid] = i
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Warning: could not load patch {pid}")

        self._embeddings = None
        print(f"Loaded {len(self.patches)} patches ({len(self.promoted)} promoted) from {self.save_dir}")

    def load_weights(self, patch: CognitivePatch) -> None:
        """Lazy-load weights for a specific patch."""
        if not patch.lora_weights:
            loaded = CognitivePatch.load(self.save_dir, patch.patch_id, load_weights=True)
            patch.lora_weights = loaded.lora_weights

    def get_patch(self, patch_id: str) -> CognitivePatch | None:
        """Return an in-memory patch record by id."""
        idx = self._index.get(patch_id)
        if idx is None:
            return None
        return self.patches[idx]

    def stats(self) -> dict:
        """Get patch bank statistics."""
        if not self.patches:
            return {"total": 0}

        q_vals = [p.q_value for p in self.patches]
        return {
            "total": len(self.patches),
            "high_q": sum(1 for q in q_vals if q >= 0.7),
            "mid_q": sum(1 for q in q_vals if 0.3 <= q < 0.7),
            "low_q": sum(1 for q in q_vals if q < 0.3),
            "mean_q": float(np.mean(q_vals)),
            "total_memory_mb": sum(p.memory_bytes() for p in self.patches) / 1024 / 1024,
            "avg_rank": float(np.mean([p.rank for p in self.patches])),
        }

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _get_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            if self.patches:
                self._embeddings = np.array([p.embedding for p in self.patches])
            else:
                self._embeddings = np.zeros((0, 384))
        return self._embeddings

    def _save_index(self) -> None:
        d = Path(self.save_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "index.json", "w") as f:
            json.dump({
                "patch_ids": [p.patch_id for p in self.patches],
                "promoted": list(self.promoted),
            }, f)

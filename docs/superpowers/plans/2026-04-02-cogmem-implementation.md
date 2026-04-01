# CogMem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Q-value guided memory consolidation system that collects RL episodes from MemRL/ALFWorld, selects high-Q episodes via 5 policies, trains LoRA adapters via Together AI, and evaluates them locally.

**Architecture:** Two-phase pipeline. Phase 1 runs MemRL on ALFWorld to collect ~300 episodes with Q-values into a memory bank JSON. Phase 2 (the `cogmem` Python package) selects episodes by policy, converts proceduralized scripts to JSONL, trains LoRA adapters on Together AI, downloads adapter weights, and evaluates locally with transformers+peft. A router dispatches tasks between consolidated LoRA, episodic retrieval, or cold inference.

**Tech Stack:** Python 3.10+, pytest, sentence-transformers, openai SDK, together SDK, transformers, peft, bitsandbytes, numpy, scipy, ALFWorld

**Design Spec:** `docs/superpowers/specs/2026-04-02-cogmem-design.md`

---

## File Structure

```
F:/Newgeneration/CogMem/
├── pyproject.toml
├── cogmem/
│   ├── __init__.py
│   ├── config.py                      # CogMemConfig dataclass — all hyperparams
│   ├── consolidation/
│   │   ├── __init__.py
│   │   ├── select.py                  # 5 selection policies (q_top_k, recency, frequency, random, all)
│   │   ├── abstract.py               # Format converter: proceduralized scripts → JSONL
│   │   ├── train_lora.py             # Together AI LoRA training + adapter download
│   │   ├── verify.py                 # Holdout verification with confidence intervals
│   │   ├── prune.py                  # Episode pruning after consolidation
│   │   ├── router.py                 # Task routing: LoRA vs episodic vs cold
│   │   └── pipeline.py              # Experiment 1 + 2 orchestration
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── memory_bank.py            # Load/save/query/split episode bank
│   │   └── embeddings.py             # sentence-transformers wrapper
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── alfworld_eval.py          # ALFWorld eval with transformers+peft
│   │   └── compare.py                # Multi-policy comparison tables
│   └── utils/
│       ├── __init__.py
│       ├── llm_client.py             # Unified LLM client (Ollama/Groq/Together)
│       └── logging.py                # JSON experiment logging
├── scripts/
│   ├── convert_memrl.py              # MemRL cube dump → cogmem memory_bank.json
│   ├── run_phase0.py                 # Pre-flight checks
│   └── run_collection.py             # MemRL runner wrapper with 3B fallback logic
├── tests/
│   ├── conftest.py                   # Shared fixtures (sample memory bank)
│   ├── test_config.py
│   ├── test_memory_bank.py
│   ├── test_embeddings.py
│   ├── test_llm_client.py
│   ├── test_select.py
│   ├── test_abstract.py
│   ├── test_train_lora.py
│   ├── test_verify.py
│   ├── test_router.py
│   ├── test_compare.py
│   └── test_convert_memrl.py
└── configs/
    └── default.yaml                  # Default CogMemConfig as YAML
```

**Parallelization notes for subagent execution:**
- After Task 1: Tasks 2-5 can run in parallel
- After Tasks 2-5: Tasks 6-8 can run in parallel
- Tasks 9-11 have dependencies, run sequentially
- Tasks 12-15 depend on most earlier tasks, run sequentially
- Tasks 16-17 are independent of the cogmem module, can run anytime

---

## Task 1: Project Scaffolding + Test Fixtures

**Files:**
- Create: `pyproject.toml`
- Create: `cogmem/__init__.py`
- Create: `cogmem/consolidation/__init__.py`
- Create: `cogmem/memory/__init__.py`
- Create: `cogmem/evaluation/__init__.py`
- Create: `cogmem/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "cogmem"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "openai>=1.0",
    "together>=1.0",
    "sentence-transformers>=2.2",
    "torch>=2.0",
    "transformers>=4.36",
    "peft>=0.7",
    "bitsandbytes>=0.41",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-mock>=3.10"]
alfworld = ["alfworld>=0.3"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init files**

All `__init__.py` files are empty except the root:

`cogmem/__init__.py`:
```python
__version__ = "0.1.0"
```

Create empty `__init__.py` in: `cogmem/consolidation/`, `cogmem/memory/`, `cogmem/evaluation/`, `cogmem/utils/`, `tests/`.

- [ ] **Step 3: Create test fixtures in conftest.py**

`tests/conftest.py`:
```python
import json
import pytest
from pathlib import Path


@pytest.fixture
def sample_episodes():
    """15 episodes across 6 task types, mix of success/failure, varying Q-values."""
    return [
        {
            "episode_id": "ep_001",
            "task_description": "put a clean mug in shelf 1",
            "task_type": "clean",
            "script": "1. Find mug on countertop 1\n2. Take mug\n3. Go to sinkbasin 1\n4. Clean mug\n5. Go to shelf 1\n6. Put mug in shelf 1",
            "intent_embedding": [0.1] * 384,
            "success": True,
            "q_value": 0.92,
            "q_visits": 8,
            "num_steps": 6,
            "timestamp": "2026-04-01T10:00:00",
        },
        {
            "episode_id": "ep_002",
            "task_description": "put a hot apple in fridge 1",
            "task_type": "heat",
            "script": "1. Find apple on dining table\n2. Take apple\n3. Go to microwave 1\n4. Heat apple\n5. Go to fridge 1\n6. Put apple in fridge 1",
            "intent_embedding": [0.2] * 384,
            "success": True,
            "q_value": 0.85,
            "q_visits": 6,
            "num_steps": 6,
            "timestamp": "2026-04-01T10:05:00",
        },
        {
            "episode_id": "ep_003",
            "task_description": "examine a book under desklamp",
            "task_type": "examine",
            "script": "1. Find book on shelf 2\n2. Take book\n3. Go to desklamp 1\n4. Use desklamp",
            "intent_embedding": [0.3] * 384,
            "success": True,
            "q_value": 0.78,
            "q_visits": 5,
            "num_steps": 4,
            "timestamp": "2026-04-01T10:10:00",
        },
        {
            "episode_id": "ep_004",
            "task_description": "put a pencil in drawer 1",
            "task_type": "pick",
            "script": "1. Find pencil on desk 1\n2. Take pencil\n3. Go to drawer 1\n4. Put pencil in drawer 1",
            "intent_embedding": [0.4] * 384,
            "success": True,
            "q_value": 0.71,
            "q_visits": 4,
            "num_steps": 4,
            "timestamp": "2026-04-01T10:15:00",
        },
        {
            "episode_id": "ep_005",
            "task_description": "put a cool egg in fridge 1",
            "task_type": "cool",
            "script": "1. Find egg on countertop 2\n2. Take egg\n3. Go to fridge 1\n4. Cool egg\n5. Put egg in fridge 1",
            "intent_embedding": [0.5] * 384,
            "success": True,
            "q_value": 0.65,
            "q_visits": 3,
            "num_steps": 5,
            "timestamp": "2026-04-01T10:20:00",
        },
        {
            "episode_id": "ep_006",
            "task_description": "put two pencils in drawer 1",
            "task_type": "puttwo",
            "script": "1. Find pencil on desk 1\n2. Take pencil\n3. Go to drawer 1\n4. Put pencil\n5. Find pencil on shelf 1\n6. Take pencil\n7. Go to drawer 1\n8. Put pencil",
            "intent_embedding": [0.6] * 384,
            "success": True,
            "q_value": 0.55,
            "q_visits": 3,
            "num_steps": 8,
            "timestamp": "2026-04-01T10:25:00",
        },
        {
            "episode_id": "ep_007",
            "task_description": "put a clean plate in cabinet 1",
            "task_type": "clean",
            "script": "1. Find plate on dining table\n2. Take plate\n3. Go to sinkbasin 1\n4. Clean plate\n5. Go to cabinet 1\n6. Put plate in cabinet 1",
            "intent_embedding": [0.15] * 384,
            "success": True,
            "q_value": 0.45,
            "q_visits": 2,
            "num_steps": 6,
            "timestamp": "2026-04-01T10:30:00",
        },
        {
            "episode_id": "ep_008",
            "task_description": "put a knife in drawer 2",
            "task_type": "pick",
            "script": "",
            "intent_embedding": [0.25] * 384,
            "success": False,
            "q_value": 0.30,
            "q_visits": 2,
            "num_steps": 50,
            "timestamp": "2026-04-01T10:35:00",
        },
        {
            "episode_id": "ep_009",
            "task_description": "examine a pen under desklamp",
            "task_type": "examine",
            "script": "",
            "intent_embedding": [0.35] * 384,
            "success": False,
            "q_value": 0.22,
            "q_visits": 1,
            "num_steps": 50,
            "timestamp": "2026-04-01T10:40:00",
        },
        {
            "episode_id": "ep_010",
            "task_description": "put a hot mug in shelf 2",
            "task_type": "heat",
            "script": "",
            "intent_embedding": [0.45] * 384,
            "success": False,
            "q_value": 0.15,
            "q_visits": 1,
            "num_steps": 50,
            "timestamp": "2026-04-01T10:45:00",
        },
        {
            "episode_id": "ep_011",
            "task_description": "put a cool potato in cabinet 1",
            "task_type": "cool",
            "script": "",
            "intent_embedding": [0.55] * 384,
            "success": False,
            "q_value": 0.10,
            "q_visits": 1,
            "num_steps": 50,
            "timestamp": "2026-04-01T10:50:00",
        },
        {
            "episode_id": "ep_012",
            "task_description": "put two books in shelf 1",
            "task_type": "puttwo",
            "script": "",
            "intent_embedding": [0.65] * 384,
            "success": False,
            "q_value": 0.05,
            "q_visits": 1,
            "num_steps": 50,
            "timestamp": "2026-04-01T10:55:00",
        },
        {
            "episode_id": "ep_013",
            "task_description": "put a clean fork in drawer 1",
            "task_type": "clean",
            "script": "1. Find fork on countertop 1\n2. Take fork\n3. Go to sinkbasin 1\n4. Clean fork\n5. Go to drawer 1\n6. Put fork in drawer 1",
            "intent_embedding": [0.12] * 384,
            "success": True,
            "q_value": 0.88,
            "q_visits": 7,
            "num_steps": 6,
            "timestamp": "2026-04-01T11:00:00",
        },
        {
            "episode_id": "ep_014",
            "task_description": "put a pencil in shelf 1",
            "task_type": "pick",
            "script": "1. Find pencil on desk 2\n2. Take pencil\n3. Go to shelf 1\n4. Put pencil in shelf 1",
            "intent_embedding": [0.42] * 384,
            "success": True,
            "q_value": 0.75,
            "q_visits": 5,
            "num_steps": 4,
            "timestamp": "2026-04-01T11:05:00",
        },
        {
            "episode_id": "ep_015",
            "task_description": "examine a cd under desklamp",
            "task_type": "examine",
            "script": "1. Find cd on shelf 3\n2. Take cd\n3. Go to desklamp 1\n4. Use desklamp",
            "intent_embedding": [0.32] * 384,
            "success": True,
            "q_value": 0.40,
            "q_visits": 2,
            "num_steps": 4,
            "timestamp": "2026-04-01T11:10:00",
        },
    ]


@pytest.fixture
def sample_memory_bank_path(tmp_path, sample_episodes):
    """Write sample episodes to a temp JSON file."""
    path = tmp_path / "memory_bank.json"
    path.write_text(json.dumps(sample_episodes, indent=2))
    return str(path)


@pytest.fixture
def sample_replay_buffer():
    """MemRL-style few-shot examples for replay buffer."""
    return [
        {
            "task": "clean",
            "instruction": "put a clean mug in shelf 1",
            "response": "1. Find mug\n2. Take mug\n3. Go to sinkbasin\n4. Clean mug\n5. Go to shelf 1\n6. Put mug",
        },
        {
            "task": "pick",
            "instruction": "put a pencil in drawer 1",
            "response": "1. Find pencil\n2. Take pencil\n3. Go to drawer 1\n4. Put pencil",
        },
    ]
```

- [ ] **Step 4: Verify scaffolding**

Run: `cd F:/Newgeneration/CogMem && pip install -e ".[dev]"`
Expected: Installs successfully (dependencies may warn about CUDA — OK)

Run: `pytest tests/ --co -q`
Expected: Shows `tests/conftest.py` collected, no errors

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml cogmem/ tests/ docs/
git commit -m "feat: project scaffolding with test fixtures and design spec"
```

---

## Task 2: CogMemConfig

**Files:**
- Create: `cogmem/config.py`
- Create: `tests/test_config.py`
- Create: `configs/default.yaml`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
import os
from cogmem.config import CogMemConfig


def test_default_config():
    cfg = CogMemConfig()
    assert cfg.q_threshold == 0.7
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_epochs == 3
    assert cfg.base_model == "meta-llama/Llama-3.2-3B-Instruct"
    assert cfg.verification_holdout == 20
    assert cfg.regression_threshold == 0.05


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-123")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key-456")
    cfg = CogMemConfig.from_env()
    assert cfg.together_api_key == "test-key-123"
    assert cfg.groq_api_key == "groq-key-456"


def test_config_from_yaml(tmp_path):
    yaml_content = "q_threshold: 0.8\nlora_rank: 8\n"
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(yaml_content)
    cfg = CogMemConfig.from_yaml(str(yaml_path))
    assert cfg.q_threshold == 0.8
    assert cfg.lora_rank == 8
    assert cfg.lora_epochs == 3  # unchanged default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cogmem.config'`

- [ ] **Step 3: Write implementation**

`cogmem/config.py`:
```python
import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class CogMemConfig:
    # Paths
    memory_bank_path: str = "results/memory_bank.json"
    adapters_dir: str = "adapters/"
    logs_dir: str = "logs/"
    replay_buffer_path: str = ""

    # Selection
    q_threshold: float = 0.7
    min_cluster_size: int = 3

    # LoRA training
    lora_provider: str = "together"
    together_api_key: str = ""
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_epochs: int = 3
    lora_learning_rate: float = 1e-5
    lora_batch_size: str = "max"

    # Verification
    verification_holdout: int = 20
    regression_threshold: float = 0.05
    eval_seeds: list[int] | None = None

    # Router
    consolidation_match_threshold: float = 0.75
    retrieval_min_q: float = 0.3

    # Evaluation
    eval_model_api_base: str = "http://localhost:11434/v1"
    eval_model: str = "llama3.2:3b"

    # Providers
    groq_api_key: str = ""
    ollama_api_base: str = "http://localhost:11434/v1"

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.eval_seeds is None:
            self.eval_seeds = [42, 123, 456]

    @classmethod
    def from_env(cls) -> "CogMemConfig":
        cfg = cls()
        cfg.together_api_key = os.environ.get("TOGETHER_API_KEY", "")
        cfg.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "CogMemConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Create default YAML config**

`configs/default.yaml`:
```yaml
memory_bank_path: results/memory_bank.json
adapters_dir: adapters/
logs_dir: logs/
q_threshold: 0.7
lora_rank: 16
lora_alpha: 32
lora_epochs: 3
base_model: meta-llama/Llama-3.2-3B-Instruct
verification_holdout: 20
seed: 42
```

- [ ] **Step 6: Commit**

```bash
git add cogmem/config.py tests/test_config.py configs/default.yaml
git commit -m "feat: CogMemConfig dataclass with YAML and env loading"
```

---

## Task 3: Memory Bank

**Files:**
- Create: `cogmem/memory/memory_bank.py`
- Create: `tests/test_memory_bank.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_memory_bank.py`:
```python
import json
import pytest
from cogmem.memory.memory_bank import MemoryBank


class TestMemoryBankLoad:
    def test_load_from_json(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        assert len(bank) == 15

    def test_episode_access(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        ep = bank.get("ep_001")
        assert ep["task_type"] == "clean"
        assert ep["q_value"] == 0.92

    def test_get_missing_returns_none(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        assert bank.get("nonexistent") is None


class TestMemoryBankSave:
    def test_save_roundtrip(self, sample_memory_bank_path, tmp_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        out = str(tmp_path / "out.json")
        bank.save(out)
        bank2 = MemoryBank.load(out)
        assert len(bank2) == len(bank)


class TestMemoryBankQuery:
    def test_successful_episodes(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        successes = bank.successful()
        assert all(ep["success"] for ep in successes)
        assert len(successes) == 9

    def test_by_task_type(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        clean_eps = bank.by_task_type("clean")
        assert len(clean_eps) == 3
        assert all(ep["task_type"] == "clean" for ep in clean_eps)

    def test_task_types(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        types = bank.task_types()
        assert types == {"clean", "cool", "examine", "heat", "pick", "puttwo"}


class TestMemoryBankSplit:
    def test_holdout_split_size(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        holdout, available = bank.stratified_holdout(n=6, seed=42)
        assert len(holdout) == 6
        assert len(available) == 9
        assert len(holdout) + len(available) == 15

    def test_holdout_stratified(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        holdout, _ = bank.stratified_holdout(n=6, seed=42)
        holdout_types = {ep["task_type"] for ep in holdout}
        assert len(holdout_types) >= 3  # at least 3 task types represented

    def test_holdout_deterministic(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        h1, _ = bank.stratified_holdout(n=6, seed=42)
        h2, _ = bank.stratified_holdout(n=6, seed=42)
        ids1 = {ep["episode_id"] for ep in h1}
        ids2 = {ep["episode_id"] for ep in h2}
        assert ids1 == ids2


class TestMemoryBankMetrics:
    def test_summary_metrics(self, sample_memory_bank_path):
        bank = MemoryBank.load(sample_memory_bank_path)
        m = bank.summary_metrics()
        assert m["total_episodes"] == 15
        assert 0 < m["success_rate"] < 1
        assert "clean" in m["success_rate_by_type"]
        assert "mean" in m["q_value_stats"]
        assert "high_q_episodes" in m
        assert "low_q_episodes" in m
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_bank.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`cogmem/memory/memory_bank.py`:
```python
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


class MemoryBank:
    def __init__(self, episodes: list[dict]):
        self._episodes = episodes
        self._index = {ep["episode_id"]: ep for ep in episodes}

    @classmethod
    def load(cls, path: str) -> "MemoryBank":
        with open(path) as f:
            data = json.load(f)
        return cls(data)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._episodes, f, indent=2)

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self):
        return iter(self._episodes)

    def get(self, episode_id: str) -> dict | None:
        return self._index.get(episode_id)

    def successful(self) -> list[dict]:
        return [ep for ep in self._episodes if ep["success"]]

    def by_task_type(self, task_type: str) -> list[dict]:
        return [ep for ep in self._episodes if ep["task_type"] == task_type]

    def task_types(self) -> set[str]:
        return {ep["task_type"] for ep in self._episodes}

    def stratified_holdout(
        self, n: int, seed: int = 42
    ) -> tuple[list[dict], list[dict]]:
        rng = random.Random(seed)
        by_type = defaultdict(list)
        for ep in self._episodes:
            by_type[ep["task_type"]].append(ep)

        holdout = []
        remaining_budget = n
        types = sorted(by_type.keys())
        per_type = max(1, n // len(types))

        for t in types:
            eps = list(by_type[t])
            rng.shuffle(eps)
            take = min(per_type, len(eps), remaining_budget)
            holdout.extend(eps[:take])
            remaining_budget -= take
            if remaining_budget <= 0:
                break

        # Fill remaining budget from any type
        if remaining_budget > 0:
            used_ids = {ep["episode_id"] for ep in holdout}
            pool = [ep for ep in self._episodes if ep["episode_id"] not in used_ids]
            rng.shuffle(pool)
            holdout.extend(pool[:remaining_budget])

        holdout_ids = {ep["episode_id"] for ep in holdout}
        available = [ep for ep in self._episodes if ep["episode_id"] not in holdout_ids]
        return holdout, available

    def summary_metrics(self) -> dict:
        q_values = [ep["q_value"] for ep in self._episodes]
        successes = [ep for ep in self._episodes if ep["success"]]

        by_type = defaultdict(lambda: {"success": 0, "total": 0})
        for ep in self._episodes:
            by_type[ep["task_type"]]["total"] += 1
            if ep["success"]:
                by_type[ep["task_type"]]["success"] += 1

        return {
            "total_episodes": len(self._episodes),
            "success_rate": len(successes) / len(self._episodes) if self._episodes else 0,
            "success_rate_by_type": {
                t: d["success"] / d["total"] for t, d in sorted(by_type.items())
            },
            "q_value_stats": {
                "mean": mean(q_values),
                "std": stdev(q_values) if len(q_values) > 1 else 0,
                "min": min(q_values),
                "max": max(q_values),
            },
            "high_q_episodes": sum(1 for q in q_values if q > 0.7),
            "low_q_episodes": sum(1 for q in q_values if q < 0.3),
        }

    def sha256(self) -> str:
        content = json.dumps(self._episodes, sort_keys=True).encode()
        return hashlib.sha256(content).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_memory_bank.py -v`
Expected: All 10 tests pass

- [ ] **Step 5: Commit**

```bash
git add cogmem/memory/memory_bank.py tests/test_memory_bank.py
git commit -m "feat: MemoryBank with load/save/query/split/metrics"
```

---

## Task 4: Embeddings Wrapper

**Files:**
- Create: `cogmem/memory/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write the failing test**

`tests/test_embeddings.py`:
```python
import numpy as np
import pytest
from cogmem.memory.embeddings import LocalEmbedder


@pytest.fixture(scope="module")
def embedder():
    return LocalEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")


class TestLocalEmbedder:
    def test_embed_single(self, embedder):
        vec = embedder.embed("put a clean mug in shelf 1")
        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_embed_batch(self, embedder):
        texts = ["put a mug in shelf", "examine a book"]
        vecs = embedder.embed_batch(texts)
        assert len(vecs) == 2
        assert len(vecs[0]) == 384

    def test_cosine_similarity(self, embedder):
        v1 = embedder.embed("put a clean mug in shelf 1")
        v2 = embedder.embed("put a clean plate in shelf 2")
        v3 = embedder.embed("examine a book under desklamp")
        sim_close = embedder.cosine_sim(v1, v2)
        sim_far = embedder.cosine_sim(v1, v3)
        assert sim_close > sim_far  # similar tasks should be closer
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`cogmem/memory/embeddings.py`:
```python
import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    @staticmethod
    def cosine_sim(a: list[float], b: list[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embeddings.py -v`
Expected: 3 passed (first run downloads the model, may take ~30s)

- [ ] **Step 5: Commit**

```bash
git add cogmem/memory/embeddings.py tests/test_embeddings.py
git commit -m "feat: LocalEmbedder wrapper for sentence-transformers"
```

---

## Task 5: Unified LLM Client

**Files:**
- Create: `cogmem/utils/llm_client.py`
- Create: `tests/test_llm_client.py`

- [ ] **Step 1: Write the failing test**

`tests/test_llm_client.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from cogmem.utils.llm_client import LLMClient


def test_init_ollama():
    client = LLMClient(provider="ollama", model="llama3.2:3b")
    assert client.api_base == "http://localhost:11434/v1"


def test_init_groq():
    client = LLMClient(provider="groq", model="llama-3.1-8b-instant", api_key="test")
    assert client.api_base == "https://api.groq.com/openai/v1"


def test_init_together():
    client = LLMClient(provider="together", model="meta-llama/Llama-3.2-3B-Instruct", api_key="test")
    assert client.api_base == "https://api.together.xyz/v1"


@patch("cogmem.utils.llm_client.OpenAI")
def test_generate_calls_openai_client(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello world"))]
    )
    client = LLMClient(provider="ollama", model="llama3.2:3b")
    result = client.generate("say hello")
    assert result == "Hello world"
    mock_client.chat.completions.create.assert_called_once()


@patch("cogmem.utils.llm_client.OpenAI")
def test_generate_retries_on_rate_limit(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    from openai import RateLimitError
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {}
    rate_err = RateLimitError(
        message="rate limit",
        response=mock_resp,
        body=None,
    )
    mock_client.chat.completions.create.side_effect = [
        rate_err,
        MagicMock(choices=[MagicMock(message=MagicMock(content="OK"))]),
    ]
    client = LLMClient(provider="groq", model="test", api_key="k")
    client._retry_base_delay = 0.01  # fast retry for test
    result = client.generate("test")
    assert result == "OK"
    assert mock_client.chat.completions.create.call_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`cogmem/utils/llm_client.py`:
```python
import os
import time

from openai import OpenAI


class LLMClient:
    PROVIDER_BASES = {
        "ollama": "http://localhost:11434/v1",
        "groq": "https://api.groq.com/openai/v1",
        "together": "https://api.together.xyz/v1",
    }

    def __init__(
        self,
        provider: str,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.api_base = api_base or self.PROVIDER_BASES[provider]
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY", "none")
        self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        self._retry_base_delay = 2.0
        self._max_retries = 5

    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0,
    ) -> str:
        for attempt in range(self._max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = self._retry_base_delay * (2**attempt)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Max retries ({self._max_retries}) exceeded")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_client.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add cogmem/utils/llm_client.py tests/test_llm_client.py
git commit -m "feat: unified LLM client for Ollama/Groq/Together"
```

---

## Task 6: Selection Policies

**Files:**
- Create: `cogmem/consolidation/select.py`
- Create: `tests/test_select.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_select.py`:
```python
import pytest
from cogmem.config import CogMemConfig
from cogmem.consolidation.select import (
    select_q_top_k,
    select_recency,
    select_frequency,
    select_random,
    select_all,
)


@pytest.fixture
def config():
    return CogMemConfig(q_threshold=0.7)


class TestQTopK:
    def test_selects_above_threshold(self, sample_episodes, config):
        selected = select_q_top_k(sample_episodes, config)
        assert all(ep["q_value"] >= 0.7 for ep in selected)

    def test_sorted_by_q_descending(self, sample_episodes, config):
        selected = select_q_top_k(sample_episodes, config)
        q_values = [ep["q_value"] for ep in selected]
        assert q_values == sorted(q_values, reverse=True)

    def test_count(self, sample_episodes, config):
        selected = select_q_top_k(sample_episodes, config)
        expected = sum(1 for ep in sample_episodes if ep["q_value"] >= 0.7)
        assert len(selected) == expected


class TestRecency:
    def test_matches_q_top_k_count(self, sample_episodes, config):
        n_q = len(select_q_top_k(sample_episodes, config))
        selected = select_recency(sample_episodes, config)
        assert len(selected) == n_q

    def test_sorted_by_timestamp_descending(self, sample_episodes, config):
        selected = select_recency(sample_episodes, config)
        timestamps = [ep["timestamp"] for ep in selected]
        assert timestamps == sorted(timestamps, reverse=True)


class TestFrequency:
    def test_matches_q_top_k_count(self, sample_episodes, config):
        n_q = len(select_q_top_k(sample_episodes, config))
        selected = select_frequency(sample_episodes, config)
        assert len(selected) == n_q

    def test_sorted_by_visits_descending(self, sample_episodes, config):
        selected = select_frequency(sample_episodes, config)
        visits = [ep["q_visits"] for ep in selected]
        assert visits == sorted(visits, reverse=True)


class TestRandom:
    def test_matches_q_top_k_count(self, sample_episodes, config):
        n_q = len(select_q_top_k(sample_episodes, config))
        selected = select_random(sample_episodes, config)
        assert len(selected) == n_q

    def test_deterministic_with_seed(self, sample_episodes, config):
        s1 = select_random(sample_episodes, config, seed=42)
        s2 = select_random(sample_episodes, config, seed=42)
        ids1 = [ep["episode_id"] for ep in s1]
        ids2 = [ep["episode_id"] for ep in s2]
        assert ids1 == ids2

    def test_different_seed_different_result(self, sample_episodes, config):
        s1 = select_random(sample_episodes, config, seed=42)
        s2 = select_random(sample_episodes, config, seed=99)
        ids1 = {ep["episode_id"] for ep in s1}
        ids2 = {ep["episode_id"] for ep in s2}
        assert ids1 != ids2


class TestAll:
    def test_returns_everything(self, sample_episodes, config):
        selected = select_all(sample_episodes, config)
        assert len(selected) == len(sample_episodes)


class TestFairComparison:
    def test_all_policies_same_count_except_all(self, sample_episodes, config):
        n_q = len(select_q_top_k(sample_episodes, config))
        assert len(select_recency(sample_episodes, config)) == n_q
        assert len(select_frequency(sample_episodes, config)) == n_q
        assert len(select_random(sample_episodes, config)) == n_q
        # 'all' is intentionally different
        assert len(select_all(sample_episodes, config)) == len(sample_episodes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_select.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`cogmem/consolidation/select.py`:
```python
import random as _random


def select_q_top_k(episodes: list[dict], config) -> list[dict]:
    selected = [ep for ep in episodes if ep["q_value"] >= config.q_threshold]
    selected.sort(key=lambda x: x["q_value"], reverse=True)
    return selected


def _match_count(episodes: list[dict], config) -> int:
    return len(select_q_top_k(episodes, config))


def select_recency(episodes: list[dict], config, n: int | None = None) -> list[dict]:
    if n is None:
        n = _match_count(episodes, config)
    sorted_eps = sorted(episodes, key=lambda x: x["timestamp"], reverse=True)
    return sorted_eps[:n]


def select_frequency(episodes: list[dict], config, n: int | None = None) -> list[dict]:
    if n is None:
        n = _match_count(episodes, config)
    sorted_eps = sorted(episodes, key=lambda x: x.get("q_visits", 0), reverse=True)
    return sorted_eps[:n]


def select_random(
    episodes: list[dict], config, n: int | None = None, seed: int = 42
) -> list[dict]:
    if n is None:
        n = _match_count(episodes, config)
    rng = _random.Random(seed)
    pool = list(episodes)
    rng.shuffle(pool)
    return pool[: min(n, len(pool))]


def select_all(episodes: list[dict], config) -> list[dict]:
    return list(episodes)


POLICIES = {
    "q_top_k": select_q_top_k,
    "recency": select_recency,
    "frequency": select_frequency,
    "random": select_random,
    "all": select_all,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_select.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add cogmem/consolidation/select.py tests/test_select.py
git commit -m "feat: 5 episode selection policies with fair-N constraint"
```

---

## Task 7: Abstraction Format Converter

**Files:**
- Create: `cogmem/consolidation/abstract.py`
- Create: `tests/test_abstract.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_abstract.py`:
```python
import json
import pytest
from cogmem.consolidation.abstract import (
    episode_to_training_pair,
    prepare_training_dataset,
    q_weighted_duplicates,
    save_as_jsonl,
)


class TestEpisodeToTrainingPair:
    def test_basic_conversion(self, sample_episodes):
        ep = sample_episodes[0]  # successful, clean task
        pair = episode_to_training_pair(ep)
        assert pair["instruction"] == ep["task_description"]
        assert pair["response"] == ep["script"]
        assert pair["weight"] == ep["q_value"]
        assert pair["source_episode"] == ep["episode_id"]

    def test_skips_failed_episodes(self, sample_episodes):
        ep = sample_episodes[7]  # failed
        pair = episode_to_training_pair(ep)
        assert pair is None


class TestQWeightedDuplicates:
    def test_high_q_gets_more_copies(self):
        pair = {"instruction": "x", "response": "y", "weight": 0.9}
        copies = q_weighted_duplicates([pair])
        assert len(copies) == 3  # round(0.9 * 3) = 3

    def test_medium_q_gets_two_copies(self):
        pair = {"instruction": "x", "response": "y", "weight": 0.7}
        copies = q_weighted_duplicates([pair])
        assert len(copies) == 2  # round(0.7 * 3) = 2

    def test_low_q_gets_one_copy(self):
        pair = {"instruction": "x", "response": "y", "weight": 0.2}
        copies = q_weighted_duplicates([pair])
        assert len(copies) == 1  # max(1, round(0.2 * 3)) = 1


class TestPrepareTrainingDataset:
    def test_only_successful(self, sample_episodes, sample_replay_buffer):
        dataset = prepare_training_dataset(
            sample_episodes, replay_buffer=sample_replay_buffer
        )
        # 9 successful episodes + 2 replay = 11 base pairs (before duplication)
        source_eps = {p.get("source_episode") for p in dataset}
        failed_ids = {ep["episode_id"] for ep in sample_episodes if not ep["success"]}
        assert not source_eps & failed_ids

    def test_includes_replay_buffer(self, sample_episodes, sample_replay_buffer):
        dataset = prepare_training_dataset(
            sample_episodes, replay_buffer=sample_replay_buffer
        )
        replay_instructions = {r["instruction"] for r in sample_replay_buffer}
        dataset_instructions = {p["instruction"] for p in dataset}
        assert replay_instructions.issubset(dataset_instructions)


class TestSaveAsJsonl:
    def test_jsonl_format(self, tmp_path):
        pairs = [
            {"instruction": "do task", "response": "step 1\nstep 2", "weight": 0.9},
        ]
        path = str(tmp_path / "train.jsonl")
        save_as_jsonl(pairs, path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert "messages" in obj
        assert obj["messages"][0]["role"] == "user"
        assert obj["messages"][1]["role"] == "assistant"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_abstract.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`cogmem/consolidation/abstract.py`:
```python
import json
from pathlib import Path


def episode_to_training_pair(episode: dict) -> dict | None:
    if not episode.get("success"):
        return None
    if not episode.get("script"):
        return None
    return {
        "instruction": episode["task_description"],
        "response": episode["script"],
        "weight": episode["q_value"],
        "source_episode": episode["episode_id"],
    }


def q_weighted_duplicates(pairs: list[dict]) -> list[dict]:
    result = []
    for pair in pairs:
        copies = max(1, round(pair["weight"] * 3))
        result.extend([pair] * copies)
    return result


def prepare_training_dataset(
    episodes: list[dict],
    replay_buffer: list[dict] | None = None,
) -> list[dict]:
    pairs = []
    for ep in episodes:
        pair = episode_to_training_pair(ep)
        if pair is not None:
            pairs.append(pair)

    if replay_buffer:
        for example in replay_buffer:
            pairs.append(
                {
                    "instruction": example["instruction"],
                    "response": example["response"],
                    "weight": 1.0,
                    "source_episode": "replay",
                }
            )

    return pairs


def save_as_jsonl(pairs: list[dict], path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    weighted = q_weighted_duplicates(pairs)
    with open(path, "w") as f:
        for pair in weighted:
            obj = {
                "messages": [
                    {"role": "user", "content": pair["instruction"]},
                    {"role": "assistant", "content": pair["response"]},
                ]
            }
            f.write(json.dumps(obj) + "\n")
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_abstract.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add cogmem/consolidation/abstract.py tests/test_abstract.py
git commit -m "feat: format converter for episodes to weighted JSONL training data"
```

---

## Task 8: LoRA Training via Together AI

**Files:**
- Create: `cogmem/consolidation/train_lora.py`
- Create: `tests/test_train_lora.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_train_lora.py`:
```python
import pytest
from unittest.mock import MagicMock, patch, mock_open
from cogmem.config import CogMemConfig
from cogmem.consolidation.train_lora import (
    upload_training_file,
    start_lora_job,
    wait_for_job,
    download_adapter,
    train_lora_together,
)


@pytest.fixture
def config():
    return CogMemConfig(together_api_key="test-key")


@patch("cogmem.consolidation.train_lora.Together")
def test_upload_training_file(mock_together_cls, tmp_path, config):
    mock_client = MagicMock()
    mock_together_cls.return_value = mock_client
    mock_client.files.upload.return_value = MagicMock(id="file-abc123")

    jsonl_path = str(tmp_path / "train.jsonl")
    with open(jsonl_path, "w") as f:
        f.write('{"messages": [{"role": "user", "content": "test"}]}\n')

    file_id = upload_training_file(jsonl_path, config)
    assert file_id == "file-abc123"
    mock_client.files.upload.assert_called_once()


@patch("cogmem.consolidation.train_lora.Together")
def test_start_lora_job(mock_together_cls, config):
    mock_client = MagicMock()
    mock_together_cls.return_value = mock_client
    mock_client.fine_tuning.create.return_value = MagicMock(id="ft-job-xyz")

    job_id = start_lora_job("file-abc123", config, suffix="test-run")
    assert job_id == "ft-job-xyz"
    call_kwargs = mock_client.fine_tuning.create.call_args[1]
    assert call_kwargs["training_file"] == "file-abc123"
    assert call_kwargs["lora"] is True
    assert call_kwargs["n_epochs"] == config.lora_epochs


@patch("cogmem.consolidation.train_lora.Together")
def test_wait_for_job_completed(mock_together_cls, config):
    mock_client = MagicMock()
    mock_together_cls.return_value = mock_client
    mock_client.fine_tuning.retrieve.return_value = MagicMock(
        status="completed", output_name="user/model-ft-xyz"
    )

    result = wait_for_job("ft-job-xyz", config, poll_interval=0.01)
    assert result["status"] == "completed"
    assert result["model_id"] == "user/model-ft-xyz"


@patch("cogmem.consolidation.train_lora.Together")
def test_wait_for_job_failed(mock_together_cls, config):
    mock_client = MagicMock()
    mock_together_cls.return_value = mock_client
    mock_client.fine_tuning.retrieve.return_value = MagicMock(
        status="failed", output_name=None
    )

    with pytest.raises(RuntimeError, match="failed"):
        wait_for_job("ft-job-xyz", config, poll_interval=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train_lora.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`cogmem/consolidation/train_lora.py`:
```python
import time

from together import Together


def _client(config) -> Together:
    return Together(api_key=config.together_api_key)


def upload_training_file(jsonl_path: str, config) -> str:
    client = _client(config)
    resp = client.files.upload(file=jsonl_path)
    return resp.id


def start_lora_job(file_id: str, config, suffix: str = "cogmem") -> str:
    client = _client(config)
    resp = client.fine_tuning.create(
        training_file=file_id,
        model=config.base_model,
        n_epochs=config.lora_epochs,
        learning_rate=config.lora_learning_rate,
        batch_size=config.lora_batch_size,
        lora=True,
        lora_r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        suffix=suffix,
        n_checkpoints=1,
    )
    return resp.id


def wait_for_job(
    job_id: str, config, poll_interval: float = 30.0, max_wait: float = 7200.0
) -> dict:
    client = _client(config)
    elapsed = 0.0
    while elapsed < max_wait:
        status = client.fine_tuning.retrieve(id=job_id)
        if status.status == "completed":
            return {"status": "completed", "model_id": status.output_name, "job_id": job_id}
        if status.status == "failed":
            raise RuntimeError(f"LoRA training job {job_id} failed")
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"LoRA training job {job_id} timed out after {max_wait}s")


def download_adapter(job_id: str, output_dir: str, config) -> str:
    client = _client(config)
    client.fine_tuning.download(id=job_id, output=output_dir, checkpoint_type="adapter")
    return output_dir


def train_lora_together(
    jsonl_path: str, config, policy_name: str = "q_top_k"
) -> dict:
    file_id = upload_training_file(jsonl_path, config)
    suffix = f"cogmem-{policy_name}"
    job_id = start_lora_job(file_id, config, suffix=suffix)
    result = wait_for_job(job_id, config)

    adapter_dir = f"{config.adapters_dir}/{policy_name}"
    download_adapter(job_id, adapter_dir, config)
    result["adapter_dir"] = adapter_dir

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_lora.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add cogmem/consolidation/train_lora.py tests/test_train_lora.py
git commit -m "feat: Together AI LoRA training with upload, poll, and adapter download"
```

---

## Task 9: Experiment Logging

**Files:**
- Create: `cogmem/utils/logging.py`

- [ ] **Step 1: Write implementation**

`cogmem/utils/logging.py`:
```python
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yaml


def save_results(results: dict, logs_dir: str, name: str) -> str:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = f"{logs_dir}/{name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def save_config_snapshot(config, logs_dir: str) -> str:
    from dataclasses import asdict

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    path = f"{logs_dir}/config_snapshot.yaml"
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
    return path


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "from cogmem.utils.logging import save_results, file_sha256; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add cogmem/utils/logging.py
git commit -m "feat: experiment logging utilities (JSON results, config snapshots, hashing)"
```

---

## Task 10: ALFWorld Evaluation with transformers+peft

**Files:**
- Create: `cogmem/evaluation/alfworld_eval.py`

This task involves heavy external dependencies (ALFWorld, transformers, peft, bitsandbytes). Unit testing requires the full model loaded. Instead, we write the module with a clear interface and test it via integration later.

- [ ] **Step 1: Write implementation**

`cogmem/evaluation/alfworld_eval.py`:
```python
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class LocalLoRAModel:
    def __init__(self, base_model: str, adapter_path: str | None = None):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        if adapter_path and Path(adapter_path).exists():
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_alfworld_task(
    task_description: str,
    env,
    model: LocalLoRAModel | None = None,
    llm_client=None,
    max_steps: int = 50,
) -> dict:
    """Run a single ALFWorld task. Uses either local model or LLM client.

    Returns: {"success": bool, "steps": int, "trajectory": list}
    """
    obs, info = env.reset()
    trajectory = []

    for step in range(max_steps):
        if model is not None:
            prompt = _build_prompt(task_description, obs, trajectory)
            action_text = model.generate(prompt)
        elif llm_client is not None:
            prompt = _build_prompt(task_description, obs, trajectory)
            action_text = llm_client.generate(prompt)
        else:
            raise ValueError("Either model or llm_client must be provided")

        action = _extract_action(action_text)
        obs, reward, done, info = env.step(action)
        trajectory.append({"step": step + 1, "action": action, "observation": obs})

        if done:
            return {"success": reward > 0, "steps": step + 1, "trajectory": trajectory}

    return {"success": False, "steps": max_steps, "trajectory": trajectory}


def _build_prompt(task: str, obs: str, trajectory: list[dict]) -> str:
    lines = [f"Task: {task}\n"]
    for t in trajectory[-5:]:  # last 5 steps for context window
        lines.append(f"Action: {t['action']}")
        lines.append(f"Observation: {t['observation']}")
    lines.append(f"Current observation: {obs}")
    lines.append("What is your next action?")
    return "\n".join(lines)


def _extract_action(text: str) -> str:
    text = text.strip()
    for prefix in ["Action:", "action:", "> "]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.split("\n")[0].strip()
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "from cogmem.evaluation.alfworld_eval import LocalLoRAModel; print('OK')"`
Expected: `OK` (or import warning about CUDA — OK on machines without GPU)

- [ ] **Step 3: Commit**

```bash
git add cogmem/evaluation/alfworld_eval.py
git commit -m "feat: ALFWorld evaluation with local transformers+peft LoRA inference"
```

---

## Task 11: Verification

**Files:**
- Create: `cogmem/consolidation/verify.py`
- Create: `tests/test_verify.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_verify.py`:
```python
import pytest
from cogmem.consolidation.verify import (
    binomial_ci,
    aggregate_seed_results,
    verification_passed,
)


class TestBinomialCI:
    def test_basic_ci(self):
        low, high = binomial_ci(successes=7, total=10, confidence=0.95)
        assert 0.3 < low < 0.7
        assert 0.8 < high <= 1.0

    def test_zero_successes(self):
        low, high = binomial_ci(successes=0, total=10, confidence=0.95)
        assert low == 0.0
        assert high > 0.0

    def test_all_successes(self):
        low, high = binomial_ci(successes=10, total=10, confidence=0.95)
        assert low > 0.0
        assert high == 1.0


class TestAggregateSeeds:
    def test_mean_and_std(self):
        results = [
            {"success_rate": 0.5, "successes": 5, "total": 10},
            {"success_rate": 0.6, "successes": 6, "total": 10},
            {"success_rate": 0.7, "successes": 7, "total": 10},
        ]
        agg = aggregate_seed_results(results)
        assert abs(agg["mean"] - 0.6) < 0.01
        assert agg["std"] > 0
        assert agg["n_seeds"] == 3


class TestVerificationPassed:
    def test_passes_when_better(self):
        assert verification_passed(
            consolidated_rate=0.6, baseline_rate=0.4, threshold=0.05
        )

    def test_passes_when_equal(self):
        assert verification_passed(
            consolidated_rate=0.4, baseline_rate=0.4, threshold=0.05
        )

    def test_fails_when_regressed(self):
        assert not verification_passed(
            consolidated_rate=0.3, baseline_rate=0.4, threshold=0.05
        )

    def test_passes_within_threshold(self):
        assert verification_passed(
            consolidated_rate=0.36, baseline_rate=0.4, threshold=0.05
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_verify.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`cogmem/consolidation/verify.py`:
```python
from statistics import mean, stdev

from scipy import stats


def binomial_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    low, high = stats.binom.interval(confidence, total, successes / total)
    return low / total, high / total


def run_verification_single_seed(
    holdout_episodes: list[dict],
    run_task_fn,
    seed: int,
) -> dict:
    import random
    random.seed(seed)

    successes = 0
    for ep in holdout_episodes:
        result = run_task_fn(ep["task_description"])
        if result.get("success"):
            successes += 1

    rate = successes / len(holdout_episodes) if holdout_episodes else 0
    ci_low, ci_high = binomial_ci(successes, len(holdout_episodes))

    return {
        "success_rate": rate,
        "successes": successes,
        "total": len(holdout_episodes),
        "ci_95": (ci_low, ci_high),
        "seed": seed,
    }


def aggregate_seed_results(results: list[dict]) -> dict:
    rates = [r["success_rate"] for r in results]
    return {
        "mean": mean(rates),
        "std": stdev(rates) if len(rates) > 1 else 0.0,
        "min": min(rates),
        "max": max(rates),
        "n_seeds": len(results),
        "per_seed": results,
    }


def verification_passed(
    consolidated_rate: float, baseline_rate: float, threshold: float = 0.05
) -> bool:
    return consolidated_rate >= baseline_rate - threshold
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_verify.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add cogmem/consolidation/verify.py tests/test_verify.py
git commit -m "feat: verification with binomial CIs and multi-seed aggregation"
```

---

## Task 12: Router

**Files:**
- Create: `cogmem/consolidation/router.py`
- Create: `tests/test_router.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_router.py`:
```python
import pytest
from dataclasses import dataclass
from cogmem.config import CogMemConfig
from cogmem.consolidation.router import ConsolidatedDomain, route_task


@pytest.fixture
def config():
    return CogMemConfig(consolidation_match_threshold=0.75, retrieval_min_q=0.3)


@pytest.fixture
def clean_domain():
    return ConsolidatedDomain(
        name="clean",
        centroid=[0.1] * 384,
        adapter_path="adapters/clean",
    )


class TestRouteTask:
    def test_routes_to_consolidated_when_similar(self, config, clean_domain):
        task_embedding = [0.1] * 384  # identical to centroid
        result = route_task(
            task_embedding=task_embedding,
            consolidated_domains=[clean_domain],
            memory_bank_episodes=[],
            config=config,
        )
        assert result[0] == "consolidated"
        assert result[1] == "adapters/clean"

    def test_routes_to_episodic_when_not_consolidated(self, config):
        task_embedding = [0.9] * 384
        episodes = [
            {"intent_embedding": [0.85] * 384, "q_value": 0.8, "episode_id": "ep_1"}
        ]
        result = route_task(
            task_embedding=task_embedding,
            consolidated_domains=[],
            memory_bank_episodes=episodes,
            config=config,
        )
        assert result[0] == "episodic"

    def test_routes_to_cold_when_nothing_matches(self, config):
        task_embedding = [0.99] * 384
        result = route_task(
            task_embedding=task_embedding,
            consolidated_domains=[],
            memory_bank_episodes=[],
            config=config,
        )
        assert result[0] == "cold"
        assert result[1] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`cogmem/consolidation/router.py`:
```python
from dataclasses import dataclass

import numpy as np


@dataclass
class ConsolidatedDomain:
    name: str
    centroid: list[float]
    adapter_path: str


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm)


def route_task(
    task_embedding: list[float],
    consolidated_domains: list[ConsolidatedDomain],
    memory_bank_episodes: list[dict],
    config,
) -> tuple[str, object]:
    # Check consolidated domains first
    for domain in consolidated_domains:
        sim = _cosine_sim(task_embedding, domain.centroid)
        if sim > config.consolidation_match_threshold:
            return ("consolidated", domain.adapter_path)

    # Fallback to episodic retrieval
    if memory_bank_episodes:
        scored = []
        for ep in memory_bank_episodes:
            emb = ep.get("intent_embedding")
            if emb:
                sim = _cosine_sim(task_embedding, emb)
                scored.append((sim, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][1].get("q_value", 0) > config.retrieval_min_q:
            top_episodes = [ep for _, ep in scored[:3]]
            return ("episodic", top_episodes)

    return ("cold", None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add cogmem/consolidation/router.py tests/test_router.py
git commit -m "feat: task router for consolidated/episodic/cold dispatch"
```

---

## Task 13: Comparison Tables

**Files:**
- Create: `cogmem/evaluation/compare.py`
- Create: `tests/test_compare.py`

- [ ] **Step 1: Write the failing test**

`tests/test_compare.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_compare.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`cogmem/evaluation/compare.py`:
```python
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
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_compare.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add cogmem/evaluation/compare.py tests/test_compare.py
git commit -m "feat: comparison table formatting and best-policy selection"
```

---

## Task 14: Pipeline — Experiment 1

**Files:**
- Create: `cogmem/consolidation/pipeline.py`
- Create: `cogmem/consolidation/prune.py`

- [ ] **Step 1: Write prune.py (simple utility)**

`cogmem/consolidation/prune.py`:
```python
def prune_consolidated(
    episodes: list[dict], consolidated_ids: set[str]
) -> list[dict]:
    return [ep for ep in episodes if ep["episode_id"] not in consolidated_ids]
```

- [ ] **Step 2: Write pipeline.py**

`cogmem/consolidation/pipeline.py`:
```python
import json
from pathlib import Path
from statistics import mean

from cogmem.config import CogMemConfig
from cogmem.consolidation.abstract import prepare_training_dataset, save_as_jsonl
from cogmem.consolidation.select import POLICIES
from cogmem.consolidation.train_lora import train_lora_together
from cogmem.consolidation.verify import (
    aggregate_seed_results,
    run_verification_single_seed,
    verification_passed,
)
from cogmem.evaluation.compare import best_policy, format_comparison_table, save_comparison
from cogmem.memory.memory_bank import MemoryBank
from cogmem.utils.logging import file_sha256, save_config_snapshot, save_results


def _load_replay_buffer(path: str) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    with open(path) as f:
        return json.load(f)


def run_consolidation(
    policy_name: str,
    available_episodes: list[dict],
    holdout_episodes: list[dict],
    replay_buffer: list[dict],
    config: CogMemConfig,
    run_task_fn=None,
) -> dict:
    # Select
    policy_fn = POLICIES[policy_name]
    selected = policy_fn(available_episodes, config)

    # Abstract + save JSONL
    training_pairs = prepare_training_dataset(selected, replay_buffer=replay_buffer)
    jsonl_dir = f"{config.logs_dir}/jsonl"
    jsonl_path = save_as_jsonl(training_pairs, f"{jsonl_dir}/{policy_name}.jsonl")

    # Train LoRA
    train_result = train_lora_together(jsonl_path, config, policy_name=policy_name)

    # Verify with multiple seeds
    seed_results = []
    if run_task_fn is not None:
        for seed in config.eval_seeds:
            r = run_verification_single_seed(holdout_episodes, run_task_fn, seed)
            seed_results.append(r)

    verification = aggregate_seed_results(seed_results) if seed_results else {}

    result = {
        "policy": policy_name,
        "episodes_selected": len(selected),
        "training_pairs": len(training_pairs),
        "q_value_mean": mean([ep["q_value"] for ep in selected]) if selected else 0,
        "verification": verification,
        "train_result": train_result,
        "jsonl_path": jsonl_path,
    }

    save_results(result, config.logs_dir, f"consolidation_{policy_name}")
    return result


def run_experiment_1(config: CogMemConfig, run_task_fn=None) -> dict:
    # Load and hash memory bank
    bank = MemoryBank.load(config.memory_bank_path)
    bank_hash = bank.sha256()

    # Determine holdout size
    n_success = len(bank.successful())
    holdout_n = 30 if n_success > 200 else config.verification_holdout

    # Split
    holdout, available = bank.stratified_holdout(n=holdout_n, seed=config.seed)

    # Replay buffer
    replay_buffer = _load_replay_buffer(config.replay_buffer_path)

    # Save reproducibility artifacts
    save_config_snapshot(config, config.logs_dir)

    all_results = {"memory_bank_hash": bank_hash, "holdout_ids": [ep["episode_id"] for ep in holdout]}

    for policy_name in ["q_top_k", "recency", "frequency", "random", "all"]:
        print(f"\n{'=' * 60}")
        print(f"Running consolidation: {policy_name}")
        print(f"{'=' * 60}")
        result = run_consolidation(
            policy_name, available, holdout, replay_buffer, config, run_task_fn
        )
        all_results[policy_name] = result

    # Print comparison
    policy_results = {k: v for k, v in all_results.items() if k in POLICIES}
    print("\n" + format_comparison_table(policy_results))
    print(f"\nBest policy: {best_policy(policy_results)}")

    save_comparison(all_results, f"{config.logs_dir}/experiment_1_results.json")
    return all_results
```

- [ ] **Step 3: Verify it imports cleanly**

Run: `python -c "from cogmem.consolidation.pipeline import run_experiment_1; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add cogmem/consolidation/pipeline.py cogmem/consolidation/prune.py
git commit -m "feat: Experiment 1 pipeline orchestration with 5-policy comparison"
```

---

## Task 15: Pipeline — Experiment 2

**Files:**
- Modify: `cogmem/consolidation/pipeline.py`

- [ ] **Step 1: Add run_experiment_2 to pipeline.py**

Append to `cogmem/consolidation/pipeline.py`:

```python
def run_experiment_2(config: CogMemConfig, exp1_results: dict, run_task_fns: dict) -> dict:
    """Full system ablation. run_task_fns maps variant name to a callable.

    Expected keys in run_task_fns:
    - "cold_3b": base 3B, no adapter, no memory
    - "memrl_3b": base 3B with full episodic retrieval
    - "consolidated_3b": base 3B + best LoRA from Exp 1, no memory
    - "cogmem_3b": base 3B + best LoRA + router (consolidated + episodic fallback)
    - "cold_8b": Groq 8B, no memory (optional)
    - "cold_70b": Groq 70B, no memory (optional)
    """
    bank = MemoryBank.load(config.memory_bank_path)
    holdout_n = 30 if len(bank.successful()) > 200 else config.verification_holdout
    holdout, _ = bank.stratified_holdout(n=holdout_n, seed=config.seed)

    all_results = {}
    for variant_name, run_fn in run_task_fns.items():
        print(f"\nEvaluating variant: {variant_name}")
        seed_results = []
        for seed in config.eval_seeds:
            r = run_verification_single_seed(holdout, run_fn, seed)
            seed_results.append(r)
        all_results[variant_name] = aggregate_seed_results(seed_results)

    print("\n" + format_comparison_table(
        {k: {"verification": v, "episodes_selected": "-"} for k, v in all_results.items()}
    ))

    save_comparison(all_results, f"{config.logs_dir}/experiment_2_results.json")
    return all_results
```

- [ ] **Step 2: Verify imports**

Run: `python -c "from cogmem.consolidation.pipeline import run_experiment_2; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add cogmem/consolidation/pipeline.py
git commit -m "feat: Experiment 2 full system ablation (6 variants)"
```

---

## Task 16: MemRL Conversion Script

**Files:**
- Create: `scripts/convert_memrl.py`
- Create: `tests/test_convert_memrl.py`

- [ ] **Step 1: Write the failing test**

`tests/test_convert_memrl.py`:
```python
import json
import pytest
from scripts.convert_memrl import convert_textual_memory_item, convert_cube_dump


@pytest.fixture
def sample_memrl_item():
    """Simulates a single item from MemRL's textual_memory.json."""
    return {
        "id": "mem-uuid-001",
        "memory": "put a clean mug in shelf 1",
        "metadata": {
            "type": "procedure",
            "source_benchmark": "alfworld",
            "full_content": "Task: put a clean mug in shelf 1\n\nProcedure:\n1. Find mug on countertop 1\n2. Take mug\n3. Go to sinkbasin 1\n4. Clean mug\n5. Go to shelf 1\n6. Put mug in shelf 1",
            "success": True,
            "q_value": 0.89,
            "q_visits": 5,
            "confidence": 0.85,
            "q_updated_at": "2026-04-01T10:30:00",
            "last_used_at": "2026-04-01T10:30:00",
        },
    }


def test_convert_single_item(sample_memrl_item):
    result = convert_textual_memory_item(sample_memrl_item, episode_num=1)
    assert result["episode_id"] == "ep_001"
    assert result["task_description"] == "put a clean mug in shelf 1"
    assert result["success"] is True
    assert result["q_value"] == 0.89
    assert result["q_visits"] == 5
    assert "script" in result
    assert result["script"].startswith("1.")


def test_convert_extracts_task_type(sample_memrl_item):
    result = convert_textual_memory_item(sample_memrl_item, episode_num=1)
    assert result["task_type"] == "clean"


def test_convert_cube_dump(tmp_path, sample_memrl_item):
    cube_dir = tmp_path / "cube"
    cube_dir.mkdir()
    (cube_dir / "textual_memory.json").write_text(json.dumps([sample_memrl_item]))

    output_path = str(tmp_path / "memory_bank.json")
    convert_cube_dump(str(cube_dir), output_path)

    with open(output_path) as f:
        bank = json.load(f)
    assert len(bank) == 1
    assert bank[0]["episode_id"] == "ep_001"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_convert_memrl.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`scripts/convert_memrl.py`:
```python
"""Convert MemRL cube dump to CogMem memory bank JSON format.

Usage:
    python scripts/convert_memrl.py <cube_dir> <output_path>

Example:
    python scripts/convert_memrl.py MemRL/results/snapshot/10/cube results/memory_bank.json
"""

import json
import re
import sys
from pathlib import Path

TASK_TYPE_PATTERNS = {
    "clean": r"clean",
    "heat": r"hot|heat",
    "cool": r"cool|cold",
    "examine": r"examine|look",
    "puttwo": r"put two|puttwo",
    "pick": r"put a|put an|place",
}


def _infer_task_type(task_description: str) -> str:
    desc = task_description.lower()
    for task_type, pattern in TASK_TYPE_PATTERNS.items():
        if re.search(pattern, desc):
            return task_type
    return "unknown"


def _extract_script(full_content: str) -> str:
    lines = full_content.split("\n")
    script_lines = []
    in_procedure = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("procedure:"):
            in_procedure = True
            continue
        if in_procedure and stripped:
            script_lines.append(stripped)
        elif re.match(r"^\d+\.", stripped):
            script_lines.append(stripped)
    return "\n".join(script_lines) if script_lines else full_content


def convert_textual_memory_item(item: dict, episode_num: int) -> dict:
    meta = item.get("metadata", {})
    full_content = meta.get("full_content", "")
    task_desc = item.get("memory", "")

    return {
        "episode_id": f"ep_{episode_num:03d}",
        "task_description": task_desc,
        "task_type": _infer_task_type(task_desc),
        "script": _extract_script(full_content),
        "intent_embedding": [],  # re-embed with LocalEmbedder later
        "success": meta.get("success", False),
        "q_value": meta.get("q_value", 0.0),
        "q_visits": meta.get("q_visits", 0),
        "num_steps": 0,  # not available in cube dump
        "timestamp": meta.get("q_updated_at", ""),
    }


def convert_cube_dump(cube_dir: str, output_path: str) -> str:
    tm_path = Path(cube_dir) / "textual_memory.json"
    with open(tm_path) as f:
        items = json.load(f)

    episodes = []
    for i, item in enumerate(items, 1):
        ep = convert_textual_memory_item(item, episode_num=i)
        episodes.append(ep)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(episodes, f, indent=2)

    print(f"Converted {len(episodes)} episodes to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert_cube_dump(sys.argv[1], sys.argv[2])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_convert_memrl.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/convert_memrl.py tests/test_convert_memrl.py
git commit -m "feat: MemRL cube dump to CogMem memory bank converter"
```

---

## Task 17: Phase 0 Pre-Flight Script

**Files:**
- Create: `scripts/run_phase0.py`

- [ ] **Step 1: Write pre-flight script**

`scripts/run_phase0.py`:
```python
"""Phase 0: Pre-flight checks before spending compute or money.

Run this first. It verifies:
1. Together AI supports the target model for fine-tuning
2. Ollama is running and serves llama3.2:3b
3. ALFWorld is installed and data is downloaded
4. Required Python packages are importable

Usage:
    python scripts/run_phase0.py
"""

import json
import sys


def check_ollama():
    print("[1/4] Checking Ollama...")
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        resp = client.chat.completions.create(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content
        print(f"  OK — Ollama responded: {text.strip()[:50]}")
        return True
    except Exception as e:
        print(f"  FAIL — {e}")
        print("  Fix: run 'ollama pull llama3.2:3b && ollama serve'")
        return False


def check_together_api():
    print("[2/4] Checking Together AI model support...")
    try:
        import os
        from together import Together
        key = os.environ.get("TOGETHER_API_KEY", "")
        if not key:
            print("  WARN — TOGETHER_API_KEY not set. Set it before Phase 2.")
            return True  # non-blocking
        client = Together(api_key=key)
        models = client.models.list()
        model_ids = [m.id for m in models]
        target = "meta-llama/Llama-3.2-3B-Instruct"
        if target in model_ids:
            print(f"  OK — {target} is available")
        else:
            print(f"  WARN — {target} not in model list. Check fine-tuning docs.")
        return True
    except Exception as e:
        print(f"  WARN — Could not verify: {e}")
        return True  # non-blocking


def check_alfworld():
    print("[3/4] Checking ALFWorld...")
    try:
        import alfworld
        print(f"  OK — alfworld {alfworld.__version__} installed")
        return True
    except ImportError:
        print("  FAIL — alfworld not installed")
        print("  Fix: pip install alfworld && alfworld-download")
        return False


def check_packages():
    print("[4/4] Checking Python packages...")
    required = [
        "torch", "transformers", "peft", "bitsandbytes",
        "sentence_transformers", "openai", "together", "numpy", "scipy", "yaml",
    ]
    all_ok = True
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  OK — {pkg}")
        except ImportError:
            print(f"  FAIL — {pkg} not found")
            all_ok = False
    return all_ok


def main():
    print("=" * 50)
    print("CogMem Phase 0: Pre-Flight Checks")
    print("=" * 50 + "\n")

    results = [
        check_ollama(),
        check_together_api(),
        check_alfworld(),
        check_packages(),
    ]

    print("\n" + "=" * 50)
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} checks passed")
    if all(results):
        print("All checks passed. Ready for Phase 1.")
    else:
        print("Some checks failed. Fix issues before proceeding.")
    print("=" * 50)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the script runs**

Run: `python scripts/run_phase0.py`
Expected: Prints check results. Some may fail (Ollama not running, ALFWorld not installed) — that's fine, this is the point of the pre-flight script.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_phase0.py
git commit -m "feat: Phase 0 pre-flight checks for Ollama, Together AI, ALFWorld"
```

---

## Task 18: Full Test Suite Verification

- [ ] **Step 1: Run the complete test suite**

Run: `cd F:/Newgeneration/CogMem && pytest tests/ -v --tb=short`
Expected: All tests pass (approximately 40+ tests across all modules)

- [ ] **Step 2: Save frozen requirements**

Run: `pip freeze > requirements_frozen.txt`

- [ ] **Step 3: Final commit**

```bash
git add requirements_frozen.txt
git commit -m "chore: pin dependency versions for reproducibility"
```

- [ ] **Step 4: Verify git log looks clean**

Run: `git log --oneline`
Expected: ~15 commits, one per task, clean history

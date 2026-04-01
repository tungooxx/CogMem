# CogMem: Q-Value Guided Memory Consolidation for AI Agents

**Date:** 2026-04-02
**Status:** Approved — ready for implementation planning

## Overview

CogMem is a research system that uses RL-learned Q-values to decide which episodic memories should be consolidated into LoRA model weights during offline "sleep" phases. The core hypothesis: selecting high-Q-value episodes for consolidation produces better LoRA adapters than alternative selection heuristics (recency, frequency, random, all).

The system has two phases:
1. **Phase 1 (Collection):** Run MemRL on ALFWorld to collect ~300 episodes with Q-values
2. **Phase 2 (Consolidation):** Select high-Q episodes, abstract them into training data, train LoRA adapters via Together AI, evaluate locally

## Hardware Constraints

| Resource | Capability |
|----------|-----------|
| GPU (RTX 1650 / RTX 3050) | 4GB VRAM each |
| Local LLM inference | Llama 3.2-3B via Ollama (~2.5GB VRAM) |
| Local LoRA eval inference | 3B 4-bit + LoRA adapter via transformers+peft (~3GB VRAM) |
| Local embeddings | sentence-transformers on CPU/GPU |
| LoRA training | Together AI API (cloud) or Kaggle T4 (free fallback) |
| Cold baselines (8B, 70B) | Groq free tier |

**Key constraint:** Training is cloud-only. Inference (both episode collection and evaluation) is local.

---

## Phase 1: Episode Collection

### Architecture

```
Ollama (3B) ──► MemRL Agent ──► ALFWorld Env
                    │
                    ▼
              Memory Service
              (Q-values, proceduralized scripts,
               embeddings — all persisted)
```

### MemRL Configuration

Modify `configs/rl_alf_config.yaml`:

```yaml
llm:
  provider: openai
  model: llama3.2:3b
  base_url: http://localhost:11434/v1
  api_key: ollama
  temperature: 0.0
  max_tokens: 4096

embedding:
  provider: local
  model: all-MiniLM-L6-v2
  device: cuda  # or cpu
```

Existing MemRL defaults (keep unchanged):
- sections: 10, batch_size: 32 → ~320 episodes total
- max_steps: 50 per episode
- Q-learning: alpha=0.3, gamma=0.0, rewards +-1.0

**Note on gamma=0.0:** No discounting is correct for ALFWorld because episodes are independent — success/failure is determined within a single episode with no cross-episode dependencies. For benchmarks with sequential dependencies (e.g., Lifelong Agent Bench), gamma should be increased.

### Using Proceduralized Scripts Directly

MemRL already compresses raw trajectories into LLM-generated scripts during its proceduralization step. These scripts are stored in the memory bank as `full_content`. Rather than patching MemRL to log raw trajectories and then re-abstracting them with a second LLM call, we use the proceduralized scripts directly as training data. The conversion to (instruction, response) JSONL format is a simple format transform — no LLM call needed.

This eliminates:
- A trajectory logging patch to MemRL (less code to maintain)
- Abstraction LLM calls in Phase 2 (faster, no quality loss from 3B reformatting)
- A potential source of noise in the training data

### 3B Fallback Logic

After 30 episodes, check success rate:
- If >= 10%: continue with 3B locally
- If < 10%: pivot to Groq (llama-3.1-8b-instant, 30 RPM, ~6 hours, $0)
  - `base_url` → `https://api.groq.com/openai/v1`
  - `api_key` → `$GROQ_API_KEY`
  - `model` → `llama-3.1-8b-instant`
  - Add exponential backoff retry for rate limits
  - Paper framing: "knowledge from 8B distilled into 3B via consolidation"

### Output: Memory Bank JSON

Conversion script transforms MemRL's `TextualMemoryItem` objects (including proceduralized scripts) into:

```json
{
  "episode_id": "ep_001",
  "task_description": "put a clean mug in shelf 1",
  "task_type": "clean",
  "intent_embedding": [0.12, -0.34, "..."],
  "trajectory": [
    {"step": 1, "action": "go to coffeetable 1", "observation": "You arrive at..."}
  ],
  "success": true,
  "q_value": 0.89,
  "q_visits": 5,
  "num_steps": 6,
  "timestamp": "2026-04-02T10:00:00"
}
```

Key fields: `task_description`, `trajectory`, `success`, `q_value`, `intent_embedding`.

Summary metrics also saved (totals, success rates by type, Q-value distribution stats).

---

## Phase 2: Consolidation Pipeline

### Directory Structure

```
cogmem/
├── __init__.py
├── config.py                  # CogMemConfig dataclass
├── consolidation/
│   ├── __init__.py
│   ├── select.py              # 5 selection policies
│   ├── abstract.py            # Format converter: scripts → JSONL (no LLM call)
│   ├── train_lora.py          # LoRA training via Together AI
│   ├── verify.py              # Post-consolidation verification
│   ├── prune.py               # Episode pruning after consolidation
│   ├── router.py              # Decides LoRA vs episodic retrieval at runtime
│   └── pipeline.py            # End-to-end orchestration (Exp 1 + Exp 2)
├── memory/
│   ├── __init__.py
│   ├── memory_bank.py         # Load/save/query episode memory bank
│   └── embeddings.py          # Local embedding model wrapper
├── evaluation/
│   ├── __init__.py
│   ├── alfworld_eval.py       # Run model on ALFWorld tasks
│   └── compare.py             # Compare multiple policies/models
└── utils/
    ├── __init__.py
    ├── llm_client.py          # Unified LLM client (Ollama/Groq/Together)
    └── logging.py             # Experiment logging
```

### Selection Policies (select.py)

Five policies, each returns a list of episode IDs:

| Policy | Method | Role |
|--------|--------|------|
| `q_top_k` | Episodes with Q >= 0.7, sorted by Q descending | **Novel contribution** |
| `recency` | N most recent episodes | "Let Them Sleep" baseline |
| `frequency` | N highest q_visits episodes | Heuristic baseline |
| `random` | N random episodes (seed=42) | Random baseline |
| `all` | Every episode | No-filter baseline |

**Critical:** recency, frequency, random all select the SAME N as q_top_k for fair comparison.

### Abstraction (abstract.py)

Simple format converter (no LLM call). For each selected successful episode:
1. Extract `task_description` as instruction, `full_content` (proceduralized script) as response
2. Convert to (instruction, response) JSONL format for LoRA training
3. Add replay buffer (from MemRL's few-shot ALFWorld examples) to prevent catastrophic forgetting

### LoRA Training (train_lora.py)

**Q-value weighting via example duplication:**
Together AI's fine-tuning API does not support per-example loss weighting. To implement Q-value weighting, duplicate training examples proportional to their Q-value:
```
copies = max(1, round(episode.q_value * 3))
# Q=0.9 → 3 copies, Q=0.7 → 2 copies, Q=0.4 → 1 copy
```
This ensures the gradient is dominated by high-utility episodes. The paper should acknowledge this approximation and note that exact loss weighting (via custom Trainer) is a refinement.

**Primary path: Together AI**
1. Convert training pairs to JSONL (messages format, with Q-weighted duplication)
2. Upload file to Together AI
3. Start LoRA fine-tuning job (base: Llama-3.2-3B-Instruct, rank=16, alpha=32, epochs=3)
4. Poll until complete
5. Download adapter weights

**Fallback: Kaggle T4 notebook**
- If Together AI doesn't support Llama 3.2-3B
- Upload JSONL as Kaggle dataset
- Run QLoRA training in notebook (T4 has 16GB VRAM)
- Download adapter weights

### Evaluation Inference (alfworld_eval.py)

```
Base model (3B, 4-bit quantized) ──► transformers + bitsandbytes
                                          │
LoRA adapter weights ──────────────► peft (loaded on top)
                                          │
                                     ALFWorld eval loop (local, $0)
```

NOT Ollama. The evaluation path uses `transformers` + `peft` + `bitsandbytes` to load the 4-bit quantized base model with the LoRA adapter. This fits in ~3GB VRAM on the RTX 3050.

### Verification (verify.py)

- 20 held-out episodes (increase to 30 if total successful episodes > 200), stratified by task type, fixed seed=42
- Same holdout across all 5 policies
- Run each consolidated model (base+LoRA) on holdout tasks
- Run base model (no adapter) as baseline
- Run each policy evaluation with 3 random seeds, report mean +/- std
- Report results with 95% binomial confidence intervals
- Pass if: consolidated rate >= baseline rate - 0.05

### Pipeline (pipeline.py)

`run_experiment_1()` orchestrates:
1. Load memory bank
2. Split holdout (stratified, seeded)
3. For each of 5 policies:
   - Select episodes
   - Convert to training JSONL (format transform, no LLM call)
   - Train LoRA (Together AI)
   - Download adapter weights
   - Verify on holdout (3 seeds, confidence intervals)
4. Record MemRL baseline (no consolidation)
5. Print comparison table, save all results to JSON

### Router (router.py)

Decides how to handle a new task at runtime — needed for Experiment 2 (full system ablation):

```python
def route_task(task_description, consolidated_domains, memory_bank, config):
    """Returns one of:
    - ("consolidated", adapter_path) → use base + LoRA, no retrieval
    - ("episodic", retrieved_memories) → use MemRL-style retrieval
    - ("cold", None) → no relevant memory, use base model only
    """
    task_embedding = embed(task_description)

    # Check if task falls within a consolidated domain
    for domain in consolidated_domains:
        similarity = cosine_sim(task_embedding, domain.centroid)
        if similarity > config.consolidation_match_threshold:  # e.g., 0.75
            return ("consolidated", domain.adapter_path)

    # Fall back to episodic retrieval
    retrieved = memrl_retrieve(task_embedding, memory_bank)
    if retrieved and retrieved[0].q_value > config.retrieval_min_q:
        return ("episodic", retrieved)

    return ("cold", None)
```

### Experiment 2: Full System Ablation (pipeline.py)

`run_experiment_2()` compares 4 system variants on ALFWorld test split:

| Variant | LoRA Adapter | Episodic Memory | Provider |
|---------|-------------|-----------------|----------|
| `cold_3b` | None | None | Local |
| `memrl_3b` | None | Full bank | Local |
| `consolidated_3b` | Best from Exp 1 | None | Local |
| `cogmem_3b` | Best from Exp 1 | Reduced (via router) | Local |
| `cold_8b` | None | None | Groq |
| `cold_70b` | None | None | Groq |

The `cogmem_3b` variant uses the router to dispatch tasks: consolidated domains go through LoRA, unfamiliar tasks fall back to episodic retrieval, truly novel tasks get cold inference.

---

## Phase 0: Pre-Flight Checklist

Before spending any money or significant compute time:

- [ ] Check Together AI supported models for fine-tuning: https://docs.together.ai/docs/fine-tuning#supported-models — if Llama 3.2-3B is NOT listed, switch primary LoRA training to Kaggle T4
- [ ] Verify Ollama serves llama3.2:3b on the OpenAI-compatible endpoint
- [ ] Confirm ALFWorld downloads and initializes correctly
- [ ] Run 30 test episodes to check 3B success rate (>= 10% threshold)

## Provider Matrix

| Task | Provider | Model | Cost |
|------|----------|-------|------|
| Episode collection (Phase 1) | Ollama (local) | llama3.2:3b | $0 |
| Episode collection fallback | Groq | llama-3.1-8b-instant | $0 |
| Embeddings | Local sentence-transformers | all-MiniLM-L6-v2 | $0 |
| LoRA training | Together AI | Llama-3.2-3B-Instruct | ~$5/run |
| LoRA training fallback | Kaggle T4 | Llama-3.2-3B-Instruct | $0 |
| Evaluation inference | Local transformers+peft | 3B 4-bit + LoRA | $0 |
| Cold baselines (8B, 70B) | Groq | llama-3.1-8b/70b-instant | $0 |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| 3B too weak for ALFWorld (<10% success) | Phase 1 invalid | Pivot to Groq 8B after 30 episodes |
| Together AI doesn't support 3B fine-tuning | Can't train LoRA | Phase 0 check; fallback to Kaggle T4 (free) |
| 5 training runs too expensive | Budget overrun | Check Together AI pricing in Phase 0; fallback to Kaggle |
| Holdout too small (20 episodes) | Noisy results | Report 95% CIs; increase to 30 if >200 successful episodes; 3 seeds |

## Success Criteria

After Phase 1:
- `results/memory_bank.json` with ~300 episodes, each with Q-values
- Baseline success rate for "MemRL on 3B on ALFWorld"

After Phase 2:
- Comparison table: 5 selection policies + MemRL baseline
- Q-top-k should have best or near-best verification score
- Evidence that Q-value selection produces better LoRA adapters
- Saved adapter weights (or model IDs) for each policy
- All results in JSON for paper tables

## Reproducibility

- All random seeds fixed: seed=42 for selection, seed=42 for holdout split
- Save exact training JSONL files for each policy
- Log Together AI job IDs (or Kaggle notebook URLs)
- Pin package versions: save `pip freeze > requirements_frozen.txt`
- Save the exact `memory_bank.json` used (SHA-256 hash it)
- All configs saved as YAML alongside results

## Future: Tier 3 Integration

The consolidation pipeline is designed modularly so that Tier 3 (executable skill crystallization with multi-signal triggers) can be added on top of the cluster detection in `select.py`. The `cogmem/` package structure supports adding a `skills/` submodule later.

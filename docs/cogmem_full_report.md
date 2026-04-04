# CogMem — Full Project Report (for Paper & Presentation)

## 1. Project Overview

**CogMem** (Cognitive Memory Consolidation) is a research system that improves small language models on sequential decision tasks by consolidating experience into LoRA weight updates, guided by Q-values from reinforcement learning.

**Key Insight:** MemRL uses Q-values to decide which memories to RETRIEVE. CogMem inverts this — using Q-values to decide which experiences to TRAIN on. Low Q (hard tasks) get more training emphasis.

**Environment:** ALFWorld — text-based household tasks (clean, heat, cool, examine, pick, puttwo)

**Base Model:** llama3.2:3b-Instruct (3.2B parameters, Q4_K_M quantization)

## 2. Architecture

```
Phase 0: Pre-flight checks (Ollama, ALFWorld, packages)
Phase 1: Collect Q-valued episodes from MemRL on ALFWorld
Phase 2: Q-weighted LoRA training using expert trajectories
Phase 2.5: Evaluation (base 3b vs CogMem 3b)
Phase 3: Crystallize skill (future)
Phase 4: Full experiment + cold baselines (future)
```

### CogMem Pipeline
```
MemRL episodes → Q-values per task → Match to expert trajectories
→ Q-weighted training data → QLoRA fine-tuning → CogMem-3b model
```

### Training Data Evolution
| Version | Format | Content | Issues |
|---------|--------|---------|--------|
| v1 | Raw task → expert plan | Task description + full expert action sequence | Format mismatch with MemRL prompts |
| v2 | MemRL format → first action | System prompt + few-shot + task → expert first action | Only MemRL format |
| **v3** | **Both formats** | **MemRL format + Raw format, both Q-weighted** | **Current version** |

### Why v3 (Dual Format)?
- **MemRL format**: Model learns to respond correctly within MemRL's context (system prompt + few-shot examples + "Now, it's your turn")
- **Raw format**: Model learns to respond to plain task descriptions (general use, not tied to MemRL)
- Both weighted by Q-values: hard tasks (low Q) get up to 5x more training copies

## 3. Phase 1 — Episode Collection

### Configuration
| Parameter | Value |
|-----------|-------|
| Compute | Paperspace A4000 (16GB VRAM) |
| LLM | llama3.2:3b via Ollama |
| Embedding | nomic-embed-text (768-dim) |
| Framework | MemRL (MemTensor/MemRL) |
| Games sampled | 248 (~7% of 3553 total) |
| Batch size | 8 |
| Max steps | 20 |
| Sections | 1 |

### Results
| Metric | Value |
|--------|-------|
| Total memories | 248 |
| Successful games | 9 (~3.6% success rate) |
| Q-value range | [-0.882, 0.000] |
| Q-value mean | -0.613 |
| Task types | examine (141), clean (42), cool (39), heat (26) |

### Key Observation
- 3b model has ~3.6% success rate on ALFWorld zero-shot
- 8b model achieves ~100% success rate (14/14 games tested separately)
- This gap motivates CogMem: can Q-weighted LoRA close it?

### Meta-Reflection Problem
- MemRL generates "proceduralized reflections" after each episode
- With 3b model, these reflections are garbage: "avoid repetitive tasks", "provide clear instructions"
- With 8b model, reflections are useful: actual procedural knowledge
- Solution for cycle 1: use expert trajectories instead of reflections
- Solution for cycle 2+: improved model generates better reflections

## 4. Phase 2 — LoRA Training

### Training Data (v3)
| Metric | Value |
|--------|-------|
| Total examples | 1568 |
| MemRL format | 784 |
| Raw format | 784 |
| Unique tasks | 237 |
| Avg Q-weighted copies | 3.3 per task |
| Q-weighting | abs(Q) → 1-5 copies (lower Q = more copies) |

### Expert Trajectory Source
- ALFWorld ships with 2,435 expert trajectories (human-annotated)
- Each trajectory has: GotoLocation, PickupObject, PutObject, CleanObject, HeatObject, CoolObject, ToggleObject
- Converted to ReAct format: "Thought: I need to... Action: go to dresser 1"

### Q-Value Weighting Logic
```python
def q_to_copies(q_value, max_copies=5):
    # Q range: [-0.88, 0.0]
    # Lower Q = more copies (model struggles more)
    normalized = abs(q_value)  # 0.0 to 0.88
    copies = 1 + int(normalized * (max_copies - 1) / 0.9)
    return min(copies, max_copies)

# Q = -0.88 → 5 copies (hardest tasks)
# Q = -0.50 → 3 copies
# Q = 0.00 → 1 copy (easiest tasks)
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Base model | meta-llama/Llama-3.2-3B-Instruct |
| Method | QLoRA (4-bit NF4 quantization) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Dropout | 0.05 |
| Epochs | 3 |
| Learning rate | 1e-5 |
| Batch size | 4 (gradient accumulation: 2) |
| Optimizer | paged_adamw_8bit |
| Trainable parameters | 24,313,856 (0.75% of total) |
| Compute | Paperspace A4000, ~40 min |

### Training Results (v1 — raw format only)
| Step | Loss |
|------|------|
| 5 | 2.16 |
| 50 | 1.35 |
| 100 | 0.60 |
| 200 | 0.23 |
| 290 | 0.21 |

### v1 Evaluation Problem
- Trained on raw format (just task description)
- MemRL sends system prompt + few-shot examples + formatted task
- Model didn't recognize the MemRL format → generated Python code instead of actions
- **Fix: v3 training data includes both formats**

## 5. CogMem Cycle Theory

### Cycle 1 (Current)
```
Base 3b → MemRL episodes → Q-values → Expert actions (weighted by Q) → LoRA → CogMem-3b v1
```
- Expert actions because base 3b can't reflect well

### Cycle 2 (Future)
```
CogMem-3b v1 → MemRL episodes → New Q-values → Model's own reflections (weighted by Q) → LoRA → CogMem-3b v2
```
- Improved model generates useful meta-reflections
- No longer needs expert data

### Cycle N
```
CogMem-3b v(N-1) → Better episodes → Better Q-values → Better reflections → Better LoRA → CogMem-3b vN
```
- Each cycle: model improves → better reflections → better training → better model
- Potential to outperform much larger models on ALFWorld

## 6. Infrastructure Notes

### Split Environment (Local Development)
- **Windows**: Ollama (LLM + embeddings), ML stack
- **WSL Ubuntu**: ALFWorld + TextWorld (Linux-only)
- **Radmin VPN**: Connected GTX 1650 + RTX 3050 for distributed inference
- **Custom load balancer**: Round-robin across multiple Ollama instances

### Paperspace (Production)
- A4000 (16GB VRAM)
- CUDA 12.4, Driver 550
- torch 2.1.1+cu121 (original machine) or 2.6+cu124 (newer machines)
- transformers==4.43.4 (pinned for torch 2.1.1 compatibility)

### Dependency Issues Encountered
1. Ollama 128K context OOM on GTX 1650 (fix: 4K context)
2. WSL localhost ≠ Windows localhost (fix: gateway IP 172.31.192.1)
3. Embedding dimension mismatch: MemOS hardcoded 3072, nomic-embed-text is 768
4. chonkie API change: `tokenizer_or_token_counter` → `tokenizer`
5. Paperspace torch/transformers version conflicts
6. MemRL only saves cube dumps at section boundaries (data loss risk)
7. Training format mismatch: raw vs MemRL prompt format

## 7. Key Numbers for Presentation

| Metric | Value |
|--------|-------|
| Base model size | 3.2B parameters |
| LoRA trainable params | 24.3M (0.75%) |
| Training data | 1,568 examples (v3) |
| Training time | ~40 min on A4000 |
| Training loss | 2.16 → 0.21 |
| Base 3b success rate | ~3.6% on ALFWorld |
| 8b success rate | ~100% (reference) |
| CogMem 3b success rate | TBD (Phase 2.5) |
| Expert trajectories used | 2,435 (ALFWorld built-in) |
| Q-value range | [-0.882, 0.000] |
| Total Phase 1 time | ~15 hours |
| Total Phase 2 time | ~1 hour (including setup) |

## 8. Paper Comparison Table (Expected)

| Method | Model | Memory | LoRA | Success Rate |
|--------|-------|--------|------|-------------|
| Cold (no memory) | 3b | No | No | ~3.6% |
| MemRL (retrieval) | 3b | Yes | No | ~4-8% (est.) |
| SFT uniform | 3b | No | Yes (uniform) | TBD |
| **CogMem (Q-weighted)** | **3b** | **No** | **Yes (Q-weighted)** | **TBD** |
| Cold | 8b | No | No | ~100% |

## 9. Repository Structure

```
CogMem/
├── cogmem/
│   ├── consolidation/
│   │   ├── select.py          # 5 selection policies (q_top_k, recency, frequency, random, all)
│   │   ├── abstract.py        # Episode → training pair conversion
│   │   ├── train_lora.py      # Together AI LoRA training
│   │   ├── train_lora_local.py # Local PEFT/QLoRA training
│   │   ├── verify.py          # Binomial CI verification
│   │   ├── router.py          # Task routing (consolidated vs episodic)
│   │   ├── prune.py           # tag_consolidated() function
│   │   └── pipeline.py        # Experiment orchestration
│   ├── memory/
│   │   └── memory_bank.py     # Memory bank with stratified holdout
│   ├── evaluation/
│   │   └── compare.py         # Policy comparison + formatting
│   └── config.py              # CogMemConfig dataclass
├── scripts/
│   ├── run_phase0.py           # Pre-flight checks
│   ├── convert_memrl.py        # Cube dump → memory bank
│   ├── build_training_data.py  # v1: raw format
│   ├── build_training_data_v2.py # v2: MemRL format
│   └── build_training_data_v3.py # v3: both formats
├── results/
│   ├── memory_bank.json        # 248 memories with Q-values
│   ├── training.jsonl          # v1 training data (784 examples)
│   └── training_v3.jsonl       # v3 training data (1568 examples)
├── docs/
│   ├── phase1_results_v1.md    # Phase 1 detailed report
│   └── cogmem_full_report.md   # This file
├── paperspace.ipynb            # Phase 1 notebook
├── paperspace_phase2.ipynb     # Phase 2 notebook
└── paperspace_phase2_5.ipynb   # Phase 2.5 eval notebook
```

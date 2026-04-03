# CogMem Phase 1 Results — Episode Collection (v1)

## Overview

Phase 1 collects Q-valued episodes from MemRL's ALFWorld environment using a local LLM (llama3.2:3b via Ollama). The collected memories with Q-values serve as training data for Phase 2 (LoRA fine-tuning).

## Infrastructure

| Component | Details |
|-----------|---------|
| Compute | Paperspace A4000 (16GB VRAM) |
| LLM | llama3.2:3b (Q4_K_M quantization, ~2GB) |
| Embedding | nomic-embed-text (768 dimensions) |
| Framework | MemRL (MemTensor/MemRL) |
| Memory System | MemOS with Qdrant local vector DB |
| Environment | ALFWorld (text-based household tasks) |

### Environment Split (Local Development)

During development, a split Windows/WSL environment was used:
- **Windows**: Ollama (LLM + embeddings), ML stack (torch, transformers, peft)
- **WSL Ubuntu**: ALFWorld + TextWorld (Linux-only dependencies), MemRL runner
- **Radmin VPN**: Connected GTX 1650 and RTX 3050 laptops for distributed inference
- **Load Balancer**: Custom round-robin proxy across multiple Ollama instances

Production runs were moved to Paperspace for faster iteration.

## Configuration

```yaml
llm:
  model: llama3.2:3b
  temperature: 0
  max_tokens: 4096

experiment:
  random_seed: 42
  enable_value_driven: true
  num_sections: 1
  batch_size: 8
  dataset_ratio: 0.07  # ~248 games from 3553 total
  max_steps: 20

rl_config:
  tau: 0.62        # unknown detection threshold
  alpha: 0.3       # Q-learning step size
  gamma: 0.0       # discount factor (single-step)
  success_reward: 1.0
  failure_reward: -1.0
  topk: 3          # candidate set size for value-aware selection
  novelty_threshold: 0.85
  weight_sim: 0.5  # similarity weight in combined score
  weight_q: 0.5    # Q-value weight in combined score
```

## Memory Strategy

| Strategy | Value | Description |
|----------|-------|-------------|
| Build | proceduralization | Convert episode trajectories into procedural memories |
| Retrieve | query | Query-based memory retrieval using embedding similarity |
| Update | adjustment | Adjust Q-values based on episode outcomes |

## Results

### Data Collected

| Metric | Value |
|--------|-------|
| Total memories | 248 |
| Games attempted | ~248 (31 batches x 8 games) |
| Successful games | 9 (~3.6% success rate) |
| Failed games | ~239 |
| Q-value range | [-0.882, 0.000] |
| Q-value mean | -0.613 |
| Q visits (updates per memory) | 1-3+ |

### Memory Structure

Each memory contains:
- **id**: UUID
- **vector**: 768-dim embedding (nomic-embed-text)
- **payload**:
  - `memory`: Proceduralized task reflection text
  - `q_value`: Learned Q-value from RL updates
  - `q_visits`: Number of times Q-value was updated
  - `success`: Whether the episode succeeded (True/False)
  - `last_reward`: Final reward signal (-1.0 or 1.0)
  - `task_description`: ALFWorld task description
  - `full_content`: Complete reflection including task context
  - `confidence`: Memory confidence score
  - `strategy_build/retrieve/update`: Strategy labels
  - `related_memory_ids`: Links to related memories

### Q-Value Distribution

- Most memories have negative Q-values (mean: -0.613)
- This is expected for a 3b model on ALFWorld — the task is challenging
- Negative Q-values are valuable: they teach the model what NOT to do
- The mix of positive (success) and negative (failure) signals provides training diversity

### Success Rate Context

- **3b model zero-shot**: ~3.6% success rate on ALFWorld
- **8b model zero-shot**: ~100% success rate (tested separately)
- The low 3b success rate motivates CogMem: can we improve it through Q-value guided LoRA consolidation?

## Output Files

```
phase1_backup/notebooks/MemRL/results/alfworld/
  exp_cogmem_phase1_3b_20260403-113733/
    local_cache/
      snapshot/1/
        cube/
          textual_memory.json    # 22.8 MB — 248 memories with Q-values
          config.json            # Cube configuration
        qdrant/                  # Vector DB snapshot
        snapshot_meta.json       # Checkpoint metadata
      token_usage.jsonl          # LLM token usage logs
```

## Issues Encountered

1. **Ollama 128K context OOM**: Default context window (128K) caused out-of-memory on GTX 1650. Fixed by creating `llama3.2:3b-ctx2k` variant or setting Ollama context slider to 4K.
2. **WSL networking**: WSL's localhost != Windows localhost. Used gateway IP (172.31.192.1) for cross-environment access.
3. **Embedding dimension mismatch**: MemOS hardcoded Qdrant collection to 3072 dimensions (OpenAI default). nomic-embed-text produces 768. Fixed by patching `memory_service.py`.
4. **chonkie API incompatibility**: MemOS 1.0.0 used deprecated `tokenizer_or_token_counter` parameter. Fixed by pinning `chonkie==1.2.1`.
5. **Paperspace package conflicts**: Pre-installed torch/transformers versions conflicted with MemRL dependencies. Resolved by pinning compatible versions.
6. **No intermediate checkpoints**: MemRL only saves cube dumps at section boundaries, risking data loss on crashes. Mitigated by reducing section size.

## Next Steps

1. **Phase 2**: Run `convert_memrl.py` to transform cube dump into CogMem memory bank format
2. **Phase 3**: Q-value guided LoRA fine-tuning on llama3.2:3b using the 248 memories
3. **Evaluation**: Compare base 3b vs CogMem-3b on ALFWorld tasks
4. **Iteration**: Collect more episodes with the improved model (cycle 2)

## Timing

| Activity | Duration |
|----------|----------|
| Phase 0 (pre-flight) | ~2 hours (dependency setup) |
| Local attempts (1650 + 3050) | ~6 hours (OOM issues, networking) |
| Paperspace setup + debugging | ~3 hours |
| Successful data collection | ~4 hours (31 batches on A4000) |
| **Total Phase 1** | **~15 hours** |

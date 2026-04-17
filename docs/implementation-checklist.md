# CogMem Implementation Checklist

This checklist turns the recommendations in [deep-research-report.md](F:/Newgeneration/CogMem/docs/deep-research-report.md) into a concrete execution plan for the current repo.

## Phase 0: Stabilize the current baseline

- [ ] Freeze the current notebook baseline.
  Files: `paperspace_patches.ipynb`, `docs/powerpoint_notes_transfer_memory.md`, `docs/phase1_results_v1.md`
  Done when: one documented "patch-memory baseline" exists with exact train/eval split, retrieval settings, and expected metrics.

- [ ] Mark the patch path as a research path, not the final production memory path.
  Files: `docs/deep-research-report.md`, `docs/powerpoint_notes_transfer_memory.md`
  Done when: docs explicitly say cluster patches are a consolidation probe, not the end-state memory abstraction.

## Phase 1: Add split manifests and leakage guards

- [ ] Add `cogmem/benchmarks/bigcodebench/splits.py`.
  Implement:
  - deterministic split builder from task ids
  - manifest hashing
  - save/load helpers
  - `official_baseline` vs `bigcodebench_cl` labels

- [ ] Make dataset loading split-aware.
  Edit: `cogmem/benchmarks/bigcodebench/dataset.py`
  Add:
  - split label injection
  - library and family metadata passthrough where available

- [ ] Stamp every created episode with split lineage.
  Edit: `cogmem/benchmarks/bigcodebench/runner.py`
  Add fields:
  - `split_name`
  - `manifest_id`
  - `task_hash`
  - `source_benchmark`

- [ ] Block holdout contamination in training builders.
  Edit: `cogmem/consolidation/pipeline.py`, `cogmem/consolidation/select.py`, `cogmem/consolidation/abstract.py`
  Done when: no episode or training example is selectable unless its `manifest_id` is explicitly allowed.

## Phase 2: Replace overloaded `q_value`

- [ ] Add `cogmem/memory/schema.py`.
  Define explicit metrics:
  - `episode_helpfulness`
  - `card_transfer_gain`
  - `adapter_dev_gain`
  - `retrieval_confidence`
  - `negative_transfer_rate`

- [ ] Migrate episodic memory away from generic `q_value`.
  Edit: `cogmem/memory/memory_bank.py`, `cogmem/benchmarks/bigcodebench/runner.py`
  Done when:
  - episodic storage uses `episode_helpfulness`
  - any `q_value` field is compatibility-only

- [ ] Migrate consolidation code away from generic `q_value`.
  Edit:
  - `cogmem/consolidation/select.py`
  - `cogmem/consolidation/abstract.py`
  - `cogmem/consolidation/experience_summary.py`
  - `cogmem/consolidation/pipeline.py`
  Done when: selection and weighting use explicit episodic metrics.

- [ ] Finish the naming cleanup in the patch path.
  Edit:
  - `cogmem/patches/memory_bank.py`
  - `cogmem/patches/bank.py`
  - `cogmem/patches/patch.py`
  - `cogmem/patches/compose.py`
  Done when:
  - `promotion_score` is the promotion metric
  - `Q_use` / `final_use` are query-time metrics
  - `q_value` remains only as legacy serialization if still required

## Phase 3: Build a typed episodic store

- [ ] Add `cogmem/memory/episodic_store.py`.
  It should own:
  - append/read/update APIs
  - metadata filtering by split, family, library, and outcome
  - lineage checks

- [ ] Route raw episode writes through the store API.
  Edit callers:
  - `collect_sequential.py`
  - `cogmem/benchmarks/bigcodebench/runner.py`
  - `cogmem/patches/wake.py`
  Done when: new code paths stop writing ad hoc episode dicts directly.

- [ ] Add richer episode features.
  Record:
  - prompt hash
  - retrieved ids
  - adapter ids
  - error family
  - AST fingerprint if available
  - validation recipe
  Done when: clustering can use bug/fix features, not just prompt family.

## Phase 4: Add first-class procedural memory

- [ ] Add `cogmem/memory/skill_store.py`.
  Required fields:
  - `skill_id`
  - `triggers`
  - `plan_steps`
  - `validation`
  - `anti_patterns`
  - `evidence_episode_ids`
  - `transfer_gain`
  - `confidence`
  - `manifest_ids`

- [ ] Add `cogmem/consolidation/proceduralize.py`.
  Inputs:
  - clustered episodes
  - fail/pass contrasts
  - error families
  Outputs:
  - candidate skill cards

- [ ] Keep `experience_summary.py` only as an intermediate helper.
  Edit: `cogmem/consolidation/experience_summary.py`
  Done when:
  - summaries are debugging aids
  - skill cards are the actual procedural memory object

- [ ] Add skill-card validation.
  Add a `validate_skill_card(card_id, dev_task_ids)` entry point.
  Done when: cards are promoted only after held-out or out-of-cluster gain is measured.

## Phase 5: Narrow the patch system to a research probe

- [ ] Keep cluster patches behind a clear boundary.
  Edit:
  - `cogmem/patches/memory_bank.py`
  - `cogmem/patches/wake.py`
  - `cogmem/patches/cycle.py`
  Goal:
  - patches remain one experimental consolidation route
  - they do not become the only production memory object

- [ ] Make top-1 retrieval the default.
  Edit runtime defaults and notebook defaults.
  Done when: wider retrieval is an explicit ablation only.

- [ ] Add a research-only note around old patch-bank sleep.
  Edit: `cogmem/patches/sleep.py`
  Goal: avoid confusion between old patch sleep and future skill-card consolidation.

## Phase 6: Add routed parametric memory

- [ ] Add `cogmem/consolidation/adapter_registry.py`.
  Track:
  - adapter id
  - training manifest
  - source skill cards
  - dev gain
  - compatible families
  - rollback status

- [ ] Make adapter training manifest-aware.
  Edit:
  - `cogmem/consolidation/train_lora_local.py`
  - `cogmem/consolidation/train_generator.py`
  Done when: every adapter artifact traces back to skill-card ids and split manifests.

- [ ] Route adapters at inference instead of merging into the base.
  Edit or add around: `cogmem/consolidation/router.py`
  Goal:
  - frozen base model
  - optional global coding adapter
  - optional one family adapter
  - routing decision recorded in episode metadata

## Phase 7: Build a proper evaluation harness

- [ ] Add `cogmem/benchmarks/bigcodebench/cl_eval.py`.
  Include:
  - split-aware labels
  - pass@1 and pass@3
  - negative transfer rate
  - retrieval helpfulness A/B sampling
  - consolidation gain

- [ ] Separate official baseline reporting from continual-learning reporting.
  Done when outputs are always labeled either:
  - `Official BigCodeBench baseline`
  - `BigCodeBench-CL`

- [ ] Add an ablation runner.
  Required settings:
  - base
  - episodic-sim
  - episodic-utility
  - procedural
  - param-global
  - param-family

## Phase 8: Retire notebook-only logic into package code

- [ ] Port stable parts of `paperspace_patches.ipynb` into modules.
  Candidate homes:
  - retrieval and scoring helpers -> `cogmem/patches/memory_bank.py` or a new retrieval module
  - evaluation loops -> `cogmem/benchmarks/bigcodebench/cl_eval.py`
  - resume/checkpoint helpers -> dedicated experiment runner module

- [ ] Keep the notebook only as an experiment UI.
  Done when: the notebook mostly imports package functions instead of holding source-of-truth logic.

## Phase 9: Testing and CI

- [ ] Add unit tests for the new stores and schemas.
  Add:
  - `tests/test_episodic_store.py`
  - `tests/test_skill_store.py`
  - `tests/test_split_manifest.py`

- [ ] Add leakage tests.
  Goal:
  - fail if holdout task ids enter training manifests
  - fail if a run mixes incompatible split manifests

- [ ] Extend retrieval smoke tests.
  Extend: `tests/test_cluster_memory_bank.py`
  Verify:
  - top-1 default
  - abstention still works
  - harmful memories demote correctly

- [ ] Add CI tiers.
  CPU CI:
  - schema validation
  - unit tests
  - split and leakage tests
  GPU nightly:
  - tiny adapter training smoke test
  - tiny retrieval/eval smoke test

## Suggested execution order

1. Ship split manifests and leakage guards.
2. Rename and separate utility semantics.
3. Add the typed episodic store.
4. Add skill-card schema and proceduralizer.
5. Keep patches as a research-only consolidation path.
6. Add adapter registry and routed adapter loading.
7. Build the BigCodeBench-CL evaluation harness.
8. Retire notebook-only logic into package modules.
9. Add CI and leakage tests.

## Migration done criteria

- [ ] No holdout task enters memory or training without an explicit manifest match.
- [ ] No production path relies on generic `q_value` semantics.
- [ ] Skill cards exist as a first-class saved object in the package.
- [ ] Adapter training consumes promoted skill cards, not raw episodes.
- [ ] The notebook is no longer the source of truth for retrieval or evaluation logic.
- [ ] Continual-learning results are labeled `BigCodeBench-CL`.

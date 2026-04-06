# CogMem v1 Baseline — Qwen2.5:3b on BigCodeBench Full (1140 tasks)

**Date:** 2026-04-05
**Model:** qwen2.5:3b (Qwen/Qwen2.5-3B-Instruct)
**Dataset:** BigCodeBench Full v0.1.4 (1140 tasks)
**Settings:** max_tokens=2048, temperature=0

## Results

| Metric | Value |
|---|---|
| **Total tasks** | 1140 |
| **Passed** | 254 (22.3%) |
| **Failed** | 886 (77.7%) |

## Q-Value Distribution (Cycle 1 — binary)

| Zone | Count | Percentage |
|---|---|---|
| High (Q > 0.5) | 254 | 22.3% |
| Mid (-0.5 to 0.5) | 0 | 0.0% |
| Low (Q < -0.5) | 886 | 77.7% |

Note: Q-values are binary (1.0/−1.0) in Cycle 1. Real Q-learning starts Cycle 2.

## Error Breakdown

| Error Type | Count | % of Failures |
|---|---|---|
| TestFailure | 346 | 39.1% |
| Other | 197 | 22.2% |
| TypeError | 125 | 14.1% |
| ImportError | 93 | 10.5% |
| AttributeError | 78 | 8.8% |
| NameError | 29 | 3.3% |
| SyntaxError | 11 | 1.2% |
| Timeout | 7 | 0.8% |

## Key Observations

- 39% of failures are TestFailure (code runs but produces wrong output)
  → Model understands the task but gets logic wrong. DPO can help here.
- 14% TypeError + 9% AttributeError = 23% wrong API usage
  → Model calls wrong methods. SFT on correct examples fixes this.
- 10.5% ImportError = missing/wrong package imports
  → Model hallucinates package names. DPO contrast learning helps.
- Only 1.2% SyntaxError = model generates valid Python almost always.
- Only 0.8% Timeout = model rarely writes infinite loops.

## Baseline for Paper

This is the **v1 baseline** (before any CogMem training).
Qwen2.5:3b (general, not code-specialized): **22.3% pass@1** on BigCodeBench Full.

The paper will show improvement over this baseline across Q-STaR cycles.

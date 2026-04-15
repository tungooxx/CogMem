# Transfer Memory Notes

These are the points to carry into the PowerPoint when explaining why the
episode-first redesign matters and what changed in retrieval.

## One-Line Message

Transfer improves only when memory stops being a pile of single-task patches and
becomes an evidence-backed repair object with applicability boundaries, negative
support, and transfer-aware Q.

## Old vs New

### Old

- Patch-first memory.
- Immediate promotion from a single task into the active bank.
- Retrieval was dominated by nearest prompt embedding.
- Q mostly sounded like generic usefulness.
- Little or no explicit negative support.
- No real abstention boundary per memory.
- Easy for broad memories to help seen tasks but hurt unseen tasks.

### New

- Episode-first memory.
- Save episode -> cluster episodes -> distill candidate patch -> validate -> then retrieve.
- Retrievable object is no longer just a patch id plus embedding.
- Retrievable memory object now groups:
  - `cluster_metadata`
  - `evidence`
  - `transfer_stats`
  - `patch_ids`
- Retrieval now separates:
  - durable promotion trust
  - query-time applicability and use trust
- Every memory has an abstention boundary through `retrieval_threshold`.
- Promotion score is transfer-aware, so unseen hurt can lower trust and eventually disable retrieval.

## What Changed

### Patch creation

- Do not immediately add a task-created patch to the active bank.
- Save an episode first.
- Cluster related episodes.
- Build a candidate patch from the cluster.
- Validate it on held-out examples.
- Only then attach it as a provisional retrievable artifact.

### Retrieval

Retrieval should not be "nearest prompt embedding wins".

Wake now uses a three-step policy:

`applicability(m, q) = w1 * sim(q, pos_prototype_m) - w2 * sim(q, neg_prototype_m) + w3 * structural_match(q, m)`

`Q_use(m, q) = applicability(m, q) * (u1 * transfer_gain + u2 * recent_success + u3 * reuse - u4 * online_hurt)`

`FinalUse(m, q) = Q_use(m, q) + 0.15 * Q_promote(m)`

Wake behavior:

- compute applicability
- drop memories below their applicability threshold
- rank the remaining memories by `FinalUse`
- use top-1 unless confidence is very high

### Current code formula

The current implementation uses:

```text
applicability(m, q) = clip(
    0.60 * sim(q, pos_prototype_m)
  - 0.25 * sim(q, neg_prototype_m)
  + 0.15 * structural_match(q, m),
  0, 1
)

Q_use(m, q) = applicability(m, q) * clip(
    0.45 * transfer_gain
  + 0.20 * recent_success_rate
  + 0.15 * log_reuse
  - 0.20 * online_hurt_rate,
  0, 1
)

FinalUse(m, q) = clip(
    Q_use(m, q) + 0.15 * Q_promote(m),
    0, 1
)
```

Where:

- `pos_prototype_m` is the memory centroid / positive prototype
- `neg_prototype_m` is a small hard-negative pool near the cluster boundary
- `structural_match(q, m)` mixes prompt marker overlap with family match
- `log_reuse` is `log(1 + reuse_count)` normalized by the largest reuse count in the bank

The current gate is on applicability:

- retrieve only if `applicability(m, q) > retrieval_threshold_m`
- `retrieval_threshold_m` is set from the midpoint between:
  - the 25th percentile of positive-example applicability scores
  - the 85th percentile of negative-pool applicability scores

The current composition rule is:

- return only the best memory by default
- only return more than one memory when:
  - the top memory has very high `FinalUse`
  - and the next memories stay within a small margin of the top memory

The current code now models this with:

- `positive_prototype`: where the memory tends to work
- `negative_prototype`: hard negatives near the cluster boundary
- `structural_markers`: lightweight repair/applicability cues from episode prompts
- `retrieval_threshold`: per-memory abstention gate
- `promotion_score`: durable trust for promotion / demotion decisions
- `recent_success_rate`: online recency-weighted usefulness signal
- `online_hurt_rate`: online recency-weighted harm signal
- `transfer_online_gain`: online success ratio from actual memory uses
- retrievable memory payload grouped as:
  - `cluster_metadata`
  - `evidence`
  - `transfer_stats`
  - `patch_ids`

### Q-value

One scalar should not do every job.

The system now separates:

- `Q_promote(m)`: should this memory stay trusted, merge, or be demoted?
- `Q_use(m, q)`: should this memory fire on this query right now?

The split now maps cleanly onto wake vs sleep:

- wake decides activation with `applicability`, `Q_use`, and `FinalUse`
- sleep decides promotion, demotion, and pruning with `Q_promote`

## Why This Improves Transfer

Transfer fails when memory stores only:

- what worked once

Transfer improves when memory stores:

- what worked
- why it worked
- where it worked
- where it failed
- when to abstain

That is the difference between recall and skill.

## First Four Changes To Emphasize

1. Store negatives for each memory, not just positives.
2. Store cluster-level support, not only source task id.
3. Add an abstention threshold to every memory.
4. Make Q punish unseen hurt, not just reward seen gain.

## Current Transfer-Aware Q Shape

The intended durable utility is no longer "generic usefulness". It is a
promotion score built from transfer evidence, support, and harm:

`Q_promote(m) = a * heldout_gain + b * transfer_gain + c * local_support_gain + d * support - e * harm - f * redundancy`

### Current code formula

The current implementation uses:

```text
Q_promote(m) = 0.28 * held_out_steering_gain
             + 0.16 * transfer_gain
             + 0.06 * transfer_online_gain
             + 0.12 * local_support_gain
             + 0.10 * distillation_success
             + 0.08 * log_support
             + 0.10 * recent_success_rate
             - 0.10 * online_hurt_rate
             - 0.15 * utility_regression
             - 0.15 * unseen_hurt_rate
             - 0.10 * redundancy_penalty
```

Where:

- `log_support` is `log(1 + support_count)` normalized by the largest support count in the bank
- `transfer_gain` remains the held-out distillation-time signal from cluster validation
- `transfer_online_gain` is the online success ratio from seen/unseen help-hurt counters
- legacy `q_value` metadata now mirrors `promotion_score` for compatibility with saved patch artifacts

And a memory is demoted from retrieval if:

- it has no distilled patch ids, or distillation failed
- or `promotion_score < 0.40`
- or `support_count < 3`
- or it is harming the preserve set:
  - `online_hurt_rate >= 0.60`
  - or `utility_regression >= 0.35`
  - or `unseen_hurt_count >= 2` and `unseen_hurt_count > unseen_help_count`

And a memory may be pruned during sleep if:

- `promotion_score <= 0.10`
- `support_count < 3`
- and it is already harming the preserve set

Operationally this means a memory should score highly only if it:

- helps on support tasks
- helps on held-out similar tasks
- carries enough repeated support to be worth keeping
- keeps helping in actual online use
- does not hurt confusing nearby tasks
- is not redundant with other memories

And if a memory helps seen tasks but repeatedly hurts unseen tasks, it should be
demoted or stop being retrievable.

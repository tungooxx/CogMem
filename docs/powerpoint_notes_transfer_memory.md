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
- Retrieval uses a gated score:
  - positive prototype similarity
  - minus negative prototype similarity
  - plus transfer-aware Q
  - plus structural match
- Every memory has an abstention boundary through `retrieval_threshold`.
- Q is now transfer-aware, so unseen hurt can lower Q and eventually disable retrieval.

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

Use a gated score:

`RetrieveScore(m, q) = w1 * sim(q, pos_prototype_m) - w2 * sim(q, neg_prototype_m) + w3 * Q_m + w4 * structural_match(q, m)`

Only retrieve when:

`RetrieveScore(m, q) > tau_m`

### Current code formula

The current implementation uses:

`RetrieveScore(m, q) = 0.45 * sim(q, pos_prototype_m) - 0.20 * sim(q, neg_prototype_m) + 0.25 * Q_m + 0.10 * structural_match(q, m)`

Where:

- `pos_prototype_m` is the memory centroid / positive prototype
- `neg_prototype_m` is a small hard-negative pool near the cluster boundary
- `structural_match(q, m)` mixes prompt marker overlap with family match

The current gate is:

- retrieve only if `RetrieveScore(m, q) > retrieval_threshold_m`
- `retrieval_threshold_m` is set from the midpoint between:
  - the 25th percentile of positive example scores
  - the 85th percentile of negative-pool scores

The current code now models this with:

- `positive_prototype`: where the memory tends to work
- `negative_prototype`: hard negatives near the cluster boundary
- `structural_markers`: lightweight repair/applicability cues from episode prompts
- `retrieval_threshold`: per-memory abstention gate
- retrievable memory payload grouped as:
  - `cluster_metadata`
  - `evidence`
  - `transfer_stats`
  - `patch_ids`

### Q-value

Q should not only reward held-out or seen-side fit.

Q should also punish transfer failures, especially:

- unseen-task hurt
- repeated misuse
- redundant broad memories

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

The intended utility is no longer "generic usefulness". It is transfer-aware
utility:

`Q(m) = a * local_gain + b * heldout_gain + c * transfer_gain - d * unseen_hurt - e * redundancy`

### Current code formula

The current implementation uses:

`Q(m) = 0.22 * local_support_gain + 0.23 * held_out_steering_gain + 0.20 * transfer_gain`

`      + 0.05 * distillation_success + 0.05 * recency_score + 0.05 * seen_help_rate + 0.05 * normalized_reuse`

`      - 0.10 * utility_regression - 0.10 * negative_steering_penalty - 0.10 * redundancy_penalty - 0.15 * unseen_hurt_rate`

And a memory is demoted from retrieval if:

- it has no distilled patch ids, or distillation failed
- or `unseen_hurt_count >= 2` and `unseen_hurt_count > unseen_help_count`

Operationally this means a memory should score highly only if it:

- helps on support tasks
- helps on held-out similar tasks
- does not hurt confusing nearby tasks
- is not redundant with other memories

And if a memory helps seen tasks but repeatedly hurts unseen tasks, it should be
demoted or stop being retrievable.

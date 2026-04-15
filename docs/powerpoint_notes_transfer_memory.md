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

Operationally this means a memory should score highly only if it:

- helps on support tasks
- helps on held-out similar tasks
- does not hurt confusing nearby tasks
- is not redundant with other memories

And if a memory helps seen tasks but repeatedly hurts unseen tasks, it should be
demoted or stop being retrievable.

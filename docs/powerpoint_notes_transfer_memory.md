# Transfer Memory Notes

These are the points to carry into the PowerPoint when explaining why the
episode-first redesign matters and what changed in retrieval.

## One-Line Message

Transfer improves only when memory stops being a pile of single-task patches and
becomes an evidence-backed repair object with applicability boundaries, negative
support, and transfer-aware Q.

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

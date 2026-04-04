"""CogMem consolidation — Q-value triage sleep pipeline.

Core pipeline (new):
    triage -> sft_train + grpo_train -> merge -> verify

Legacy pipeline:
    select -> abstract -> train_lora -> verify
"""

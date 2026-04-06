"""CogMem consolidation — Q-STaR sleep pipeline.

Core pipeline:
    select -> abstract (SFT + preference pairs) -> train_generator (SFT+DPO)
    -> train_verifier (DPO) -> verify

Legacy pipeline:
    select -> abstract -> train_lora -> verify
"""

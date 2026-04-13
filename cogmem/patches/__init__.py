"""CogMem Cognitive Patches — micro-LoRA adapters from experience.

Each patch captures one lesson from a coding experience.
Patches compose dynamically per-task, so the model thinks
differently based on accumulated experience.
"""

from cogmem.patches.bank import PatchBank
from cogmem.patches.memory_bank import ClusterMemoryBank

__all__ = ["PatchBank", "ClusterMemoryBank"]

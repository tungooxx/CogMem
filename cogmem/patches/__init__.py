"""Experimental CogMem cognitive patches.

The patch modules remain a research-only compression path. They are useful for
ablation and consolidation experiments, but the production memory ladder is
episodes -> skill cards -> optional routed adapters.
"""

from cogmem.patches.memory_bank import ClusterMemoryBank
from cogmem.patches.bank import PatchBank

__all__ = ["ClusterMemoryBank", "PatchBank"]

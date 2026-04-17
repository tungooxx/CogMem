"""Experimental CogMem cognitive patches.

The patch modules remain a research-only compression path. They are useful for
ablation and consolidation experiments, but the production memory ladder is
episodes -> skill cards -> optional routed adapters.
"""

__all__ = ["ClusterMemoryBank", "PatchBank"]


def __getattr__(name: str):
    if name == "ClusterMemoryBank":
        from cogmem.patches.memory_bank import ClusterMemoryBank

        return ClusterMemoryBank
    if name == "PatchBank":
        from cogmem.patches.bank import PatchBank

        return PatchBank
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

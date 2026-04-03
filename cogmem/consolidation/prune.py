def tag_consolidated(
    episodes: list[dict], consolidated_ids: set[str], domain: str = ""
) -> list[dict]:
    """Tag episodes that were included in consolidation instead of deleting them."""
    for ep in episodes:
        if ep["episode_id"] in consolidated_ids:
            ep["consolidated"] = True
            ep["consolidated_domain"] = domain
    return episodes

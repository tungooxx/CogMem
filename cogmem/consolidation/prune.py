def prune_consolidated(
    episodes: list[dict], consolidated_ids: set[str]
) -> list[dict]:
    return [ep for ep in episodes if ep["episode_id"] not in consolidated_ids]

from __future__ import annotations


def should_stop(
    pairs_played: int,
    games_played: int,
    min_blocks: int,
    max_games: int,
    ci_halfwidth_elo: float,
    target_halfwidth_elo: float,
) -> tuple[bool, str]:
    if games_played >= max_games:
        return True, "max_games_reached"
    if pairs_played < min_blocks:
        return False, "min_blocks_not_reached"
    if ci_halfwidth_elo <= target_halfwidth_elo:
        return True, "ci_halfwidth_reached"
    return False, "continue"

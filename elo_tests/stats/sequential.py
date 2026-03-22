"""Sequential stopping criteria for Elo testing.

This module provides functions for determining when to stop an Elo
estimation run based on various criteria.
"""

from __future__ import annotations


def should_stop(
    pairs_played: int,
    games_played: int,
    min_blocks: int,
    max_games: int,
    ci_halfwidth_elo: float,
    target_halfwidth_elo: float,
) -> tuple[bool, str]:
    """Determine whether to stop the Elo estimation run.

    Args:
        pairs_played: Number of paired blocks completed.
        games_played: Total number of games played.
        min_blocks: Minimum blocks required before early stop.
        max_games: Maximum games allowed.
        ci_halfwidth_elo: Current confidence interval half-width.
        target_halfwidth_elo: Target half-width for stopping.

    Returns:
        Tuple of (should_stop, reason).

    """
    if games_played >= max_games:
        return True, "max_games_reached"
    if pairs_played < min_blocks:
        return False, "min_blocks_not_reached"
    if ci_halfwidth_elo <= target_halfwidth_elo:
        return True, "ci_halfwidth_reached"
    return False, "continue"

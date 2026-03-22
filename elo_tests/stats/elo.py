"""Elo calculation and confidence interval functions.

This module provides statistical functions for computing Elo ratings
from game scores and calculating confidence intervals.
"""

from __future__ import annotations

import math
from statistics import NormalDist


def mean_score(scores: list[float]) -> float:
    """Calculate the mean score from a list of game scores.

    Args:
        scores: List of scores (0.0, 0.5, or 1.0 per game).

    Returns:
        The mean score, or 0.5 if the list is empty.

    """
    return sum(scores) / len(scores) if scores else 0.5


def score_rate_to_elo(p: float) -> float:
    """Convert a score rate to an Elo difference.

    Args:
        p: Score rate (0.0 to 1.0).

    Returns:
        Elo difference corresponding to the score rate.

    """
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10((1.0 / p) - 1.0)


def elo_to_score_rate(elo: float) -> float:
    """Convert an Elo difference to a score rate.

    Args:
        elo: Elo difference.

    Returns:
        Expected score rate (0.0 to 1.0).

    """
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def normal_ci_for_elo(scores: list[float], level: float) -> tuple[float, float]:
    """Calculate a normal-based confidence interval for Elo.

    Args:
        scores: List of game scores.
        level: Confidence level (e.g., 0.95 for 95%).

    Returns:
        Tuple of (lower_bound, upper_bound) for the Elo estimate.

    """
    if not scores:
        return 0.0, 0.0
    if len(scores) == 1:
        e = score_rate_to_elo(scores[0])
        return e, e

    n = len(scores)
    p_hat = mean_score(scores)
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    variance = max(p_hat * (1.0 - p_hat), 1e-12)
    half = z * math.sqrt(variance / n)
    p_low = min(max(p_hat - half, 1e-6), 1 - 1e-6)
    p_high = min(max(p_hat + half, 1e-6), 1 - 1e-6)
    return score_rate_to_elo(p_low), score_rate_to_elo(p_high)

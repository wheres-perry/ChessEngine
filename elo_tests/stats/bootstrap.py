"""Bootstrap confidence interval calculation for paired data.

This module provides paired bootstrap resampling for computing
confidence intervals on Elo estimates.
"""

from __future__ import annotations

import random

from elo_tests.stats.elo import score_rate_to_elo


def paired_bootstrap_elo_ci(
    pair_scores: list[float],
    level: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Calculate a bootstrap confidence interval for Elo from paired scores.

    Args:
        pair_scores: Mean scores for each paired block.
        level: Confidence level (e.g., 0.95 for 95%).
        n_resamples: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound) for the Elo estimate.

    """
    if not pair_scores:
        return 0.0, 0.0

    rng = random.Random(seed)
    n = len(pair_scores)
    estimates: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += pair_scores[rng.randrange(0, n)]
        estimates.append(score_rate_to_elo(total / n))

    estimates.sort()
    alpha = 1.0 - level
    low_idx = int((alpha / 2.0) * (n_resamples - 1))
    high_idx = int((1.0 - alpha / 2.0) * (n_resamples - 1))
    return estimates[low_idx], estimates[high_idx]

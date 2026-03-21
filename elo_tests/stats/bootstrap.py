from __future__ import annotations

import random

from elo_tests.stats.elo import score_rate_to_elo


def paired_bootstrap_elo_ci(
    pair_scores: list[float],
    level: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
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

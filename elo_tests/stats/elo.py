from __future__ import annotations

import math
from statistics import NormalDist


def mean_score(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.5


def score_rate_to_elo(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10((1.0 / p) - 1.0)


def elo_to_score_rate(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def normal_ci_for_elo(scores: list[float], level: float) -> tuple[float, float]:
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

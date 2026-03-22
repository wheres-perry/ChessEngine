"""Statistical functions for Elo estimation.

This package provides statistical utilities for computing Elo ratings,
confidence intervals, and sequential stopping criteria.
"""

from elo_tests.stats.bootstrap import paired_bootstrap_elo_ci
from elo_tests.stats.elo import mean_score, normal_ci_for_elo, score_rate_to_elo
from elo_tests.stats.sequential import should_stop

__all__ = [
    "mean_score",
    "normal_ci_for_elo",
    "paired_bootstrap_elo_ci",
    "score_rate_to_elo",
    "should_stop",
]

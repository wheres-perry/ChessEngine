"""Data models for Elo estimation runs.

This module defines the core data structures used throughout the Elo testing
package, including engine specifications, run configuration, game records,
and result summaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

CI_METHOD = Literal["bootstrap", "normal"]
"""Type alias for supported confidence interval calculation methods."""


@dataclass(frozen=True)
class EngineSpec:
    """Specification for a chess engine under test.

    Attributes:
        engine_id: Unique identifier for the engine.
        kind: Type of engine (currently only "simulated" is supported).
        strength_elo: Engine strength in Elo points.
        draw_bias: Additional draw probability bias for this engine.
        version: Version string for the engine.

    """

    engine_id: str
    kind: Literal["simulated"] = "simulated"
    strength_elo: float = 0.0
    draw_bias: float = 0.0
    version: str = "0.1"


@dataclass(frozen=True)
class TimeControlSpec:
    """Time control settings for game play.

    Attributes:
        movetime_ms: Time per move in milliseconds, or None for no limit.
        depth: Maximum search depth, or None for depth-unlimited search.

    """

    movetime_ms: int | None = 50
    depth: int | None = None


@dataclass(frozen=True)
class RunConfig:
    """Complete configuration for an Elo estimation run.

    Attributes:
        candidate: The engine being evaluated.
        baseline: The reference engine for comparison.
        time_control: Time control settings for games.
        openings_file: Path to openings file, or None for defaults.
        paired_blocks_target: Target number of paired game blocks.
        min_blocks: Minimum number of blocks before allowing early stop.
        max_games: Maximum number of games to play.
        ci_method: Method for confidence interval calculation.
        ci_level: Confidence level (e.g., 0.95 for 95%).
        stop_ci_halfwidth_elo: Stop when CI half-width is at most this value.
        recompute_every_pairs: Recompute statistics every N pairs.
        workers: Number of parallel workers (currently unused).
        seed: Random seed for reproducibility.
        output_dir: Directory for output files.
        bootstrap_resamples: Number of bootstrap resamples for CI.

    """

    candidate: EngineSpec
    baseline: EngineSpec
    time_control: TimeControlSpec
    openings_file: str | None
    paired_blocks_target: int
    min_blocks: int
    max_games: int
    ci_method: CI_METHOD
    ci_level: float
    stop_ci_halfwidth_elo: float
    recompute_every_pairs: int
    workers: int
    seed: int
    output_dir: str
    bootstrap_resamples: int


@dataclass(frozen=True)
class ScheduledGame:
    """A single scheduled game in a paired match.

    Attributes:
        pair_id: Identifier for the paired block this game belongs to.
        opening_id: Index of the opening position used.
        opening_fen: FEN string of the starting position.
        game_index_in_pair: 0 for first game, 1 for second (reversed colors).
        seed: Random seed for this game.
        candidate_is_white: True if candidate plays white in this game.

    """

    pair_id: int
    opening_id: int
    opening_fen: str
    game_index_in_pair: int
    seed: int
    candidate_is_white: bool


@dataclass(frozen=True)
class GameRecord:
    """Record of a completed game.

    Attributes:
        run_id: Identifier for the run this game belongs to.
        pair_id: Identifier for the paired block.
        opening_id: Index of the opening position used.
        seed: Random seed used for this game.
        candidate_side: Side the candidate played ("white" or "black").
        candidate_score: Score from candidate's perspective (1.0, 0.5, or 0.0).
        ply_count: Number of half-moves played.
        termination: Game result description ("win", "loss", or "draw").
        candidate_time_ms: Time used by candidate in milliseconds.
        baseline_time_ms: Time used by baseline in milliseconds.

    """

    run_id: str
    pair_id: int
    opening_id: int
    seed: int
    candidate_side: Literal["white", "black"]
    candidate_score: float
    ply_count: int
    termination: str
    candidate_time_ms: int
    baseline_time_ms: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class EloSummary:
    """Summary of an Elo estimation run.

    Attributes:
        run_id: Identifier for this run.
        seed: Random seed used.
        candidate_id: Identifier for the candidate engine.
        baseline_id: Identifier for the baseline engine.
        games_played: Total number of games played.
        pairs_played: Total number of paired blocks completed.
        mean_score: Mean score for the candidate (0.0 to 1.0).
        elo_point: Point estimate of Elo difference.
        ci_level: Confidence level used for intervals.
        ci_primary_low: Lower bound of primary confidence interval.
        ci_primary_high: Upper bound of primary confidence interval.
        ci_normal_low: Lower bound of normal-based confidence interval.
        ci_normal_high: Upper bound of normal-based confidence interval.
        ci_bootstrap_low: Lower bound of bootstrap confidence interval.
        ci_bootstrap_high: Upper bound of bootstrap confidence interval.
        ci_method: Method used for primary confidence interval.
        stopped_early: Whether the run stopped before max games.
        stop_reason: Reason for stopping.
        results_path: Path to the detailed results file.

    """

    run_id: str
    seed: int
    candidate_id: str
    baseline_id: str
    games_played: int
    pairs_played: int
    mean_score: float
    elo_point: float
    ci_level: float
    ci_primary_low: float
    ci_primary_high: float
    ci_normal_low: float
    ci_normal_high: float
    ci_bootstrap_low: float
    ci_bootstrap_high: float
    ci_method: CI_METHOD
    stopped_early: bool
    stop_reason: str
    results_path: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary to a dictionary.

        Returns:
            Dictionary representation of this summary.

        """
        payload = asdict(self)
        payload["results_path"] = str(Path(self.results_path))
        return payload

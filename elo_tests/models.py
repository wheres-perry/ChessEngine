from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

CI_METHOD = Literal["bootstrap", "normal"]


@dataclass(frozen=True)
class EngineSpec:
    engine_id: str
    kind: Literal["simulated"] = "simulated"
    strength_elo: float = 0.0
    draw_bias: float = 0.0
    version: str = "0.1"


@dataclass(frozen=True)
class TimeControlSpec:
    movetime_ms: int | None = 50
    depth: int | None = None


@dataclass(frozen=True)
class RunConfig:
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
    pair_id: int
    opening_id: int
    opening_fen: str
    game_index_in_pair: int
    seed: int
    candidate_is_white: bool


@dataclass(frozen=True)
class GameRecord:
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
        return asdict(self)


@dataclass(frozen=True)
class EloSummary:
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
        payload = asdict(self)
        payload["results_path"] = str(Path(self.results_path))
        return payload

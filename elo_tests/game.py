from __future__ import annotations

import math
import random

from elo_tests.engines.base import EngineAdapter
from elo_tests.models import GameRecord, ScheduledGame, TimeControlSpec


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score_probability(delta_elo: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, -delta_elo / 400.0))


def play_game(
    run_id: str,
    candidate: EngineAdapter,
    baseline: EngineAdapter,
    scheduled: ScheduledGame,
    tc: TimeControlSpec,
) -> GameRecord:
    rng = random.Random(scheduled.seed)
    candidate_side = "white" if scheduled.candidate_is_white else "black"

    candidate.new_game(scheduled.seed, scheduled.opening_fen, candidate_side)
    baseline.new_game(
        scheduled.seed,
        scheduled.opening_fen,
        "black" if candidate_side == "white" else "white",
    )

    color_bonus = 18.0 if candidate_side == "white" else -18.0
    opening_noise = rng.uniform(-12.0, 12.0)
    total_delta = (
        candidate.strength_elo
        - baseline.strength_elo
        + color_bonus
        + opening_noise
    )

    expected = _score_probability(total_delta)
    draw_prob = _clamp(0.30 + math.exp(-abs(total_delta) / 200.0) * 0.22, 0.12, 0.60)

    roll = rng.random()
    if roll < draw_prob:
        candidate_score = 0.5
        termination = "draw"
    else:
        decisive_roll = rng.random()
        win_prob = _clamp((expected - draw_prob / 2.0) / (1.0 - draw_prob), 0.0, 1.0)
        candidate_score = 1.0 if decisive_roll < win_prob else 0.0
        termination = "win" if candidate_score == 1.0 else "loss"

    movetime = tc.movetime_ms or 30
    candidate_time = int(movetime + rng.randint(0, movetime))
    baseline_time = int(movetime + rng.randint(0, movetime))
    ply_count = int(30 + rng.randint(0, 80))

    return GameRecord(
        run_id=run_id,
        pair_id=scheduled.pair_id,
        opening_id=scheduled.opening_id,
        seed=scheduled.seed,
        candidate_side=candidate_side,
        candidate_score=candidate_score,
        ply_count=ply_count,
        termination=termination,
        candidate_time_ms=candidate_time,
        baseline_time_ms=baseline_time,
    )


def game_record_to_json(record: GameRecord) -> dict[str, object]:
    return record.to_dict()

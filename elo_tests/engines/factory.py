from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from elo_tests.models import EngineSpec


@dataclass
class SimulatedEngineAdapter:
    engine_id: str
    version: str
    strength_elo: float
    draw_bias: float
    _rng: random.Random

    def new_game(self, seed: int, opening_fen: str, side: str) -> None:
        self._rng.seed(seed + hash((opening_fen, side, self.engine_id)) % 1_000_003)

    def choose_move(
        self,
        state: Any,
        move_time_ms: int | None,
        depth: int | None,
    ) -> str:
        _ = (state, move_time_ms, depth)
        return "0000"


def create_engine(spec: EngineSpec, run_seed: int) -> SimulatedEngineAdapter:
    return SimulatedEngineAdapter(
        engine_id=spec.engine_id,
        version=spec.version,
        strength_elo=spec.strength_elo,
        draw_bias=spec.draw_bias,
        _rng=random.Random(run_seed + hash(spec.engine_id) % 10_007),
    )

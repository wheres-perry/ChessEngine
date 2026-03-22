"""Factory for creating engine adapters.

This module provides factory functions for creating engine instances
based on engine specifications.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elo_tests.models import EngineSpec


@dataclass
class SimulatedEngineAdapter:
    """Simulated engine adapter for testing.

    This adapter simulates engine behavior based on configured strength
    and draw bias parameters. It does not perform actual chess calculations
    but provides deterministic behavior for statistical testing.

    Attributes:
        engine_id: Unique identifier for this engine.
        version: Version string.
        strength_elo: Engine strength in Elo points.
        draw_bias: Draw probability bias.
        _rng: Internal random number generator.

    """

    engine_id: str
    version: str
    strength_elo: float
    draw_bias: float
    _rng: random.Random

    def new_game(self, seed: int, opening_fen: str, side: str) -> None:
        """Initialize a new game with the given seed.

        Args:
            seed: Base random seed for the game.
            opening_fen: FEN string of the starting position (unused).
            side: Side to play (unused).

        """
        self._rng.seed(seed + hash((opening_fen, side, self.engine_id)) % 1_000_003)

    def choose_move(
        self,
        state: Any,
        move_time_ms: int | None,
        depth: int | None,
    ) -> str:
        """Return a placeholder move.

        Args:
            state: Current game state (unused).
            move_time_ms: Time limit (unused).
            depth: Search depth (unused).

        Returns:
            Placeholder move "0000".

        """
        _ = (state, move_time_ms, depth)
        return "0000"


def create_engine(spec: EngineSpec, run_seed: int) -> SimulatedEngineAdapter:
    """Create an engine adapter from a specification.

    Args:
        spec: Engine specification containing configuration.
        run_seed: Base random seed for the run.

    Returns:
        A configured SimulatedEngineAdapter instance.

    """
    return SimulatedEngineAdapter(
        engine_id=spec.engine_id,
        version=spec.version,
        strength_elo=spec.strength_elo,
        draw_bias=spec.draw_bias,
        _rng=random.Random(run_seed + hash(spec.engine_id) % 10_007),
    )

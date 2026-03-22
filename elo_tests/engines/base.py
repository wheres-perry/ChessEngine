"""Base protocol for engine adapters.

This module defines the EngineAdapter protocol that all engine implementations
must conform to, enabling uniform interaction with different engine types.
"""

from __future__ import annotations

from typing import Any, Protocol


class EngineAdapter(Protocol):
    """Protocol defining the interface for chess engine adapters.

    All engine implementations must conform to this protocol to be usable
    within the Elo testing framework.
    """

    @property
    def engine_id(self) -> str:
        """Return the unique identifier for this engine."""
        ...

    @property
    def version(self) -> str:
        """Return the version string for this engine."""
        ...

    @property
    def strength_elo(self) -> float:
        """Return the engine's strength in Elo points."""
        ...

    @property
    def draw_bias(self) -> float:
        """Return the engine's draw probability bias."""
        ...

    def new_game(self, seed: int, opening_fen: str, side: str) -> None:
        """Initialize a new game for this engine.

        Args:
            seed: Random seed for reproducibility.
            opening_fen: FEN string of the starting position.
            side: Side to play ("white" or "black").

        """
        ...

    def choose_move(
        self, state: Any, move_time_ms: int | None, depth: int | None
    ) -> Any:
        """Choose a move given the current state.

        Args:
            state: Current game state (board representation).
            move_time_ms: Time limit for move selection in milliseconds.
            depth: Maximum search depth.

        Returns:
            The chosen move in UCI notation.

        """
        ...

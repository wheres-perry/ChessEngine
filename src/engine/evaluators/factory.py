"""Evaluator factory and composite evaluator.

The factory reads an ``EvaluationConfig`` and assembles a ready-to-use
``Evaluator`` instance.  By default this delegates to the C++ evaluator
for performance; the pure-Python composite path is kept for testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engine._core import chess_engine_core as core
from engine.evaluators.base import EvalComponent, Evaluator, compute_game_phase
from engine.evaluators.components import (
    KingSafetyComponent,
    MaterialComponent,
    MobilityComponent,
    PawnStructureComponent,
    PSTComponent,
)

if TYPE_CHECKING:
    from engine._core import chess_engine_core as chess
    from engine.config import EvaluationConfig


class CppEvaluatorWrapper:
    """Thin wrapper around the C++ evaluator satisfying the ``Evaluator`` protocol."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Build a C++ evaluator matching *config* flags.

        Args:
            config: Python-side evaluation configuration.

        """
        cpp_config = core.EvalConfig()
        cpp_config.use_pst = config.use_pst
        cpp_config.use_pawn_structure = config.use_pawn_structure
        cpp_config.use_mobility = config.use_mobility
        cpp_config.use_king_safety = config.use_king_safety
        cpp_config.game_stage_conscious = config.game_stage_conscious
        self._inner = core.CppEvaluator(cpp_config)

    def go(self, board: chess.Board) -> float:
        """Return centipawn score (positive = White advantage)."""
        return float(self._inner.go(board))


class CompositeEvaluator:
    """Sums contributions from an ordered list of ``EvalComponent`` objects.

    Satisfies the ``Evaluator`` protocol.
    """

    def __init__(self, components: list[EvalComponent]) -> None:
        """Initialize with a list of components.

        Args:
            components: List of evaluation components.

        """
        self._components = components

    def go(self, board: chess.Board) -> float:
        """Return aggregate centipawn score (positive = White advantage)."""
        phase = compute_game_phase(board)
        return sum(c.score(board, phase) for c in self._components)

    @property
    def components(self) -> list[EvalComponent]:
        """Introspection helper — mainly useful for tests."""
        return list(self._components)


class MockEvaluator:
    """Always returns 0.0 — useful for testing search in isolation."""

    def go(self, board: chess.Board) -> float:  # noqa: ARG002
        """Return zero regardless of board state."""
        return 0.0


class SimpleEvaluator:
    """Material-only evaluator — useful for tests needing non-zero scores.

    Satisfies the ``Evaluator`` protocol with just a ``MaterialComponent``.
    """

    def __init__(self) -> None:
        """Initialize the factory with a default material evaluator."""
        self._inner = CompositeEvaluator([MaterialComponent()])

    def go(self, board: chess.Board) -> float:
        """Return material-only evaluation score."""
        return self._inner.go(board)


class EvaluatorFactory:
    """Build an ``Evaluator`` from an ``EvaluationConfig``.

    Usage::

        evaluator = EvaluatorFactory.create(config.evaluation)
        score = evaluator.go(board)
    """

    @staticmethod
    def create(config: EvaluationConfig) -> Evaluator:
        """Assemble the evaluator according to *config* flags.

        Returns a C++ backed evaluator for performance.
        Material counting is always included as the baseline.
        """
        return CppEvaluatorWrapper(config)

    @staticmethod
    def create_composite(config: EvaluationConfig) -> CompositeEvaluator:
        """Assemble a pure-Python composite evaluator (for testing).

        Material counting is always included as the baseline.
        Each optional component is appended only when its flag is ``True``.
        When ``config.game_stage_conscious`` is set, every component that
        supports GSC will receive ``gsc=True`` so it blends its contribution
        across game phases.
        """
        gsc = config.game_stage_conscious
        components: list[EvalComponent] = [MaterialComponent()]

        if config.use_pst:
            components.append(PSTComponent(gsc=gsc))

        if config.use_pawn_structure:
            components.append(PawnStructureComponent(gsc=gsc))

        if config.use_mobility:
            components.append(MobilityComponent(gsc=gsc))

        if config.use_king_safety:
            components.append(KingSafetyComponent(gsc=gsc))

        return CompositeEvaluator(components)

    @staticmethod
    def create_mock() -> Evaluator:
        """Create a zero-valued mock evaluator."""
        return MockEvaluator()

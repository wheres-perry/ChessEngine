"""Evaluator factory and composite evaluator.

The factory reads an ``EvaluationConfig`` and assembles a ready-to-use
``Evaluator`` instance.  The composite wires together the enabled
``EvalComponent`` objects so the search engine sees a single
``.go(board)`` call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


class CompositeEvaluator:
    """Sums contributions from an ordered list of ``EvalComponent`` objects.

    Satisfies the ``Evaluator`` protocol.
    """

    def __init__(self, components: list[EvalComponent]) -> None:
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

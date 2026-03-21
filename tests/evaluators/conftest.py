"""Evaluator test fixtures."""

from __future__ import annotations

import pytest

from engine._core import chess_engine_core as chess
from engine.config import EvaluationConfig
from engine.evaluators import (
    CompositeEvaluator,
    Evaluator,
    EvaluatorFactory,
    MockEvaluator,
    SimpleEvaluator,
)


@pytest.fixture
def simple_evaluator() -> SimpleEvaluator:
    """Return a material-only evaluator for quick sanity checks."""
    return SimpleEvaluator()


@pytest.fixture
def full_evaluator() -> Evaluator:
    """Return a fully-featured evaluator built from default config (all components)."""
    return EvaluatorFactory.create(EvaluationConfig())

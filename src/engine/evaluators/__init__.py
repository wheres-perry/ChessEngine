"""Chess position evaluators.

Public API
----------
- ``Evaluator``          - protocol every evaluator satisfies
- ``EvalComponent``      - ABC for composable heuristic components
- ``CompositeEvaluator`` - sums a list of components
- ``EvaluatorFactory``   - builds an ``Evaluator`` from ``EvaluationConfig``
- ``MockEvaluator``      - always-zero evaluator for testing
- ``SimpleEvaluator``    - material-only evaluator for testing

Individual components (``MaterialComponent``, ``PSTComponent``, etc.) are
importable from ``engine.evaluators.components`` when needed for fine-grained
testing.
"""

from engine.evaluators.base import EvalComponent, Evaluator, compute_game_phase
from engine.evaluators.factory import (
    CompositeEvaluator,
    EvaluatorFactory,
    MockEvaluator,
    SimpleEvaluator,
)

__all__ = [
    "CompositeEvaluator",
    "EvalComponent",
    "Evaluator",
    "EvaluatorFactory",
    "MockEvaluator",
    "SimpleEvaluator",
    "compute_game_phase",
]

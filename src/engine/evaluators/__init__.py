"""
Chess position evaluators.

Modular evaluation system supporting multiple evaluation strategies:
- Base evaluator with shared utilities
- Handcoded evaluator with traditional chess heuristics
- Neural network evaluator for ML-based evaluation
- Simple/Mock evaluators for testing

All evaluators follow Tree 2 (State Evaluation Optimizations) dependency structure.
"""

from src.engine.evaluators.base_evaluator import (
    BaseEvaluator,
    MockEvaluator,
    SimpleEvaluator,
)
from src.engine.evaluators.handcoded_eval import HandcodedEvaluator
from src.engine.evaluators.nn_eval import NeuralNetworkEvaluator

__all__ = [
    "BaseEvaluator",
    "HandcodedEvaluator",
    "MockEvaluator",
    "NeuralNetworkEvaluator",
    "SimpleEvaluator",
]

"""Configuration solver for chess engine search components.

Validates that an ``EngineConfig`` satisfies all dependency and
hyperparameter rules defined in ``ConfigSolverRules``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from z3 import BoolVal, IntVal, is_true, simplify, substitute  # type: ignore

from engine.config_solver_rules import ConfigSolverRules

if TYPE_CHECKING:
    from engine.config import EngineConfig, SearchConfig

logger = logging.getLogger(__name__)


class ConfigSolverError(Exception):
    """Raised when a configuration violates dependency or hyperparameter rules."""


class ConfigSolver:
    """Validate feature dependencies and hyperparameters for search engines.

    Instantiate with an ``EngineConfig`` and call :meth:`solve` to run all
    rules.  On the first violation a ``ConfigSolverError`` is raised with a
    human-readable description; otherwise the ``SearchConfig`` is returned.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.search_config = config.search
        self._rules = ConfigSolverRules()

    def solve(self) -> SearchConfig:
        """Validate all rules and return the search config."""
        self._validate_global_bounds()
        self._check_rules(
            self._rules.eval_rules,
            self._build_eval_substitutions(),
        )
        self._check_rules(
            self._rules.search_rules,
            self._build_search_substitutions(),
        )

        logger.debug(
            "All feature dependencies and hyperparameters validated successfully"
        )
        return self.search_config

    # ------------------------------------------------------------------
    # Global bounds (not expressible as z3 Implies over config fields)
    # ------------------------------------------------------------------

    def _validate_global_bounds(self) -> None:
        if self.config.search_depth < 1:
            raise ConfigSolverError(
                f"Search depth must be at least 1, got {self.config.search_depth}"
            )
        if self.config.search_depth > 128:
            raise ConfigSolverError(
                f"Search depth too high (max 128), got {self.config.search_depth}"
            )
        if self.search_config.max_time is not None and self.search_config.max_time <= 0:
            raise ConfigSolverError(
                f"Minimax timeout must be positive, got {self.search_config.max_time}"
            )

    # ------------------------------------------------------------------
    # Substitution builders
    # ------------------------------------------------------------------

    def _build_eval_substitutions(self) -> list[tuple[object, object]]:
        evl = self.config.evaluation
        return [
            (var, BoolVal(getattr(evl, name)))
            for name, var in self._rules.eval_vars.items()
        ]

    def _build_search_substitutions(self) -> list[tuple[object, object]]:
        cfg = self.search_config
        subs: list[tuple[object, object]] = [
            (var, BoolVal(getattr(cfg, name)))
            for name, var in self._rules.search_bool_vars.items()
        ]
        subs += [
            (var, IntVal(getattr(cfg, name)))
            for name, var in self._rules.search_int_vars.items()
        ]
        return subs

    # ------------------------------------------------------------------
    # Rule checker
    # ------------------------------------------------------------------

    @staticmethod
    def _check_rules(
        rules: list[tuple[str, object]],
        substitutions: list[tuple[object, object]],
    ) -> None:
        for description, constraint in rules:
            result = simplify(substitute(constraint, substitutions))
            if not is_true(result):
                raise ConfigSolverError(description)

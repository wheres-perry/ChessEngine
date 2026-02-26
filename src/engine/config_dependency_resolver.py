"""Dependency resolver for chess engine search components."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.config import EngineConfig, SearchConfig

logger = logging.getLogger(__name__)


class DependencyResolutionError(Exception):
    """Raised when feature dependencies cannot be resolved or are misconfigured."""


class DependencyResolver:
    """Validate feature dependencies and hyperparameters for search engines."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.search_config = config.search

    def resolve(self) -> SearchConfig:
        """Validate all dependencies and hyperparameter bounds.

        Returns the search config.
        """
        self._validate_global_bounds()
        self._validate_zobrist_dependencies()
        self._validate_alpha_beta_dependencies()
        self._validate_move_ordering_dependencies()
        self._validate_search_refinement_dependencies()
        self._validate_hyperparameter_bounds()

        logger.debug(
            "All feature dependencies and hyperparameters validated successfully"
        )
        return self.search_config

    def _validate_global_bounds(self) -> None:
        """Validate top-level engine boundaries."""
        if self.config.search_depth < 1:
            raise DependencyResolutionError(
                f"Search depth must be at least 1, got {self.config.search_depth}"
            )
        if self.config.search_depth > 128:
            raise DependencyResolutionError(
                f"Search depth too high (max 128), got {self.config.search_depth}"
            )

        if self.search_config.max_time is not None and self.search_config.max_time <= 0:
            raise DependencyResolutionError(
                f"Minimax timeout must be positive, got {self.search_config.max_time}"
            )

    def _validate_zobrist_dependencies(self) -> None:
        """Validate transposition table sub-features."""
        cfg = self.search_config

        if cfg.use_tt_aging and not cfg.use_transposition_table:
            raise DependencyResolutionError(
                "TT aging requires the transposition table to be enabled."
            )

    def _validate_alpha_beta_dependencies(self) -> None:
        """Validate features that strictly require alpha-beta bounds."""
        cfg = self.search_config

        alpha_beta_features = [
            (cfg.use_pvs, "Principal Variation Search"),
            (cfg.use_aspiration_windows, "Aspiration windows"),
            (cfg.use_null_move_pruning, "Null move pruning"),
            (cfg.use_futility_pruning, "Futility pruning"),
            (cfg.use_extended_futility_pruning, "Extended futility pruning"),
            (cfg.use_reverse_futility_pruning, "Reverse futility pruning"),
            (cfg.use_check_extensions, "Check extensions"),
            (cfg.use_quiescence_search, "Quiescence search"),
        ]

        for feature_enabled, feature_name in alpha_beta_features:
            if feature_enabled and not cfg.use_alpha_beta:
                raise DependencyResolutionError(
                    f"{feature_name} requires alpha-beta pruning to be enabled."
                )

    def _validate_move_ordering_dependencies(self) -> None:
        """Validate sorting heuristics and reductions dependent on ordering."""
        cfg = self.search_config

        move_ordering_features = [
            (cfg.use_mvv_lva, "MVV-LVA ordering"),
            (cfg.use_see_ordering, "SEE ordering"),
            (cfg.use_killer_moves, "Killer moves"),
            (cfg.use_history_heuristic, "History heuristic"),
            (cfg.use_countermove_heuristic, "Countermove heuristic"),
            (cfg.use_hash_move_ordering, "Hash move ordering"),
        ]

        for feature_enabled, feature_name in move_ordering_features:
            if feature_enabled and not cfg.use_move_ordering:
                raise DependencyResolutionError(
                    f"{feature_name} requires move ordering to be enabled."
                )

        if cfg.use_hash_move_ordering and not cfg.use_transposition_table:
            raise DependencyResolutionError(
                "Hash move ordering requires the transposition table."
            )

        if cfg.use_countermove_heuristic and not cfg.use_history_heuristic:
            raise DependencyResolutionError(
                "Countermove heuristic structurally depends on the history heuristic."
            )

        if cfg.use_lmr and not (cfg.use_alpha_beta and cfg.use_move_ordering):
            raise DependencyResolutionError(
                "Late Move Reduction (LMR) requires both alpha-beta pruning "
                "and reliable move ordering to prevent filtering optimal lines."
            )

    def _validate_search_refinement_dependencies(self) -> None:
        """Validate IID and QS-specific pruning logic."""
        cfg = self.search_config

        if cfg.use_iid and not cfg.use_transposition_table:
            raise DependencyResolutionError(
                "Internal Iterative Deepening (IID) requires the transposition table."
            )

        if cfg.use_delta_pruning and not cfg.use_quiescence_search:
            raise DependencyResolutionError(
                "Delta pruning is strictly a Quiescence Search optimization."
            )

        if cfg.use_see_pruning_in_qs and not cfg.use_quiescence_search:
            raise DependencyResolutionError(
                "SEE pruning in QS requires Quiescence Search to be enabled."
            )

    def _validate_hyperparameter_bounds(self) -> None:
        """Ensure all tunable values are mathematically safe for the engine."""
        cfg = self.search_config

        if cfg.use_lmr:
            if cfg.lmr_min_depth < 1:
                raise DependencyResolutionError("LMR minimum depth must be >= 1.")
            if cfg.lmr_min_move_number < 1:
                raise DependencyResolutionError("LMR minimum move number must be >= 1.")

        if cfg.use_iid and cfg.iid_min_depth <= cfg.iid_depth_reduction:
            raise DependencyResolutionError(
                "IID minimum depth must be strictly greater than its depth reduction."
            )

        if cfg.use_null_move_pruning and cfg.nmp_reduction_r < 1:
            raise DependencyResolutionError("NMP reduction constant (R) must be >= 1.")

        if cfg.use_history_heuristic and cfg.history_max_score <= 0:
            raise DependencyResolutionError(
                "History max score ceiling must be positive."
            )

        if cfg.use_killer_moves and cfg.killer_slots_per_ply < 1:
            raise DependencyResolutionError(
                "Must allocate at least 1 killer slot per ply."
            )

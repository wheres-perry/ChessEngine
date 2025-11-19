"""Dependency resolver for chess engine search components."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.config import EngineConfig, SearchConfig

logger = logging.getLogger(__name__)


class DependencyResolutionError(Exception):
    """Raised when feature dependencies cannot be resolved."""


class DependencyResolver:
    """Validate feature dependencies for search engines."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.search_config = config.search

    def resolve(self) -> SearchConfig:
        """Validate dependencies and return the search config."""
        self._validate_dependencies()
        return self.search_config

    def _validate_dependencies(self) -> None:
        """Run all dependency checks."""
        self._validate_zobrist_dependencies()
        self._validate_alpha_beta_dependencies()
        self._validate_move_ordering_dependencies()
        self._validate_search_refinement_dependencies()
        logger.debug("All feature dependencies validated successfully")

    def _validate_zobrist_dependencies(self) -> None:
        cfg = self.search_config

        if cfg.use_transposition_table and not cfg.use_zobrist:
            raise DependencyResolutionError(
                "Transposition table requires Zobrist hashing to be enabled"
            )

        if cfg.use_zobrist and not cfg.use_transposition_table:
            raise DependencyResolutionError(
                "Zobrist hashing should only be enabled with transposition table"
            )

        if cfg.use_tt_aging and not cfg.use_zobrist:
            raise DependencyResolutionError(
                "TT aging requires Zobrist hashing to be enabled"
            )

    def _validate_alpha_beta_dependencies(self) -> None:
        cfg = self.search_config

        if cfg.use_pvs and not (cfg.use_alpha_beta and cfg.use_iddfs):
            raise DependencyResolutionError(
                "Principal Variation Search requires both alpha-beta and IDDFS"
            )

        if cfg.use_aspiration_windows and not cfg.use_alpha_beta:
            raise DependencyResolutionError(
                "Aspiration windows require alpha-beta pruning"
            )

        if cfg.use_null_move_pruning and not cfg.use_alpha_beta:
            raise DependencyResolutionError(
                "Null move pruning requires alpha-beta pruning"
            )

        if cfg.use_futility_pruning and not cfg.use_alpha_beta:
            raise DependencyResolutionError(
                "Futility pruning requires alpha-beta pruning"
            )

        if cfg.use_extended_futility_pruning and not cfg.use_alpha_beta:
            raise DependencyResolutionError(
                "Extended futility pruning requires alpha-beta pruning"
            )

        if cfg.use_reverse_futility_pruning and not cfg.use_alpha_beta:
            raise DependencyResolutionError(
                "Reverse futility pruning requires alpha-beta pruning"
            )

    def _validate_move_ordering_dependencies(self) -> None:
        cfg = self.search_config

        if cfg.use_lmr and not (cfg.use_alpha_beta and cfg.use_move_ordering):
            raise DependencyResolutionError(
                "Late Move Reduction (LMR) requires both alpha-beta pruning "
                "and move ordering"
            )

        if cfg.use_hash_move_ordering and not cfg.use_move_ordering:
            raise DependencyResolutionError(
                "Hash move ordering requires move ordering to be enabled"
            )

        if cfg.use_hash_move_ordering and not cfg.use_transposition_table:
            raise DependencyResolutionError(
                "Hash move ordering requires transposition table"
            )

        move_ordering_features = [
            (cfg.use_mvv_lva, "MVV-LVA ordering"),
            (cfg.use_see_ordering, "SEE ordering"),
            (cfg.use_killer_moves, "Killer moves"),
            (cfg.use_history_heuristic, "History heuristic"),
            (cfg.use_countermove_heuristic, "Countermove heuristic"),
        ]

        for feature_enabled, feature_name in move_ordering_features:
            if feature_enabled and not cfg.use_move_ordering:
                raise DependencyResolutionError(
                    f"{feature_name} requires move ordering to be enabled"
                )

        if cfg.use_countermove_heuristic and not cfg.use_history_heuristic:
            raise DependencyResolutionError(
                "Countermove heuristic usually depends on history heuristic"
            )

    def _validate_search_refinement_dependencies(self) -> None:
        cfg = self.search_config

        if cfg.use_iid and not (cfg.use_iddfs and cfg.use_transposition_table):
            raise DependencyResolutionError(
                "Internal Iterative Deepening (IID) requires IDDFS and TT"
            )

        if cfg.use_delta_pruning and not cfg.use_quiescence_search:
            raise DependencyResolutionError("Delta pruning requires quiescence search")

        if cfg.use_see_pruning_in_qs and not cfg.use_quiescence_search:
            raise DependencyResolutionError(
                "SEE pruning in QS requires quiescence search"
            )

        if cfg.use_check_extensions and not cfg.use_minimax:
            raise DependencyResolutionError(
                "Check extensions require the base minimax search to be enabled"
            )

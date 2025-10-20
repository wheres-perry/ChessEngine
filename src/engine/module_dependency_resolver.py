"""
Dependency resolver for chess engine search components.

This module validates feature dependencies and resolves them into a set of
flags that can be safely used by search engines. It ensures that features
that depend on other features are only enabled when their dependencies are met.
"""

import logging

from src.engine.config import EngineConfig, SearchConfig

logger = logging.getLogger(__name__)


class DependencyResolutionError(Exception):
    """Raised when feature dependencies cannot be resolved."""


class DependencyResolver:
    """
    Resolves and validates feature dependencies for search engines.

    Takes an EngineConfig and validates that all feature dependencies are met,
    returning the validated SearchConfig.
    """

    def __init__(self, config: EngineConfig) -> None:
        """
        Initialize the dependency resolver.

        Args:
            config: Engine configuration to resolve dependencies for
        """
        self.config = config
        # Accessing the search config via the updated attribute name 'search'
        self.search_config = config.search

    def resolve(self) -> SearchConfig:
        """
        Resolve dependencies and return validated search config.

        Returns:
            SearchConfig object with all dependencies validated

        Raises:
            DependencyResolutionError: If dependencies cannot be resolved
        """
        self._validate_dependencies()
        return self.search_config

    def _validate_dependencies(self) -> None:
        """
        Validate that all feature dependencies are met.
        """
        # The order matters here. Core dependencies must be checked first.
        self._validate_core_alpha_beta()
        self._validate_transposition_table()

        # Features relying on the core
        self._validate_move_ordering()
        self._validate_quiescence_search()
        self._validate_search_refinements()
        self._validate_pruning_and_reductions()
        self._validate_extensions()
        self._validate_parallelization()
        self._validate_external_knowledge()

        # Mutual exclusions must be checked last
        self._validate_mutual_exclusions()

        logger.debug("All feature dependencies validated successfully")

    def _check_dependency(
        self, feature_enabled: bool, prerequisite_enabled: bool, message: str
    ) -> None:
        """Helper to check a dependency and raise an error if not met."""
        if feature_enabled and not prerequisite_enabled:
            raise DependencyResolutionError(message)

    # ========================================================================
    # Dependency Validation Groups
    # ========================================================================

    def _validate_core_alpha_beta(self) -> None:
        """Validate features that fundamentally rely on the Alpha-Beta framework."""
        cfg = self.search_config
        if cfg.use_alpha_beta:
            return

        # If Alpha-Beta is disabled, many optimizations are incompatible.
        reason = "requires Alpha-Beta Pruning."
        self._check_dependency(
            cfg.use_move_ordering,
            prerequisite_enabled=False,
            message=f"Move Ordering {reason}",
        )
        self._check_dependency(
            cfg.use_pvs,
            prerequisite_enabled=False,
            message=f"PVS {reason}",
        )
        self._check_dependency(
            cfg.use_mtdf,
            prerequisite_enabled=False,
            message=f"MTD(f) {reason}",
        )
        self._check_dependency(
            cfg.use_aspiration_windows,
            prerequisite_enabled=False,
            message=f"Aspiration Windows {reason}",
        )
        self._check_dependency(
            cfg.use_null_move_pruning,
            prerequisite_enabled=False,
            message=f"Null Move Pruning {reason}",
        )
        self._check_dependency(
            cfg.use_lmr,
            prerequisite_enabled=False,
            message=f"LMR {reason}",
        )
        self._check_dependency(
            cfg.use_futility_pruning,
            prerequisite_enabled=False,
            message=f"Futility Pruning {reason}",
        )
        self._check_dependency(
            cfg.use_razoring,
            prerequisite_enabled=False,
            message=f"Razoring {reason}",
        )
        self._check_dependency(
            cfg.use_parallel_search,
            prerequisite_enabled=False,
            message=f"Parallel Search {reason}",
        )
        # QS relies on the AB framework (Stand-Pat mechanism)
        self._check_dependency(
            cfg.use_quiescence_search,
            prerequisite_enabled=False,
            message=f"Quiescence Search {reason}",
        )

    def _validate_transposition_table(self) -> None:
        """Validate TT and Zobrist related dependencies."""
        cfg = self.search_config

        # Zobrist is required for TT
        self._check_dependency(
            cfg.use_transposition_table,
            cfg.use_zobrist,
            "Transposition table requires Zobrist hashing.",
        )

        if not cfg.use_transposition_table:
            # Ensure features requiring TT are off if TT is off
            reason = "requires Transposition Table."
            self._check_dependency(
                cfg.use_tt_aging,
                prerequisite_enabled=False,
                message=f"TT Aging {reason}",
            )
            self._check_dependency(
                cfg.use_iid,
                prerequisite_enabled=False,
                message=f"IID {reason}",
            )
            self._check_dependency(
                cfg.use_mtdf,
                prerequisite_enabled=False,
                message=f"MTD(f) {reason}",
            )
            self._check_dependency(
                cfg.use_singular_extensions,
                prerequisite_enabled=False,
                message=f"Singular Extensions {reason}",
            )

    def _validate_move_ordering(self) -> None:
        """Validate move ordering related dependencies."""
        cfg = self.search_config

        if not cfg.use_move_ordering:
            # If master switch is off, all sub-features must be off.
            reason = "requires Move Ordering master switch."
            self._check_dependency(
                cfg.use_hash_move_ordering,
                prerequisite_enabled=False,
                message=f"Hash Move Ordering {reason}",
            )
            self._check_dependency(
                cfg.use_mvv_lva,
                prerequisite_enabled=False,
                message=f"MVV-LVA {reason}",
            )
            self._check_dependency(
                cfg.use_see_ordering,
                prerequisite_enabled=False,
                message=f"SEE Ordering {reason}",
            )
            self._check_dependency(
                cfg.use_killer_moves,
                prerequisite_enabled=False,
                message=f"Killer Moves {reason}",
            )
            self._check_dependency(
                cfg.use_history_heuristic,
                prerequisite_enabled=False,
                message=f"History Heuristic {reason}",
            )
            self._check_dependency(
                cfg.use_countermove_heuristic,
                prerequisite_enabled=False,
                message=f"Countermove Heuristic {reason}",
            )
            return

        # PVS efficiency relies heavily on move ordering
        self._check_dependency(
            cfg.use_pvs,
            cfg.use_move_ordering,
            "PVS requires Move Ordering to be effective.",
        )

        # Hash move ordering requires TT
        self._check_dependency(
            cfg.use_hash_move_ordering,
            cfg.use_transposition_table,
            "Hash move ordering requires Transposition Table.",
        )

        # Countermove often relies on History infrastructure
        self._check_dependency(
            cfg.use_countermove_heuristic,
            cfg.use_history_heuristic,
            "Countermove Heuristic usually depends on History Heuristic "
            "implementation.",
        )

        # LMR requires dynamic move ordering (History or Killer) to be effective.
        if cfg.use_lmr and not (cfg.use_history_heuristic or cfg.use_killer_moves):
            raise DependencyResolutionError(
                "LMR requires dynamic move ordering (History or Killer moves) "
                "to be effective."
            )

    def _validate_quiescence_search(self) -> None:
        """Validate QS related dependencies."""
        cfg = self.search_config

        if not cfg.use_quiescence_search:
            reason = "requires Quiescence Search."
            self._check_dependency(
                cfg.use_delta_pruning,
                prerequisite_enabled=False,
                message=f"Delta Pruning {reason}",
            )
            self._check_dependency(
                cfg.use_see_pruning_in_qs,
                prerequisite_enabled=False,
                message=f"SEE Pruning in QS {reason}",
            )

    def _validate_search_refinements(self) -> None:
        """Validate search refinement dependencies."""
        cfg = self.search_config

        # Aspiration windows require IDDFS to establish bounds from previous iterations
        self._check_dependency(
            cfg.use_aspiration_windows,
            cfg.use_iddfs,
            "Aspiration Windows require Iterative Deepening (IDDFS).",
        )

        # Note: The original check for IID requiring IDDFS is removed, as IID primarily
        # depends on the TT (checked in _validate_transposition_table).

    def _validate_pruning_and_reductions(self) -> None:
        """Validate pruning and reduction dependencies."""
        cfg = self.search_config

        # Extended futility depends on basic futility
        self._check_dependency(
            cfg.use_extended_futility_pruning,
            cfg.use_futility_pruning,
            "Extended Futility Pruning requires basic Futility Pruning.",
        )

        # ProbCut uses SEE to estimate outcomes
        self._check_dependency(
            cfg.use_probcut,
            cfg.use_see_ordering,
            "ProbCut requires SEE Ordering (Static Exchange Evaluation).",
        )

    def _validate_extensions(self) -> None:
        # Currently, only Singular Extensions have a hard dependency (on TT),
        # checked in _validate_transposition_table.
        pass

    def _validate_parallelization(self) -> None:
        """Validate parallel search dependencies."""
        cfg = self.search_config

        if not cfg.use_parallel_search:
            reason = "requires Parallel Search master switch."
            self._check_dependency(
                cfg.use_naive_parallel,
                prerequisite_enabled=False,
                message=f"Naive Parallel {reason}",
            )
            self._check_dependency(
                cfg.use_lazy_smp,
                prerequisite_enabled=False,
                message=f"Lazy SMP {reason}",
            )
            self._check_dependency(
                cfg.use_ybwc,
                prerequisite_enabled=False,
                message=f"YBWC {reason}",
            )
            self._check_dependency(
                cfg.use_dts,
                prerequisite_enabled=False,
                message=f"DTS {reason}",
            )
            return

        # If parallel search is on, threads must be > 1
        if cfg.num_threads <= 1:
            raise DependencyResolutionError(
                "Parallel Search enabled but num_threads is not > 1."
            )

    def _validate_external_knowledge(self) -> None:
        """Validate Books and Tablebases."""
        cfg = self.search_config

        if cfg.use_opening_book and not cfg.opening_book_path:
            raise DependencyResolutionError(
                "Opening Book enabled but opening_book_path is not specified."
            )

        if cfg.use_endgame_tablebases and not cfg.egtb_path:
            raise DependencyResolutionError(
                "Endgame Tablebases (EGTB) enabled but egtb_path is not specified."
            )

    # ========================================================================
    # Mutual Exclusion Validation
    # ========================================================================

    def _validate_mutual_exclusions(self) -> None:
        """Ensure mutually exclusive features are not enabled simultaneously."""
        cfg = self.search_config

        # 1. Search Drivers (MTD(f) vs PVS/Aspiration)
        # MTD(f) is an alternative driver to PVS and Aspiration Windows.
        if cfg.use_mtdf and (cfg.use_pvs or cfg.use_aspiration_windows):
            raise DependencyResolutionError(
                "MTD(f) is mutually exclusive with PVS and Aspiration Windows. "
                "Enable only one search driver approach."
            )

        # 2. Parallel Algorithms
        if cfg.use_parallel_search:
            parallel_algo_group = [
                (cfg.use_ybwc, "YBWC"),
                (cfg.use_dts, "DTS"),
                (cfg.use_lazy_smp, "Lazy SMP"),
                (cfg.use_naive_parallel, "Naive Parallel"),
            ]
            enabled_algos = [name for enabled, name in parallel_algo_group if enabled]

            if len(enabled_algos) > 1:
                raise DependencyResolutionError(
                    f"Parallel algorithms are mutually exclusive. Cannot enable "
                    f"{', '.join(enabled_algos)} simultaneously."
                )
            if len(enabled_algos) == 0:
                raise DependencyResolutionError(
                    "Parallel Search is enabled, but no specific parallel algorithm "
                    "(e.g., YBWC, Lazy SMP) is selected."
                )

"""
Dependency resolver for chess engine search components.

This module validates feature dependencies and resolves them into a set of
flags that can be safely used by search engines. It ensures that features
that depend on other features are only enabled when their dependencies are met.

This is complementary to the EngineConfig validation - while EngineConfig
validates at construction time, DependencyResolver provides runtime validation
and can be used to verify configurations during search initialization.

Tree 1: Move Exploration (Search) Optimizations
A (Minimax) -> B (Alpha-Beta), C (IDDFS), D (Move Ordering), L (TT), I (Check Ext)
B -> G (PVS), N (Quiescence)
C + L -> M (IID)
D -> E (Killer), F (History)
D + B -> K (LMR), J (Futility)
G -> H (NMP)

Tree 2: State Evaluation Optimizations
E0 (Material) -> E1 (PST) -> E2 (Tapered), E3 (Pawn), E4 (Mobility), E6 (SEE)
E4 -> E5 (King Safety)
E2 -> E9 (Endgame Tables)
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
    returning the validated SearchConfig. This provides an additional layer
    of validation beyond the EngineConfig.__post_init__ validation.
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
        """Validate that all feature dependencies are met (Tree 1)."""
        self._validate_core_dependencies()
        self._validate_zobrist_dependencies()
        self._validate_alpha_beta_dependencies()
        self._validate_move_ordering_dependencies()
        self._validate_search_refinement_dependencies()
        logger.debug("All feature dependencies validated successfully")

    def _validate_core_dependencies(self) -> None:
        """Validate core minimax dependencies (Node A)."""
        cfg = self.search_config

        if not cfg.use_minimax:
            # If minimax is disabled, no other features should be enabled
            advanced_features = [
                cfg.use_alpha_beta,
                cfg.use_iddfs,
                cfg.use_move_ordering,
                cfg.use_transposition_table,
                cfg.use_check_extensions,
            ]
            if any(advanced_features):
                raise DependencyResolutionError(
                    "All search features require basic minimax to be enabled"
                )

    def _validate_zobrist_dependencies(self) -> None:
        """Validate Zobrist hashing related dependencies (Node L)."""
        cfg = self.search_config

        # TT requires Zobrist
        if cfg.use_transposition_table and not cfg.use_zobrist:
            raise DependencyResolutionError(
                "Transposition table requires Zobrist hashing to be enabled"
            )

        # TT requires minimax
        if cfg.use_transposition_table and not cfg.use_minimax:
            raise DependencyResolutionError(
                "Transposition table requires basic minimax"
            )

        # TT aging requires Zobrist
        if cfg.use_tt_aging and not cfg.use_zobrist:
            raise DependencyResolutionError(
                "TT aging requires Zobrist hashing to be enabled"
            )

        # Zobrist is only useful with TT
        if cfg.use_zobrist and not cfg.use_transposition_table:
            raise DependencyResolutionError(
                "Zobrist hashing should only be enabled with transposition table"
            )

    def _validate_alpha_beta_dependencies(self) -> None:
        """Validate alpha-beta pruning related dependencies (Node B)."""
        cfg = self.search_config

        # Alpha-beta requires minimax
        if cfg.use_alpha_beta and not cfg.use_minimax:
            raise DependencyResolutionError("Alpha-beta pruning requires basic minimax")

        # PVS requires alpha-beta + IDDFS (Node G)
        if cfg.use_pvs and not (cfg.use_alpha_beta and cfg.use_iddfs):
            raise DependencyResolutionError(
                "Principal Variation Search requires both alpha-beta and IDDFS"
            )

        # Quiescence requires alpha-beta (Node N)
        if cfg.use_quiescence_search and not cfg.use_alpha_beta:
            raise DependencyResolutionError(
                "Quiescence search requires alpha-beta pruning"
            )

        # Aspiration windows require alpha-beta
        if cfg.use_aspiration_windows and not cfg.use_alpha_beta:
            raise DependencyResolutionError(
                "LMR requires dynamic move ordering (History or Killer moves) "
                "to be effective."
            )

        # Null move pruning requires alpha-beta (Node H)
        if cfg.use_null_move_pruning and not cfg.use_alpha_beta:
            raise DependencyResolutionError(
                "Null move pruning requires alpha-beta pruning"
            )

        # Futility pruning variants require alpha-beta (Node J)
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
        """Validate move ordering related dependencies (Nodes D, E, F)."""
        cfg = self.search_config

        # Move ordering requires minimax (Node D)
        if cfg.use_move_ordering and not cfg.use_minimax:
            raise DependencyResolutionError("Move ordering requires basic minimax")

        # LMR requires alpha-beta + move ordering (Node K)
        if cfg.use_lmr and not (cfg.use_alpha_beta and cfg.use_move_ordering):
            raise DependencyResolutionError(
                "Late Move Reduction (LMR) requires both alpha-beta pruning "
                "and move ordering"
            )
            self._check_dependency(
                cfg.use_dts,
                prerequisite_enabled=False,
                message=f"DTS {reason}",
            )
            return

        # Hash move ordering requires move ordering + TT
        if cfg.use_hash_move_ordering and not cfg.use_move_ordering:
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

        # Static ordering features require move ordering
        static_ordering_features = [
            (cfg.use_mvv_lva, "MVV-LVA ordering"),
            (cfg.use_see_ordering, "SEE ordering"),
        ]

        for feature_enabled, feature_name in static_ordering_features:
            if feature_enabled and not cfg.use_move_ordering:
                raise DependencyResolutionError(
                    f"{feature_name} requires move ordering to be enabled"
                )

        # Killer and history heuristics require move ordering + alpha-beta (Nodes E, F)
        dynamic_ordering_features = [
            (cfg.use_killer_moves, "Killer moves"),
            (cfg.use_history_heuristic, "History heuristic"),
            (cfg.use_countermove_heuristic, "Countermove heuristic"),
        ]

        for feature_enabled, feature_name in dynamic_ordering_features:
            if feature_enabled and not (cfg.use_move_ordering and cfg.use_alpha_beta):
                raise DependencyResolutionError(
                    f"{feature_name} requires both move ordering and alpha-beta pruning"
                )

    def _validate_search_refinement_dependencies(self) -> None:
        """Validate search refinement related dependencies (Nodes C, M, N)."""
        cfg = self.search_config

        # IDDFS requires minimax (Node C)
        if cfg.use_iddfs and not cfg.use_minimax:
            raise DependencyResolutionError(
                "Iterative Deepening requires basic minimax"
            )

        # IID requires IDDFS + TT (Node M)
        if cfg.use_iid and not (cfg.use_iddfs and cfg.use_transposition_table):
            raise DependencyResolutionError(
                "Internal Iterative Deepening (IID) requires both IDDFS and "
                "transposition table"
            )

        # Quiescence search extensions (Node N)
        if cfg.use_delta_pruning and not cfg.use_quiescence_search:
            raise DependencyResolutionError("Delta pruning requires quiescence search")

        if cfg.use_see_pruning_in_qs and not cfg.use_quiescence_search:
            raise DependencyResolutionError(
                "SEE pruning in QS requires quiescence search"
            )

        # Check extensions require minimax (Node I)
        if cfg.use_check_extensions and not cfg.use_minimax:
            raise DependencyResolutionError("Check extensions require basic minimax")

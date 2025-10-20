from dataclasses import dataclass, field
from typing import Literal

from src.engine.constants import DEFAULT_DEPTH, DEFAULT_TIMEOUT


@dataclass
class SearchConfig:
    """
    Configuration for the chess engine search algorithm and optimizations.
    Features are grouped by their function based on the dependency graph.
    """

    # ========================================================================
    # 1. General Search Settings
    # ========================================================================
    max_time: float | None = DEFAULT_TIMEOUT
    max_depth: int | None = None  # Useful for depth-limited searches

    # ========================================================================
    # 2. Core Search
    # ========================================================================
    # The foundational search structure (Assumes NegaMax framework).
    use_alpha_beta: bool = True  # Alpha-Beta Pruning (Essential optimization)
    use_iddfs: bool = True  # Iterative Deepening Depth-First Search
    use_quiescence_search: bool = True  # (QS) Search tactical sequences at leaf nodes

    # ========================================================================
    # 3. Memory and Transposition Tables (TT)
    # ========================================================================
    # Techniques for reusing past search results.
    use_zobrist: bool = True  # Hashing method (Required for TT)
    use_transposition_table: bool = True  # The main cache for search results
    # Strategy for replacing old entries (e.g., depth-preferred or aging)
    use_tt_aging: bool = True

    # ========================================================================
    # 4. Move Ordering
    # ========================================================================
    # Techniques to maximize Alpha-Beta cutoffs.
    use_move_ordering: bool = True  # Master switch for move ordering heuristics

    # 4a. Prioritized Moves
    use_hash_move_ordering: bool = True  # Prioritize the move suggested by the TT

    # 4b. Static Ordering (Captures)
    use_mvv_lva: bool = True  # Most Valuable Victim / Least Valuable Aggressor
    use_see_ordering: bool = (
        True  # Static Exchange Evaluation (SEE) for ordering captures
    )

    # 4c. Dynamic Ordering (Quiet Moves)
    use_killer_moves: bool = True  # Store recent moves that caused beta cutoffs
    use_history_heuristic: bool = True  # Score moves based on historical cutoff success
    use_countermove_heuristic: bool = (
        True  # Store moves that counter the opponent's previous move
    )

    # ========================================================================
    # 5. Search Refinements
    # ========================================================================
    # Improvements to the core Alpha-Beta framework.
    use_pvs: bool = True  # Principal Variation Search (PVS / NegaScout)
    use_aspiration_windows: bool = True  # Narrow the initial Alpha-Beta window
    use_iid: bool = (
        True  # Internal Iterative Deepening (Search shallow if no hash move)
    )

    # ========================================================================
    # 6. Search Reductions and Pruning
    # ========================================================================
    # Techniques to aggressively reduce the search space.

    # 6a. Reductions (Reducing depth)
    use_lmr: bool = True  # Late Move Reductions

    # 6b. Aggressive Forward Pruning
    use_null_move_pruning: bool = True  # (NMP)

    # 6c. Pruning near leaf nodes (Futility variants)
    use_futility_pruning: bool = True
    use_extended_futility_pruning: bool = True
    use_reverse_futility_pruning: bool = True  # (Static Null Move Pruning)

    # 6d. Pruning in Quiescence Search
    use_delta_pruning: bool = True
    use_see_pruning_in_qs: bool = True  # Use SEE to prune bad captures in Q-Search

    # 6e. Advanced/Experimental Pruning
    use_probcut: bool = False  # Pruning based on probability
    use_multicut_pruning: bool = False
    use_razoring: bool = False  # Aggressive pruning at shallow depths (Risky)

    # ========================================================================
    # 7. Search Extensions
    # ========================================================================
    # Techniques to increase the search depth in critical positions.
    use_check_extensions: bool = True
    use_recapture_extensions: bool = False
    use_singular_extensions: bool = (
        False  # Extend if a move is significantly better than all others
    )

    # ========================================================================
    # 8. Parallelization & Alternatives
    # ========================================================================
    use_parallel_search: bool = False  # Master switch for multi-threading

    # Specific Parallel Algorithms (Mutually Exclusive):
    use_naive_parallel: bool = False  # (NEW) Basic root split parallelism
    use_lazy_smp: bool = False  # Lazy Symmetric Multiprocessing
    use_ybwc: bool = False  # Young Brothers Wait Concept
    use_dts: bool = False  # Dynamic Tree Splitting
    num_threads: int = 1

    # Alternative Search Drivers (Usually exclusive of standard PVS/ABP)
    use_mtdf: bool = False  # Memory-enhanced Test Driver (MTD-f)

    # ========================================================================
    # 9. External Knowledge (Books and Tablebases)
    # ========================================================================
    use_opening_book: bool = False
    opening_book_path: str | None = None

    use_endgame_tablebases: bool = False  # (EGTB, e.g., Syzygy)
    egtb_path: str | None = None
    egtb_probe_depth: int = 6  # Depth at which to start probing EGTB during search


@dataclass
class EvaluationConfig:
    """Configuration for the board evaluation."""

    evaluator_type: Literal["simple", "mock", "complex"] = "complex"

    # Complex evaluation flags
    use_material: bool = True
    use_pst: bool = True  # Piece-Square Tables
    use_mobility: bool = True
    use_pawn_structure: bool = True
    use_king_safety: bool = True


@dataclass
class EngineConfig:
    """Top-level configuration for the chess engine."""

    search: SearchConfig = field(default_factory=SearchConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    search_depth: int = DEFAULT_DEPTH

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration for consistency and correctness."""
        # Validate search depth
        if self.search_depth < 1:
            raise ValueError(
                f"Search depth must be at least 1, got {self.search_depth}"
            )
        # Increased max depth limit as modern engines can go much deeper than 20.
        if self.search_depth > 128:
            raise ValueError(
                f"Search depth too high (max 128), got {self.search_depth}"
            )

        # Validate search configuration (Basic value checks)
        self._validate_search_config()

        # Validate evaluation configuration
        self._validate_evaluation_config()

    def _validate_search_config(self) -> None:
        """Validate search-specific configuration values."""
        search_config = self.search

        # Validate timeout
        if search_config.max_time is not None and search_config.max_time <= 0:
            raise ValueError(
                f"Search timeout must be positive, got {search_config.max_time}"
            )

        # Validate thread count
        if search_config.num_threads < 1:
            raise ValueError(
                f"Number of threads must be at least 1, got {search_config.num_threads}"
            )

    def _validate_evaluation_config(self) -> None:
        """Validate evaluation-specific configuration."""
        eval_config = self.evaluation

        # (Validation logic for evaluation remains the same as provided by the user)
        if eval_config.evaluator_type == "simple" and any(
            [
                eval_config.use_pst,
                eval_config.use_mobility,
                eval_config.use_pawn_structure,
                eval_config.use_king_safety,
            ]
        ):
            flag_list: list[str] = []
            for name, enabled in [
                ("use_pst", eval_config.use_pst),
                ("use_mobility", eval_config.use_mobility),
                ("use_pawn_structure", eval_config.use_pawn_structure),
                ("use_king_safety", eval_config.use_king_safety),
            ]:
                if enabled:
                    flag_list.append(name)

            formatted_flags = ", ".join(flag_list)
            raise ValueError(
                f"Complex evaluation flags [{formatted_flags}] cannot be used with "
                "simple evaluator. Use 'complex' evaluator type or disable these flags."
            )

        if eval_config.evaluator_type == "mock":
            all_flags = [
                ("use_material", eval_config.use_material),
                ("use_pst", eval_config.use_pst),
                ("use_mobility", eval_config.use_mobility),
                ("use_pawn_structure", eval_config.use_pawn_structure),
                ("use_king_safety", eval_config.use_king_safety),
            ]
            enabled_flags: list[str] = []
            for name, enabled in all_flags:
                if enabled:
                    enabled_flags.append(name)

            if enabled_flags:
                formatted_flags = ", ".join(enabled_flags)
                raise ValueError(
                    f"Eval flags [{formatted_flags}] can't be used with mock eval "
                    "Mock evaluator ignores all evaluation settings."
                )

        if eval_config.evaluator_type == "complex" and not any(
            [
                eval_config.use_material,
                eval_config.use_pst,
                eval_config.use_mobility,
                eval_config.use_pawn_structure,
                eval_config.use_king_safety,
            ]
        ):
            raise ValueError(
                "Complex evaluator must have at least one evaluation feature enabled."
            )

    def __str__(self) -> str:
        parts: list[str] = []
        parts.append(f"Depth: {self.search_depth}")

        parts.append(self._format_search_flags())
        parts.append(self._format_evaluation_flags())

        return " | ".join(parts)

    def _format_search_flags(self) -> str:  # noqa: C901, PLR0912
        s_flags: list[str] = []
        cfg = self.search

        if cfg.use_alpha_beta:
            s_flags.append("α-β")  # noqa: RUF001
        else:
            return (
                "Search: [Base Minimax]"  # If Alpha-Beta is off, most else is off too.
            )

        if cfg.use_iddfs:
            s_flags.append("IDDFS")

        # Search Driver
        if cfg.use_mtdf:
            s_flags.append("MTD(f)")
        elif cfg.use_pvs:
            s_flags.append("PVS")
            if cfg.use_aspiration_windows:
                s_flags.append("AspWin")

        # Memory
        if cfg.use_transposition_table:
            tt_flags = "TT"
            if cfg.use_zobrist:
                tt_flags += "/Z"
            if cfg.use_tt_aging:
                tt_flags += "+Age"
            s_flags.append(tt_flags)
            if cfg.use_iid:
                s_flags.append("IID")

        # Ordering
        if cfg.use_move_ordering:
            s_flags.append("MoveOrder")

        # Pruning/Reductions
        if cfg.use_lmr:
            s_flags.append("LMR")
        if cfg.use_null_move_pruning:
            s_flags.append("NMP")
        if cfg.use_futility_pruning:
            s_flags.append("Futility")
        if cfg.use_quiescence_search:
            s_flags.append("QS")

        # Parallelism
        if cfg.use_parallel_search:
            parallel_type = "Parallel"
            if cfg.use_ybwc:
                parallel_type = "YBWC"
            elif cfg.use_dts:
                parallel_type = "DTS"
            elif cfg.use_lazy_smp:
                parallel_type = "LazySMP"
            elif cfg.use_naive_parallel:
                parallel_type = "NaiveParallel"
            s_flags.append(f"{parallel_type}[{cfg.num_threads}T]")

        # External
        if cfg.use_opening_book:
            s_flags.append("Book")
        if cfg.use_endgame_tablebases:
            s_flags.append("EGTB")

        if s_flags:
            formatted = ", ".join(s_flags)
            return f"Search: [{formatted}]"
        return "Search: [Empty]"

    def _format_evaluation_flags(self) -> str:
        eval_parts: list[str] = [self.evaluation.evaluator_type.capitalize()]
        if self.evaluation.evaluator_type == "complex":
            complex_flags: list[str] = []
            if self.evaluation.use_material:
                complex_flags.append("Material")
            if self.evaluation.use_pst:
                complex_flags.append("PST")
            if self.evaluation.use_mobility:
                complex_flags.append("Mobility")
            if self.evaluation.use_pawn_structure:
                complex_flags.append("Pawns")
            if self.evaluation.use_king_safety:
                complex_flags.append("KingSafety")
            if complex_flags:
                eval_parts.append(f"[{', '.join(complex_flags)}]")
        return f"Eval: {' '.join(eval_parts)}"

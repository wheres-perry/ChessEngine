from dataclasses import dataclass, field
from typing import Literal

from src.engine.constants import DEFAULT_DEPTH, DEFAULT_TIMEOUT


@dataclass
class SearchConfig:
    """
    Configuration for the chess engine search algorithm and optimizations.
    Features are grouped by their function based on the dependency graph.
    """

    # --- Time Management and Limits ---
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
    use_lazy_smp: bool = False  # Lazy Symmetric Multiprocessing
    use_ybwc: bool = False  # Young Brothers Wait Concept
    use_dts: bool = False  # Dynamic Tree Splitting
    num_threads: int = 1

    # Alternative Search Drivers (Usually exclusive of standard PVS/ABP)
    use_mtdf: bool = False  # Memory-enhanced Test Driver (MTD-f)


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

    minimax: SearchConfig = field(default_factory=SearchConfig)
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
        if self.search_depth > 20:
            raise ValueError(f"Search depth too high (max 20), got {self.search_depth}")

        # Validate minimax configuration
        self._validate_minimax_config()

        # Validate evaluation configuration
        self._validate_evaluation_config()

    def _validate_minimax_config(self) -> None:
        """Validate minimax-specific configuration."""
        minimax_config = self.minimax

        # Validate timeout
        if minimax_config.max_time is not None and minimax_config.max_time <= 0:
            raise ValueError(
                f"Minimax timeout must be positive, got {minimax_config.max_time}"
            )

        # Validate TT aging requires Zobrist hashing
        if minimax_config.use_tt_aging and not minimax_config.use_zobrist:
            raise ValueError(
                "Transposition table aging requires Zobrist hashing to be enabled"
            )

        # Validate LMR requires both alpha-beta and move ordering (check this first)
        if minimax_config.use_lmr and not (
            minimax_config.use_alpha_beta and minimax_config.use_move_ordering
        ):
            raise ValueError(
                "Late Move Reduction (LMR) requires both alpha-beta pruning and move "
                "ordering to be enabled"
            )

        # Validate PVS requires alpha-beta pruning
        if minimax_config.use_pvs and not minimax_config.use_alpha_beta:
            raise ValueError(
                "Principal Variation Search (PVS) requires alpha-beta "
                "pruning to be enabled"
            )

    def _validate_evaluation_config(self) -> None:
        """Validate evaluation-specific configuration."""
        eval_config = self.evaluation

        # Simplify the conditional for the simple evaluator case.
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

        parts.append(self._format_minimax_flags())
        parts.append(self._format_evaluation_flags())

        return " | ".join(parts)

    def _format_minimax_flags(self) -> str:
        mm_flags: list[str] = []
        if self.minimax.use_zobrist:
            tt_flags = "TT/Zobrist"
            if self.minimax.use_tt_aging:
                tt_flags += "+Aging"
            mm_flags.append(tt_flags)
        if self.minimax.use_iddfs:
            mm_flags.append("IDDFS")
        if self.minimax.use_alpha_beta:
            mm_flags.append("α-β")  # noqa: RUF001
        if self.minimax.use_move_ordering:
            mm_flags.append("MoveOrder")
        if self.minimax.use_pvs:
            mm_flags.append("PVS")
        if self.minimax.use_lmr:
            mm_flags.append("LMR")

        if mm_flags:
            formatted = ", ".join(mm_flags)
            return f"Search: [{formatted}]"
        return "Search: [Base Minimax]"

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

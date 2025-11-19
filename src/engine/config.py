from dataclasses import dataclass, field
from typing import Literal

from src.engine.constants import DEFAULT_DEPTH, DEFAULT_TIMEOUT


@dataclass
class SearchConfig:
    """
    Configuration for the chess engine search algorithm and optimizations.
    Features are grouped by their function based on the dependency graph.

    Tree 1: Move Exploration (Search) Optimizations
    A (Core) -> B (Alpha-Beta) -> G (PVS) -> H (NMP)
    A -> C (ID) -> M (IID, with L)
    A -> D (Move Ordering) -> E (Killer), F (History)
    B & D -> K (LMR), J (Futility)
    A -> L (TT)
    A -> I (Check Extensions)
    B -> N (Quiescence Search)
    """

    # ========================================================================
    # 1. General Search Settings
    # ========================================================================
    max_time: float | None = DEFAULT_TIMEOUT
    max_depth: int | None = None  # Useful for depth-limited searches

    # ========================================================================
    # 1. Core Search Algorithm (Node A - Base)
    # ========================================================================
    use_minimax: bool = (
        True  # Basic Minimax Search (Core Algorithm, dependency for all)
    )

    # ========================================================================
    # 2. Basic Improvements (Direct dependencies on Minimax)
    # ========================================================================
    use_alpha_beta: bool = True  # Alpha-Beta Pruning (Node B, requires A)
    use_iddfs: bool = True  # Iterative Deepening (Node C, requires A)
    use_move_ordering: bool = True  # Move Ordering (Node D, requires A)
    use_transposition_table: bool = True  # Transposition Table (Node L, requires A)
    use_check_extensions: bool = True  # Check Extensions (Node I, requires A)

    # ========================================================================
    # 3. Advanced Search Refinements (Build on Alpha-Beta)
    # ========================================================================
    use_pvs: bool = True  # Principal Variation Search (Node G, requires B + C)
    use_quiescence_search: bool = True  # Quiescence Search (Node N, requires B)

    # ========================================================================
    # 4. Move Ordering Heuristics (Build on Move Ordering)
    # ========================================================================
    # Node E: Killer Heuristic (requires D + B)
    use_killer_moves: bool = True

    # Node F: History Heuristic (requires D + B)
    use_history_heuristic: bool = True
    use_countermove_heuristic: bool = True  # Related to history

    # Hash move ordering (requires D + L)
    use_hash_move_ordering: bool = True

    # Static ordering for captures (requires D)
    use_mvv_lva: bool = True  # Most Valuable Victim / Least Valuable Aggressor
    use_see_ordering: bool = True  # Static Exchange Evaluation for ordering

    # ========================================================================
    # 5. Advanced Pruning and Reductions (Build on Alpha-Beta + Move Ordering)
    # ========================================================================
    # Node K: Late Move Reductions (requires B + D)
    use_lmr: bool = True

    # Node J: Futility Pruning (requires B, uses evaluation)
    use_futility_pruning: bool = True
    use_extended_futility_pruning: bool = True
    use_reverse_futility_pruning: bool = True  # Static Null Move Pruning

    # Node H: Null Move Pruning (requires G or B)
    use_null_move_pruning: bool = True

    # ========================================================================
    # 6. Transposition Table Support (Node L)
    # ========================================================================
    use_zobrist: bool = True  # Hashing method (Required for TT)
    use_tt_aging: bool = True  # TT aging strategy

    # ========================================================================
    # 7. Internal Iterative Deepening (Node M, requires C + L)
    # ========================================================================
    use_iid: bool = True

    # ========================================================================
    # 8. Quiescence Search Extensions (Build on Quiescence)
    # ========================================================================
    use_delta_pruning: bool = True  # Requires quiescence search
    use_see_pruning_in_qs: bool = True  # SEE pruning in Q-Search

    # ========================================================================
    # 9. Additional Search Refinements
    # ========================================================================
    use_aspiration_windows: bool = True  # Requires alpha-beta
    use_recapture_extensions: bool = False
    use_singular_extensions: bool = False

    # ========================================================================
    # 10. Advanced/Experimental Features
    # ========================================================================
    use_probcut: bool = False
    use_multicut_pruning: bool = False
    use_razoring: bool = False

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
    """
    Configuration for the board evaluation.

    Tree 2: State Evaluation Optimizations
    E0 (Material) -> E1 (PST) -> E2 (Tapered), E3 (Pawn), E4 (Mobility), E6 (SEE)
    E4 -> E5 (King Safety)
    E2 -> E9 (Endgame Tables)
    E2 & E3 & E4 & E5 -> E10 (Tuning), E7 (Eval Caching)
    E0 -> E8 (Bitboards)
    """

    evaluator_type: Literal["simple", "mock", "complex"] = "complex"

    # ========================================================================
    # Node E0: Core Material Count
    # ========================================================================
    use_material: bool = True  # Base evaluation (required for all)

    # ========================================================================
    # Node E8: Bitboard Representation (builds on E0)
    # ========================================================================
    use_bitboards: bool = True  # Efficiency improvement

    # ========================================================================
    # Node E1: Piece-Square Tables (requires E0)
    # ========================================================================
    use_pst: bool = True

    # ========================================================================
    # Advanced Evaluation Components (require E1)
    # ========================================================================
    use_tapered_eval: bool = True  # Node E2: Midgame/Endgame blend
    use_pawn_structure: bool = True  # Node E3: Pawn evaluation
    use_mobility: bool = True  # Node E4: Piece mobility
    use_see: bool = True  # Node E6: Static Exchange Evaluation

    # ========================================================================
    # Node E5: King Safety (requires E4)
    # ========================================================================
    use_king_safety: bool = True

    # ========================================================================
    # Deep Evaluation Features (require E2 and others)
    # ========================================================================
    use_endgame_tables: bool = False  # Node E9: Specialized endgame knowledge
    use_eval_tuning: bool = False  # Node E10: Texel tuning or similar
    use_eval_caching: bool = False  # Node E7: Cache evaluations (requires TT)


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
        """Validate configuration for consistency and correctness."""
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

    def _validate_minimax_config(self) -> None:  # noqa: C901, PLR0912
        """Validate minimax-specific configuration (Tree 1 dependencies)."""
        cfg = self.minimax

        # Validate timeout
        if cfg.max_time is not None and cfg.max_time <= 0:
            raise ValueError(f"Minimax timeout must be positive, got {cfg.max_time}")

        # ========================================================================
        # Tree 1: Move Exploration Dependencies Validation
        # ========================================================================

        # Level 1: All features require minimax (Node A)
        if not cfg.use_minimax and any(
            [
                cfg.use_alpha_beta,
                cfg.use_iddfs,
                cfg.use_move_ordering,
                cfg.use_transposition_table,
                cfg.use_check_extensions,
                cfg.use_pvs,
                cfg.use_quiescence_search,
            ]
        ):
            raise ValueError(
                "All search optimizations require basic minimax to be enabled"
            )

        # Node B: Alpha-Beta is required by several features
        if not cfg.use_alpha_beta:
            if cfg.use_pvs:
                raise ValueError(
                    "Principal Variation Search (PVS) requires alpha-beta pruning"
                )
            if cfg.use_quiescence_search:
                raise ValueError("Quiescence search requires alpha-beta pruning")
            if cfg.use_null_move_pruning:
                raise ValueError("Null move pruning requires alpha-beta pruning")
            if cfg.use_aspiration_windows:
                raise ValueError("Aspiration windows require alpha-beta pruning")
            if cfg.use_futility_pruning:
                raise ValueError("Futility pruning requires alpha-beta pruning")
            if cfg.use_extended_futility_pruning:
                raise ValueError(
                    "Extended futility pruning requires alpha-beta pruning"
                )
            if cfg.use_reverse_futility_pruning:
                raise ValueError("Reverse futility pruning requires alpha-beta pruning")

        # Node C: Iterative Deepening dependencies
        if not cfg.use_minimax:
            if cfg.use_alpha_beta:
                raise ValueError("Alpha-beta pruning requires basic minimax")
            if cfg.use_iddfs:
                raise ValueError("Iterative deepening requires basic minimax")
            if cfg.use_move_ordering:
                raise ValueError("Move ordering requires basic minimax")
            if cfg.use_transposition_table:
                raise ValueError("Transposition table requires basic minimax")

        # Node E & F: Killer and History heuristics require move ordering + alpha-beta
        if cfg.use_killer_moves and not (cfg.use_move_ordering and cfg.use_alpha_beta):
            raise ValueError(
                "Killer heuristic requires both move ordering and alpha-beta pruning"
            )
        if cfg.use_history_heuristic and not (
            cfg.use_move_ordering and cfg.use_alpha_beta
        ):
            raise ValueError(
                "History heuristic requires both move ordering and alpha-beta pruning"
            )
        if cfg.use_countermove_heuristic and not (
            cfg.use_move_ordering and cfg.use_alpha_beta
        ):
            raise ValueError(
                "Countermove heuristic requires both move ordering and "
                "alpha-beta pruning"
            )

        # Node G: PVS requires alpha-beta + iterative deepening
        if cfg.use_pvs and not (cfg.use_alpha_beta and cfg.use_iddfs):
            raise ValueError(
                "Principal Variation Search requires both alpha-beta and "
                "iterative deepening"
            )

        # Node K: Late Move Reduction requires alpha-beta + move ordering
        if cfg.use_lmr and not (cfg.use_alpha_beta and cfg.use_move_ordering):
            raise ValueError(
                "Late Move Reduction requires both alpha-beta pruning and move ordering"
            )

        # Node L: Transposition table dependencies
        if cfg.use_transposition_table and not cfg.use_zobrist:
            raise ValueError("Transposition table requires Zobrist hashing")
        if cfg.use_zobrist and not cfg.use_transposition_table:
            raise ValueError(
                "Zobrist hashing is only useful with transposition table enabled"
            )
        if cfg.use_tt_aging and not cfg.use_zobrist:
            raise ValueError("TT aging requires Zobrist hashing")

        # Hash move ordering requires move ordering + TT
        if cfg.use_hash_move_ordering:
            if not cfg.use_move_ordering:
                raise ValueError("Hash move ordering requires move ordering")
            if not cfg.use_transposition_table:
                raise ValueError("Hash move ordering requires transposition table")

        # Node M: Internal Iterative Deepening requires ID + TT
        if cfg.use_iid and not (cfg.use_iddfs and cfg.use_transposition_table):
            raise ValueError(
                "Internal Iterative Deepening requires both IDDFS and "
                "transposition table"
            )

        # Node N: Quiescence search extensions
        if cfg.use_delta_pruning and not cfg.use_quiescence_search:
            raise ValueError("Delta pruning requires quiescence search")
        if cfg.use_see_pruning_in_qs and not cfg.use_quiescence_search:
            raise ValueError("SEE pruning in QS requires quiescence search")

        # Static ordering features require move ordering
        if cfg.use_mvv_lva and not cfg.use_move_ordering:
            raise ValueError("MVV-LVA ordering requires move ordering")
        if cfg.use_see_ordering and not cfg.use_move_ordering:
            raise ValueError("SEE ordering requires move ordering")

    def _validate_evaluation_config(self) -> None:  # noqa: C901, PLR0912
        """Validate evaluation-specific configuration (Tree 2 dependencies)."""
        eval_config = self.evaluation

        # Simplify the conditional for the simple evaluator case.
        complex_flags = [
            ("use_pst", eval_config.use_pst),
            ("use_mobility", eval_config.use_mobility),
            ("use_pawn_structure", eval_config.use_pawn_structure),
            ("use_king_safety", eval_config.use_king_safety),
            ("use_tapered_eval", eval_config.use_tapered_eval),
            ("use_see", eval_config.use_see),
            ("use_bitboards", eval_config.use_bitboards),
        ]

        if eval_config.evaluator_type == "simple":
            flag_list: list[str] = []
            for name, enabled in complex_flags:
                if enabled:
                    flag_list.append(name)

            if flag_list:
                formatted_flags = ", ".join(flag_list)
                raise ValueError(
                    f"Complex evaluation flags [{formatted_flags}] cannot be used "
                    "with simple evaluator. Use 'complex' evaluator type or disable "
                    "these flags."
                )

        if eval_config.evaluator_type == "mock":
            all_flags = [("use_material", eval_config.use_material), *complex_flags]
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

        if eval_config.evaluator_type == "complex" and not eval_config.use_material:
            raise ValueError(
                "Complex evaluator requires material evaluation (base of tree)."
            )

        # Tree 2 dependency validations
        # Node E1 (PST) requires E0 (Material)
        if eval_config.use_pst and not eval_config.use_material:
            raise ValueError("Piece-Square Tables require material evaluation")

        # Nodes E2, E3, E4, E6 require E1 (PST)
        if eval_config.use_tapered_eval and not eval_config.use_pst:
            raise ValueError("Tapered evaluation requires Piece-Square Tables")
        if eval_config.use_pawn_structure and not eval_config.use_pst:
            raise ValueError("Pawn structure evaluation requires Piece-Square Tables")
        if eval_config.use_mobility and not eval_config.use_pst:
            raise ValueError("Mobility evaluation requires Piece-Square Tables")
        if eval_config.use_see and not eval_config.use_pst:
            raise ValueError("Static Exchange Evaluation requires Piece-Square Tables")

        # Node E5 (King Safety) requires E4 (Mobility)
        if eval_config.use_king_safety and not eval_config.use_mobility:
            raise ValueError("King safety evaluation requires mobility evaluation")

        # Node E9 (Endgame Tables) requires E2 (Tapered)
        if eval_config.use_endgame_tables and not eval_config.use_tapered_eval:
            raise ValueError("Endgame tables require tapered evaluation")

        # Node E10 (Tuning) requires E2, E3, E4, E5
        if eval_config.use_eval_tuning and not all(
            [
                eval_config.use_tapered_eval,
                eval_config.use_pawn_structure,
                eval_config.use_mobility,
                eval_config.use_king_safety,
            ]
        ):
            raise ValueError(
                "Evaluation tuning requires tapered eval, pawn structure, "
                "mobility, and king safety"
            )

        # Node E7 (Eval Caching) requires full evaluation and TT
        if eval_config.use_eval_caching and not all(
            [
                eval_config.use_tapered_eval,
                eval_config.use_pawn_structure,
                eval_config.use_mobility,
                eval_config.use_king_safety,
            ]
        ):
            raise ValueError("Evaluation caching requires full evaluation features")

        if eval_config.use_eval_caching and not self.minimax.use_transposition_table:
            raise ValueError("Evaluation caching requires transposition table")

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

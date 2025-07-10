from dataclasses import dataclass, field
from typing import Literal

from src.engine.constants import DEFAULT_DEPTH, DEFAULT_TIMEOUT


@dataclass
class MinimaxConfig:
    """Configuration for the Minimax search algorithm."""

    use_zobrist: bool = True
    use_iddfs: bool = True
    use_alpha_beta: bool = True
    use_move_ordering: bool = True
    use_pvs: bool = True
    use_tt_aging: bool = True
    use_lmr: bool = True
    max_time: float | None = DEFAULT_TIMEOUT


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

    minimax: MinimaxConfig = field(default_factory=MinimaxConfig)
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

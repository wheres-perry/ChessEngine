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
    max_time: float | None = DEFAULT_TIMEOUT


@dataclass
class EvaluationConfig:
    """Configuration for the board evaluation."""

    evaluator_type: Literal["simple", "mock", "complex"] = "complex"

    # --- Flags for the Complex Evaluator ---

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

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate the configuration for consistency and correctness."""
        # Validate search depth

        if self.search_depth < 1:
            raise ValueError(
                f"Search depth must be at least 1, got {self.search_depth}"
            )
        if self.search_depth > 20:
            raise ValueError(f"Search depth too high (max 20), got {self.search_depth}")
        # Validate minimax timeout

        if self.minimax.max_time is not None and self.minimax.max_time <= 0:
            raise ValueError(
                f"Minimax timeout must be positive, got {self.minimax.max_time}"
            )
        # Validate TT aging is only used with Zobrist

        if self.minimax.use_tt_aging and not self.minimax.use_zobrist:
            raise ValueError(
                "Transposition table aging requires Zobrist hashing to be enabled"
            )
        # Validate evaluation configuration

        self._validate_evaluation_config()

    def _validate_evaluation_config(self):
        """Validate evaluation-specific configuration."""
        eval_config = self.evaluation

        # Check if complex evaluation flags are used with simple evaluator

        if eval_config.evaluator_type == "simple":
            complex_flags = [
                ("use_pst", eval_config.use_pst),
                ("use_mobility", eval_config.use_mobility),
                ("use_pawn_structure", eval_config.use_pawn_structure),
                ("use_king_safety", eval_config.use_king_safety),
            ]

            enabled_complex_flags = [name for name, enabled in complex_flags if enabled]

            if enabled_complex_flags:
                flag_list = ", ".join(enabled_complex_flags)
                raise ValueError(
                    f"Complex evaluation flags [{flag_list}] cannot be used with "
                    f"simple evaluator. Use 'complex' evaluator type or disable these flags."
                )
        # Check if mock evaluator has any evaluation flags enabled

        if eval_config.evaluator_type == "mock":
            all_flags = [
                ("use_material", eval_config.use_material),
                ("use_pst", eval_config.use_pst),
                ("use_mobility", eval_config.use_mobility),
                ("use_pawn_structure", eval_config.use_pawn_structure),
                ("use_king_safety", eval_config.use_king_safety),
            ]

            enabled_flags = [name for name, enabled in all_flags if enabled]

            if enabled_flags:
                flag_list = ", ".join(enabled_flags)
                raise ValueError(
                    f"Evaluation flags [{flag_list}] cannot be used with "
                    f"mock evaluator. Mock evaluator ignores all evaluation settings."
                )
        # Validate that complex evaluator has at least one feature enabled

        if eval_config.evaluator_type == "complex":
            if not any(
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
        parts = []
        parts.append(f"Depth: {self.search_depth}")

        # Minimax flags

        mm_flags = []
        if self.minimax.use_zobrist:
            tt_flags = "TT/Zobrist"
            if self.minimax.use_tt_aging:
                tt_flags += "+Aging"
            mm_flags.append(tt_flags)
        if self.minimax.use_iddfs:
            mm_flags.append("IDDFS")
        if self.minimax.use_alpha_beta:
            mm_flags.append("α-β")
        if self.minimax.use_move_ordering:
            mm_flags.append("MoveOrder")
        if self.minimax.use_pvs:
            mm_flags.append("PVS")
        if mm_flags:
            parts.append(f"Search: [{', '.join(mm_flags)}]")
        else:
            parts.append("Search: [Base Minimax]")
        # Evaluation flags

        eval_parts = [self.evaluation.evaluator_type.capitalize()]
        if self.evaluation.evaluator_type == "complex":
            complex_flags = []
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
        parts.append(f"Eval: {' '.join(eval_parts)}")

        return " | ".join(parts)

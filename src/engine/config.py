from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MinimaxConfig:
    use_zobrist: bool = True
    use_iddfs: bool = True
    use_alpha_beta: bool = True
    use_move_ordering: bool = True
    use_pvs: bool = True
    max_time: float | None = None  # Changed from Optional[float]
    # Future optimizations (not yet implemented)

    use_null_move_pruning: bool = False
    use_late_move_reductions: bool = False
    use_futility_pruning: bool = False
    use_delta_pruning: bool = False
    use_aspiration_windows: bool = False


@dataclass
class EvaluationConfig:
    evaluator_type: Literal["simple", "nn", "deep_cnn"] = "simple"
    # Add other evaluation-specific flags here, e.g.:
    # nn_model_path: str | None = None


@dataclass
class EngineConfig:
    minimax: MinimaxConfig = field(default_factory=MinimaxConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    search_depth: int = 4  # Default search depth
    # Add other engine-wide settings if needed

    def __str__(self) -> str:
        parts = []
        parts.append(f"Depth: {self.search_depth}")

        mm_flags = []
        if self.minimax.use_zobrist:
            mm_flags.append("TT/Zobrist")
        if self.minimax.use_iddfs:
            mm_flags.append("IDDFS")
        if self.minimax.use_alpha_beta:
            mm_flags.append("α-β")
        if self.minimax.use_move_ordering:
            mm_flags.append("MoveOrder")
        if self.minimax.use_pvs:
            mm_flags.append("PVS")
        if self.minimax.use_null_move_pruning:
            mm_flags.append("NMP")
        if self.minimax.use_late_move_reductions:
            mm_flags.append("LMR")
        if self.minimax.use_futility_pruning:
            mm_flags.append("Futility")
        if self.minimax.use_delta_pruning:
            mm_flags.append("Delta")
        if self.minimax.use_aspiration_windows:
            mm_flags.append("AspWin")
        if mm_flags:
            parts.append(f"Search: [{', '.join(mm_flags)}]")
        else:
            parts.append("Search: [Base Minimax]")
        parts.append(f"Eval: {self.evaluation.evaluator_type.capitalize()}")

        return " | ".join(parts)

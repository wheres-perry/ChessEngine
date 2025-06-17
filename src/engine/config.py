from dataclasses import dataclass, field
from typing import Literal

from src.engine.constants import DEFAULT_DEPTH, DEFAULT_TIMEOUT


@dataclass
class MinimaxConfig:
    use_zobrist: bool = True
    use_iddfs: bool = True
    use_alpha_beta: bool = True
    use_move_ordering: bool = True
    use_pvs: bool = True
    max_time: float | None = DEFAULT_TIMEOUT


@dataclass
class EvaluationConfig:
    evaluator_type: Literal["simple", "mock"] = "simple"


@dataclass
class EngineConfig:
    minimax: MinimaxConfig = field(default_factory=MinimaxConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    search_depth: int = DEFAULT_DEPTH

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
        if mm_flags:
            parts.append(f"Search: [{', '.join(mm_flags)}]")
        else:
            parts.append("Search: [Base Minimax]")
        parts.append(f"Eval: {self.evaluation.evaluator_type.capitalize()}")

        return " | ".join(parts)

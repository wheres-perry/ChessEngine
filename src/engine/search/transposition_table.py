"""Transposition table implementation used by negamax search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from engine._core import chess_engine_core as chess
    from engine.config import SearchConfig

BoundType = Literal["exact", "lower", "upper"]


@dataclass(slots=True)
class TTEntry:
    """Single transposition table entry."""

    key: int
    depth: int
    score: float
    best_move: chess.Move | None
    bound: BoundType
    age: int


class TranspositionTable:
    """In-memory transposition table with lightweight aging and replacement."""

    ESTIMATED_ENTRY_SIZE_BYTES = 32

    def __init__(self, config: SearchConfig):
        self.config = config
        estimated_capacity = (
            config.tt_size_mb * 1024 * 1024 // self.ESTIMATED_ENTRY_SIZE_BYTES
        )
        self.max_entries = max(1024, int(estimated_capacity))
        self.table: dict[int, TTEntry] = {}
        self.current_age = 0

    def increment_age(self) -> None:
        """Increment search age once before each new top-level search."""
        if self.config.use_tt_aging:
            self.current_age += 1

    def clear(self) -> None:
        self.table.clear()

    def size(self) -> int:
        return len(self.table)

    def probe(self, key: int) -> TTEntry | None:
        """Probe table and optionally refresh entry age on hit."""
        entry = self.table.get(key)
        if entry is None:
            return None
        if self.config.use_tt_aging:
            entry.age = self.current_age
        return entry

    def try_get_score(
        self,
        entry: TTEntry,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float | None:
        """Return a usable score from an entry if it can cut off at this node."""
        if entry.depth < depth:
            return None
        if entry.bound == "exact":
            return entry.score
        if entry.bound == "lower" and entry.score >= beta:
            return entry.score
        if entry.bound == "upper" and entry.score <= alpha:
            return entry.score
        return None

    def store(
        self,
        key: int,
        depth: int,
        score: float,
        best_move: chess.Move | None,
        bound: BoundType,
    ) -> None:
        """Store a search result using depth-preferred replacement."""
        existing = self.table.get(key)
        if existing is not None:
            if self.config.use_tt_aging:
                if existing.age == self.current_age and existing.depth > depth:
                    return
            elif existing.depth > depth:
                return

        if key not in self.table and len(self.table) >= self.max_entries:
            oldest_key = min(self.table, key=lambda hash_key: self.table[hash_key].age)
            del self.table[oldest_key]

        self.table[key] = TTEntry(
            key=key,
            depth=depth,
            score=score,
            best_move=best_move,
            bound=bound,
            age=self.current_age,
        )

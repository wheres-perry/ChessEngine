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

    def __init__(self, config: SearchConfig) -> None:
        """Initialize the transposition table.

        Args:
            config: The search configuration containing transposition table settings,
                including tt_size_mb for table size and use_tt_aging for aging policy.

        """
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
        """Clear all entries from the transposition table.

        Removes all stored entries while maintaining the current configuration.
        """
        self.table.clear()

    def size(self) -> int:
        """Return the number of entries currently stored in the table.

        Returns:
            The count of transposition table entries.

        """
        return len(self.table)

    def probe(self, key: int) -> TTEntry | None:
        """Probe the table for an entry and optionally refresh its age.

        Args:
            key: The hash key to look up in the table.

        Returns:
            The matching TTEntry if found, with its age updated if TT aging is enabled;
            None if no entry exists for the given key.

        """
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
        """Return a usable score from an entry if it can cut off at this node.

        Checks if the stored entry has sufficient depth and appropriate bound type
        to provide a valid score for the current search window.

        Args:
            entry: The transposition table entry to evaluate.
            depth: The current search depth required.
            alpha: The current alpha bound for the search window.
            beta: The current beta bound for the search window.

        Returns:
            The stored score if the entry is usable for cutoff at this node;
            None if the entry cannot be used (insufficient depth or bounds mismatch).

        """
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
        """Store a search result using depth-preferred replacement.

        Stores a new entry or replaces an existing one based on depth and age.
        When the table is full, removes the oldest entry to make space.

        Args:
            key: The hash key for this position.
            depth: The search depth at which this position was evaluated.
            score: The evaluated score for this position.
            best_move: The best move found from this position, if any.
            bound: The bound type ("exact", "lower", or "upper") indicating
                how the score relates to the search window.

        """
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

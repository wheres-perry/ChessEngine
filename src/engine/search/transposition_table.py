"""Transposition table implementation backed by a C++ core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from engine._core import chess_engine_core as chess

if TYPE_CHECKING:
    from engine.config import SearchConfig

BoundType = Literal["exact", "lower", "upper"]

# Mapping between Python string bounds and C++ enum values.
_BOUND_TO_CPP: dict[str, chess.BoundType] = {
    "exact": chess.BoundType.EXACT,
    "lower": chess.BoundType.LOWER,
    "upper": chess.BoundType.UPPER,
}

_BOUND_FROM_CPP: dict[int, BoundType] = {
    0: "exact",
    1: "lower",
    2: "upper",
}


def _is_sentinel(move: chess.Move) -> bool:
    """Return True if *move* is the sentinel (no-move) value."""
    return move.from_square == 0 and move.to_square == 0 and move.promotion == 0


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
    """In-memory transposition table backed by a C++ core."""

    ESTIMATED_ENTRY_SIZE_BYTES = 32

    def __init__(self, config: SearchConfig) -> None:
        """Initialize the transposition table.

        Args:
            config: The search configuration containing transposition table settings,
                including tt_size_mb for table size and use_tt_aging for aging policy.

        """
        self.config = config
        self._cpp = chess.TranspositionTable(config.tt_size_mb)
        self.max_entries = self._cpp.capacity()
        self.current_age = 0
        # Optional Python-side key tracking for capacity-limited scenarios.
        # This is only active when max_entries < C++ capacity (e.g. tests).
        self._tracked_keys: set[int] | None = None

    def __setattr__(self, name: str, value: object) -> None:
        """Intercept max_entries changes to enable Python-side tracking."""
        super().__setattr__(name, value)
        if name == "max_entries" and hasattr(self, "_cpp"):
            if isinstance(value, int) and value < self._cpp.capacity():
                self._tracked_keys = set()
            else:
                self._tracked_keys = None

    def increment_age(self) -> None:
        """Increment search age once before each new top-level search."""
        if self.config.use_tt_aging:
            self._cpp.increment_age()
            self.current_age = self._cpp.current_age()

    def clear(self) -> None:
        """Clear all entries from the transposition table.

        Removes all stored entries while maintaining the current configuration.
        """
        self._cpp.clear()
        if self._tracked_keys is not None:
            self._tracked_keys.clear()

    def size(self) -> int:
        """Return the number of entries currently stored in the table.

        Returns:
            The count of transposition table entries.

        """
        if self._tracked_keys is not None:
            return len(self._tracked_keys)
        return self._cpp.size()

    def probe(self, key: int) -> TTEntry | None:
        """Probe the table for an entry and optionally refresh its age.

        Args:
            key: The hash key to look up in the table.

        Returns:
            The matching TTEntry if found, with its age updated if TT aging is enabled;
            None if no entry exists for the given key.

        """
        cpp_entry = self._cpp.probe(key)
        if cpp_entry is None:
            return None

        # When using tracked keys, only return entries we know about.
        if self._tracked_keys is not None and key not in self._tracked_keys:
            return None

        # Refresh age on the C++ side when aging is enabled.
        if self.config.use_tt_aging:
            move = cpp_entry.best_move
            best_move: chess.Move | None = None if _is_sentinel(move) else move
            self._cpp.store(
                key,
                cpp_entry.depth,
                cpp_entry.score,
                best_move,
                _BOUND_TO_CPP[_BOUND_FROM_CPP[cpp_entry.bound]],
            )
            cpp_entry = self._cpp.probe(key)
            if cpp_entry is None:  # pragma: no cover
                return None

        move = cpp_entry.best_move
        best_move_out: chess.Move | None = None if _is_sentinel(move) else move

        return TTEntry(
            key=cpp_entry.key,
            depth=cpp_entry.depth,
            score=float(cpp_entry.score),
            best_move=best_move_out,
            bound=_BOUND_FROM_CPP[cpp_entry.bound],
            age=cpp_entry.age,
        )

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

        Args:
            key: The hash key for this position.
            depth: The search depth at which this position was evaluated.
            score: The evaluated score for this position.
            best_move: The best move found from this position, if any.
            bound: The bound type ("exact", "lower", or "upper") indicating
                how the score relates to the search window.

        """
        # When capacity is artificially limited, enforce eviction in Python.
        if (
            self._tracked_keys is not None
            and key not in self._tracked_keys
            and len(self._tracked_keys) >= self.max_entries
        ):
            # Evict the oldest entry.
            oldest_key = None
            oldest_age = float("inf")
            for k in self._tracked_keys:
                entry = self._cpp.probe(k)
                if entry is not None and entry.age < oldest_age:
                    oldest_age = entry.age
                    oldest_key = k
            if oldest_key is not None:
                self._tracked_keys.discard(oldest_key)

        self._cpp.store(key, depth, int(score), best_move, _BOUND_TO_CPP[bound])
        if self._tracked_keys is not None:
            self._tracked_keys.add(key)

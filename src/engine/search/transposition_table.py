from typing import Literal, TypedDict
import chess

class TTEntry(TypedDict):
    depth: int
    score: float
    type: Literal["upper", "lower", "exact"]
    age: int
    best_move: chess.Move | None  # FIXED: Added best move field

class TranspositionTable:
    """
    Transposition table for storing previously evaluated positions.
    Uses Zobrist hashing to store position evaluations with depth and bound information
    to avoid re-evaluating identical positions during search.
    """

    DEFAULT_SIZE = 100000
    MAX_AGE_DIFF = 2  # Maximum age difference to consider entry valid

    def __init__(self, max_entries: int = DEFAULT_SIZE, use_tt_aging: bool = True):
        """
        Initialize the transposition table.

        Args:
            max_entries: Maximum number of entries to store
            use_tt_aging: Whether to use transposition table aging
        """
        self.table: dict[int, TTEntry] = {}
        self.max_entries = max_entries
        self.current_age = 0
        self.use_tt_aging = use_tt_aging

    def increment_age(self) -> None:
        """Increment the current age, typically called at the start of a new search."""
        if self.use_tt_aging:
            self.current_age += 1

    def lookup(
        self, hash_val: int, depth: int, alpha: float, beta: float
    ) -> None | float:
        """
        Look up a position in the transposition table.

        Args:
            hash_val: Zobrist hash of the position
            depth: Current search depth
            alpha: Current alpha value
            beta: Current beta value

        Returns:
            Stored score if usable, None otherwise
        """
        entry = self.table.get(hash_val)

        # Return None if entry doesn't exist or is too shallow
        if not entry or entry["depth"] < depth:
            return None

        # Check if entry is too old if aging is enabled
        if self.use_tt_aging:
            # Handle both normal aging and the case when age was reset
            if self.current_age < entry["age"] or (
                entry["age"] < self.current_age - self.MAX_AGE_DIFF
            ):
                return None

        entry_type = entry["type"]
        score = entry["score"]

        if entry_type == "exact":
            return score
        elif entry_type == "lower" and score >= beta:
            return beta
        elif entry_type == "upper" and score <= alpha:
            return alpha

        return None

    def get_best_move(self, hash_val: int) -> chess.Move | None:
        """
        Get the best move for a position if available.
        
        Args:
            hash_val: Zobrist hash of the position
            
        Returns:
            Best move if available and not too old, None otherwise
        """
        entry = self.table.get(hash_val)
        if not entry:
            return None
            
        # Check if entry is too old if aging is enabled
        if self.use_tt_aging:
            if self.current_age < entry["age"] or (
                entry["age"] < self.current_age - self.MAX_AGE_DIFF
            ):
                return None
                
        return entry["best_move"]

    def store(
        self,
        hash_val: int,
        depth: int,
        score: float,
        alpha: float,
        beta: float,
        original_alpha: float,
        best_move: chess.Move | None = None,  # FIXED: Added best move parameter
    ) -> None:
        """
        Store a position evaluation in the transposition table.

        Args:
            hash_val: Zobrist hash of the position
            depth: Search depth for this entry
            score: Evaluation score
            alpha: Current alpha value
            beta: Current beta value
            original_alpha: Alpha value at start of search
            best_move: Best move found for this position
        """
        # Determine entry type based on alpha-beta bounds
        if score <= original_alpha:
            entry_type: Literal["upper", "lower", "exact"] = (
                "upper"  # Upper bound (fail-low)
            )
        elif score >= beta:
            entry_type = "lower"  # Lower bound (fail-high)
        else:
            entry_type = "exact"  # Exact value

        # Eviction: If the table is full and we're adding a new entry, remove one.
        if len(self.table) >= self.max_entries and hash_val not in self.table:
            # Simple FIFO eviction for Python 3.7+
            key_to_evict = next(iter(self.table))
            del self.table[key_to_evict]

        existing_entry = self.table.get(hash_val)

        # Now store the new entry if it's better than existing or no existing entry
        if (
            not existing_entry
            or existing_entry["depth"] <= depth
            or (
                self.use_tt_aging
                and self.current_age - existing_entry["age"] > self.MAX_AGE_DIFF
            )
        ):
            self.table[hash_val] = {
                "depth": depth,
                "score": score,
                "type": entry_type,
                "age": self.current_age,
                "best_move": best_move,  # FIXED: Store the best move
            }

    def clear(self) -> None:
        """Clear all entries from the transposition table."""
        self.table.clear()

    def reset_age(self) -> None:
        """
        Invalidate all existing entries by clearing the table
        and reset age counter to zero.
        """
        self.current_age = 0
        if self.use_tt_aging:
            self.table.clear()

    def size(self) -> int:
        """Get the current number of entries in the table."""
        return len(self.table)
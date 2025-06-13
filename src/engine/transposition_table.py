from typing import Dict, Literal, Optional


class TranspositionTable:
    """
    Transposition table for storing previously evaluated positions.

    Uses Zobrist hashing to store position evaluations with depth and bound information
    to avoid re-evaluating identical positions during search.
    """

    DEFAULT_SIZE = 100000

    def __init__(self, max_entries: int = DEFAULT_SIZE):
        """
        Initialize the transposition table.

        Args:
            max_entries: Maximum number of entries to store
        """
        self.table: Dict[int, Dict] = {}
        self.max_entries = max_entries

    def lookup(
        self, hash_val: int, depth: int, alpha: float, beta: float
    ) -> Optional[float]:
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
        if not entry or entry["depth"] < depth:
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

    def store(
        self,
        hash_val: int,
        depth: int,
        score: float,
        alpha: float,
        beta: float,
        original_alpha: float,
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
        """
        # Determine entry type based on alpha-beta bounds

        if score <= original_alpha:
            entry_type = "upper"  # Upper bound (fail-low)
        elif score >= beta:
            entry_type = "lower"  # Lower bound (fail-high)
        else:
            entry_type = "exact"  # Exact value
        existing_entry = self.table.get(hash_val)

        # Store if we have space or this is a deeper/better entry

        if (
            len(self.table) < self.max_entries
            or not existing_entry
            or existing_entry["depth"] <= depth
        ):

            self.table[hash_val] = {
                "depth": depth,
                "score": score,
                "type": entry_type,
            }

    def clear(self) -> None:
        """Clear all entries from the transposition table."""
        self.table.clear()

    def size(self) -> int:
        """Get the current number of entries in the table."""
        return len(self.table)

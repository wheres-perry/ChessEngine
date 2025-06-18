from typing import Literal, TypedDict


class TTEntry(TypedDict):
    depth: int
    score: float
    type: Literal["upper", "lower", "exact"]
    age: int


class TranspositionTable:
    """
    Transposition table for storing previously evaluated positions.

    Uses Zobrist hashing to store position evaluations with depth and bound information
    to avoid re-evaluating identical positions during search.
    """

    DEFAULT_SIZE = 100000

    def __init__(self, max_entries: int = DEFAULT_SIZE, use_aging: bool = True):
        """
        Initialize the transposition table.

        Args:
            max_entries: Maximum number of entries to store
            use_aging: Whether to use aging for entry replacement
        """
        self.table: dict[int, TTEntry] = {}
        self.max_entries = max_entries
        self.use_aging = use_aging
        self.current_age = 0

    def new_search(self) -> None:
        """Increment age counter for a new search iteration."""
        if self.use_aging:
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
            entry_type: Literal["upper", "lower", "exact"] = (
                "upper"  # Upper bound (fail-low)
            )
        elif score >= beta:
            entry_type = "lower"  # Lower bound (fail-high)
        else:
            entry_type = "exact"  # Exact value
        existing_entry = self.table.get(hash_val)

        # Determine if we should store this entry
        should_store = False

        if len(self.table) < self.max_entries:
            # Table not full, always store
            should_store = True
        elif not existing_entry:
            # No existing entry for this hash
            should_store = True
        elif existing_entry["depth"] <= depth:
            # New entry has equal or greater depth
            should_store = True
        elif self.use_aging:
            # Use aging-based replacement
            age_diff = self.current_age - existing_entry["age"]
            depth_diff = existing_entry["depth"] - depth
            # Replace if entry is old enough relative to depth advantage
            should_store = age_diff >= depth_diff

        if should_store:
            self.table[hash_val] = {
                "depth": depth,
                "score": score,
                "type": entry_type,
                "age": self.current_age,
            }

    def clear(self) -> None:
        """Clear all entries from the transposition table."""
        self.table.clear()
        self.current_age = 0

    def size(self) -> int:
        """Get the current number of entries in the table."""
        return len(self.table)

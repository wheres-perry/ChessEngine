"""Search statistics collected during a negamax search."""

from dataclasses import dataclass, fields


@dataclass
class SearchStats:
    """Statistics collected during a search."""

    nodes: int = 0
    depth: int = 0
    seldepth: int = 0
    tt_hits: int = 0
    hashfull: int = 0
    beta_cutoffs: int = 0
    first_move_cuts: int = 0
    killer_cuts: int = 0
    history_cuts: int = 0
    qsearch_nodes: int = 0
    null_move_cuts: int = 0
    pvs_researches: int = 0
    lmr_researches: int = 0
    qs_see_pruning: int = 0
    qs_delta_pruning: int = 0
    check_extensions: int = 0
    iid_searches: int = 0
    root_move_changes: int = 0
    history_saturation: float = 0.0
    score: int = 0

    def reset(self) -> None:
        """Reset all counters to their default zero state."""
        defaults = type(self)()
        for field in fields(self):
            setattr(self, field.name, getattr(defaults, field.name))

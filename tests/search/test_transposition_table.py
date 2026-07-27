"""Tests for the transposition table.

This module contains unit tests for the TranspositionTable class,
including store/probe operations, bounds checking, depth-preferred
replacement, aging, and capacity eviction.
"""

from __future__ import annotations

from engine._core import moray_core as chess
from engine.config import SearchConfig
from engine.search.transposition_table import TranspositionTable


def _cfg(**overrides: object) -> SearchConfig:
    cfg = SearchConfig(use_transposition_table=True, use_tt_aging=True, tt_size_mb=1)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_store_and_probe_exact_entry() -> None:
    """Verify storing and probing an exact entry returns the stored move."""
    tt = TranspositionTable(_cfg())
    move = chess.Move.from_uci("e2e4")

    tt.store(key=123, depth=4, score=25.0, best_move=move, bound="exact")
    entry = tt.probe(123)

    assert entry is not None
    assert entry.best_move == move
    assert tt.try_get_score(entry, depth=4, alpha=-100, beta=100) == 25.0


def test_lower_bound_cutoff_logic() -> None:
    """Verify lower bound entries trigger cutoffs when appropriate."""
    tt = TranspositionTable(_cfg())
    tt.store(key=1, depth=5, score=80.0, best_move=None, bound="lower")

    entry = tt.probe(1)
    assert entry is not None

    assert tt.try_get_score(entry, depth=4, alpha=-10, beta=50) == 80.0
    assert tt.try_get_score(entry, depth=4, alpha=-10, beta=90) is None


def test_upper_bound_cutoff_logic() -> None:
    """Verify upper bound entries trigger cutoffs when appropriate."""
    tt = TranspositionTable(_cfg())
    tt.store(key=2, depth=5, score=-50.0, best_move=None, bound="upper")

    entry = tt.probe(2)
    assert entry is not None

    assert tt.try_get_score(entry, depth=4, alpha=-20, beta=20) == -50.0
    assert tt.try_get_score(entry, depth=4, alpha=-80, beta=20) is None


def test_depth_preferred_replacement() -> None:
    """Verify deeper entries are preferred over shallower ones."""
    tt = TranspositionTable(_cfg(use_tt_aging=False))

    tt.store(key=9, depth=6, score=12.0, best_move=None, bound="exact")
    tt.store(key=9, depth=3, score=33.0, best_move=None, bound="exact")

    entry = tt.probe(9)
    assert entry is not None
    assert entry.depth == 6
    assert entry.score == 12.0


def test_probe_refreshes_age_when_enabled() -> None:
    """Verify probing refreshes the entry age when aging is enabled."""
    tt = TranspositionTable(_cfg(use_tt_aging=True))
    tt.increment_age()  # age = 1
    tt.store(key=5, depth=2, score=10.0, best_move=None, bound="exact")

    tt.increment_age()  # age = 2
    entry = tt.probe(5)

    assert entry is not None
    assert entry.age == 2


def test_capacity_eviction_happens_when_full() -> None:
    """Verify eviction occurs when the table reaches capacity."""
    tt = TranspositionTable(_cfg())
    tt.max_entries = 2

    tt.store(key=100, depth=2, score=1.0, best_move=None, bound="exact")
    tt.store(key=200, depth=2, score=2.0, best_move=None, bound="exact")

    tt.increment_age()
    tt.store(key=300, depth=2, score=3.0, best_move=None, bound="exact")

    assert tt.size() == 2
    assert tt.probe(300) is not None

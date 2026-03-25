"""Tests to verify the C++ branchless template metaprogramming search module.

These tests ensure that the C++ Pybind11 bindings for the search module meet the
requirements for branchless template generation, configuration handling, and API
compatibility.
"""

from __future__ import annotations

import pytest

from engine._core import chess_engine_core as chess
from engine.config import EngineConfig, SearchConfig
from engine.evaluators import MockEvaluator
# Assuming the C++ implementation is integrated as a drop-in replacement in minimax.py
# or accessible via a factory. We test the Minimax class which should be wrapping
# or aliasing the C++ implementation.
from engine.search.minimax import Minimax


def get_base_minimax_config() -> SearchConfig:
    """Return a completely minimal, 'minimax only' configuration."""
    return SearchConfig(
        use_move_ordering=False,
        use_mvv_lva=False,
        use_history_heuristic=False,
        use_countermove_heuristic=False,
        use_see_ordering=False,
        use_killer_moves=False,
        use_hash_move_ordering=False,
        use_alpha_beta=False,
        use_pvs=False,
        use_quiescence_search=False,
        use_iid=False,
        use_null_move_pruning=False,
        use_lmr=False,
        use_futility_pruning=False,
        use_extended_futility_pruning=False,
        use_reverse_futility_pruning=False,
        use_delta_pruning=False,
        use_see_pruning_in_qs=False,
        use_aspiration_windows=False,
        use_check_extensions=False,
        use_transposition_table=False,
        use_tt_aging=False,
        max_time=None,
    )


def test_base_minimax_only_configuration():
    """Verify that a pure base configuration (all flags False) instantiates and runs.
    
    This ensures that the template metaprogramming can successfully compile and 
    instantiate the absolute minimal version of the search algorithm without 
    pruning, ordering, or transposition tables.
    """
    config = EngineConfig(search=get_base_minimax_config())
    board = chess.Board()
    evaluator = MockEvaluator()
    
    # Should not raise any instantiation errors
    engine = Minimax(board, evaluator, config)
    
    # Verify the API remains unchanged and functional
    score, move = engine.find_best_move(depth=1)
    
    assert score is not None
    assert move is not None
    assert move in list(board.generate_legal_moves())


def test_python_api_unchanged():
    """Verify that the Search API exactly matches the expected interface."""
    config = EngineConfig(search=get_base_minimax_config())
    board = chess.Board()
    evaluator = MockEvaluator()
    
    engine = Minimax(board, evaluator, config)
    
    # Ensure standard methods exist
    assert hasattr(engine, "find_best_move")
    assert callable(engine.find_best_move)
    
    # Method signature and return tuple should match
    result = engine.find_best_move(depth=1)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_object_does_not_check_config_after_instantiation():
    """Verify that the engine state is baked in at instantiation time.
    
    The branchless template design means that configuration flags are evaluated
    once during object construction (to pick the right template specialization).
    Mutating the Python config object afterwards should have no effect on the 
    engine's internal behavior.
    """
    search_config = get_base_minimax_config()
    search_config.use_alpha_beta = False
    config = EngineConfig(search=search_config)
    board = chess.Board()
    evaluator = MockEvaluator()
    
    engine = Minimax(board, evaluator, config)
    
    # Mutate the configuration object post-instantiation
    config.search.use_alpha_beta = True
    config.search.use_move_ordering = True
    config.search.use_transposition_table = True
    
    # If the engine was checking the python config at runtime, this might crash
    # due to missing dependencies (e.g., TT initialized without tables).
    # Since it's branchless and compiled for the initial config, it should run 
    # as base minimax.
    score, move = engine.find_best_move(depth=1)
    assert move is not None


@pytest.mark.parametrize("flag_to_enable", [
    "use_move_ordering",
    "use_alpha_beta",
    "use_quiescence_search",
    "use_check_extensions",
    "use_null_move_pruning",
    "use_futility_pruning",
    "use_extended_futility_pruning",
    "use_reverse_futility_pruning",
    "use_delta_pruning",
    "use_see_pruning_in_qs",
    "use_aspiration_windows",
    "use_lmr",
])
def test_individual_flags_can_be_instantiated(flag_to_enable):
    """Verify that each major standalone flag can be instantiated successfully.
    
    This tests that the C++ template specialization covers all these individual 
    feature toggles without compilation or binding errors.
    """
    search_config = get_base_minimax_config()
    setattr(search_config, flag_to_enable, True)
    
    # Handle implicit dependencies for valid config initialization
    if flag_to_enable in (
        "use_pvs", "use_killer_moves", "use_aspiration_windows"
    ):
        search_config.use_alpha_beta = True
    if flag_to_enable == "use_killer_moves":
        search_config.use_move_ordering = True
        search_config.use_alpha_beta = True
    if flag_to_enable == "use_tt_aging":
        search_config.use_transposition_table = True
        
    config = EngineConfig(search=search_config)
    board = chess.Board()
    
    # Instantiation should succeed, picking the correct template
    engine = Minimax(board, MockEvaluator(), config)
    assert engine is not None


def test_complex_dependent_flags():
    """Verify instantiation of dependent flags that require others to be True."""
    search_config = get_base_minimax_config()
    
    # Enable PVS (requires Alpha-Beta)
    search_config.use_alpha_beta = True
    search_config.use_pvs = True
    
    # Enable TT and Aging (Aging requires TT)
    search_config.use_transposition_table = True
    search_config.use_tt_aging = True
    
    # Enable Move ordering and Killer Moves
    search_config.use_move_ordering = True
    search_config.use_killer_moves = True
    
    # Enable advanced ordering
    search_config.use_mvv_lva = True
    search_config.use_history_heuristic = True
    search_config.use_countermove_heuristic = True
    search_config.use_see_ordering = True
    search_config.use_hash_move_ordering = True
    
    # Enable IID
    search_config.use_iid = True
    
    config = EngineConfig(search=search_config)
    board = chess.Board()
    
    engine = Minimax(board, MockEvaluator(), config)
    score, move = engine.find_best_move(depth=1)
    
    assert move is not None


def test_all_flags_enabled():
    """Verify that the fully featured configuration instantiates correctly.
    
    This ensures that the 'everything on' template specialization compiles
    and integrates via Pybind11.
    """
    search_config = SearchConfig(
        use_move_ordering=True,
        use_mvv_lva=True,
        use_history_heuristic=True,
        use_countermove_heuristic=True,
        use_see_ordering=True,
        use_killer_moves=True,
        use_hash_move_ordering=True,
        use_alpha_beta=True,
        use_pvs=True,
        use_quiescence_search=True,
        use_iid=True,
        use_null_move_pruning=True,
        use_lmr=True,
        use_futility_pruning=True,
        use_extended_futility_pruning=True,
        use_reverse_futility_pruning=True,
        use_delta_pruning=True,
        use_see_pruning_in_qs=True,
        use_aspiration_windows=True,
        use_check_extensions=True,
        use_transposition_table=True,
        use_tt_aging=True,
        max_time=None,
    )
    
    config = EngineConfig(search=search_config)
    board = chess.Board()
    
    engine = Minimax(board, MockEvaluator(), config)
    score, move = engine.find_best_move(depth=1)
    
    assert move is not None

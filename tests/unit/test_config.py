import json
import os
from pathlib import Path

import pytest

from engine.config import EngineConfig, EvaluationConfig, SearchConfig
from engine.config_solver import ConfigSolver, ConfigSolverError


def test_engine_config_to_dict():
    config = EngineConfig(search_depth=10)
    config_dict = config.to_dict()
    assert config_dict["search_depth"] == 10
    assert "search" in config_dict
    assert "evaluation" in config_dict


def test_engine_config_save_load_json(tmp_path: Path):
    config = EngineConfig(search_depth=15)
    # Give a valid config. If alpha beta is false, pvs must be false, etc
    config.search.use_alpha_beta = False
    config.search.use_pvs = False
    config.search.use_aspiration_windows = False
    config.search.use_lmr = False
    config.search.use_null_move_pruning = False
    config.search.use_iid = False
    config.evaluation.use_mobility = False

    file_path = tmp_path / "config.json"
    config.save_to_json(file_path)

    assert file_path.exists()

    loaded_config = EngineConfig.load_from_json(file_path)
    assert loaded_config.search_depth == 15
    assert loaded_config.search.use_alpha_beta is False
    assert loaded_config.evaluation.use_mobility is False


def test_engine_config_from_dict():
    data = {
        "search_depth": 7,
        "search": {"use_pvs": False, "use_alpha_beta": True},
        "evaluation": {"use_pst": True, "use_pawn_structure": True},
    }
    config = EngineConfig.from_dict(data)
    assert config.search_depth == 7
    assert config.search.use_pvs is False
    assert config.search.use_alpha_beta is True
    assert config.evaluation.use_pst is True
    assert config.evaluation.use_pawn_structure is True


def test_engine_config_str_formatting():
    # Base configuration
    config = EngineConfig(search_depth=5)
    config.search.use_alpha_beta = False
    assert "Search: [Base Minimax]" in str(config)

    # Advanced search features
    config.search = SearchConfig(
        use_alpha_beta=True,
        use_pvs=True,
        use_aspiration_windows=True,
        use_transposition_table=True,
        use_tt_aging=True,
        use_iid=True,
        use_move_ordering=True,
        use_lmr=True,
        use_null_move_pruning=True,
        use_futility_pruning=True,
        use_quiescence_search=True,
    )
    s = str(config)
    assert "Depth: 5" in s
    assert "a-b" in s
    assert "IDDFS" in s
    assert "PVS" in s
    assert "AspWin" in s
    assert "TT/Z+Age" in s
    assert "IID" in s
    assert "MoveOrder" in s
    assert "LMR" in s
    assert "NMP" in s
    assert "Futility" in s
    assert "QS" in s

    # Eval flags disabled
    config.evaluation = EvaluationConfig(
        use_pst=False,
        use_pawn_structure=False,
        use_mobility=False,
        use_king_safety=False,
        game_stage_conscious=False,
    )
    assert "Eval: [Material]" in str(config)

    # Eval flags enabled
    config.evaluation = EvaluationConfig(
        use_pst=True,
        use_pawn_structure=True,
        use_mobility=True,
        use_king_safety=True,
        game_stage_conscious=True,
    )
    s = str(config)
    assert "PST" in s
    assert "Pawns" in s
    assert "Mobility" in s
    assert "KingSafety" in s
    assert "GSC" in s


def test_config_solver_bounds_depth_too_low():
    config = EngineConfig(search_depth=0)
    solver = ConfigSolver(config)
    with pytest.raises(ConfigSolverError, match="Search depth must be at least 1"):
        solver.solve()


def test_config_solver_bounds_depth_too_high():
    config = EngineConfig(search_depth=200)
    solver = ConfigSolver(config)
    with pytest.raises(ConfigSolverError, match="Search depth too high"):
        solver.solve()


def test_config_solver_bounds_timeout_negative():
    config = EngineConfig()
    config.search.max_time = -1.0
    solver = ConfigSolver(config)
    with pytest.raises(ConfigSolverError, match="Minimax timeout must be positive"):
        solver.solve()


def test_config_solver_bounds_timeout_zero():
    config = EngineConfig()
    config.search.max_time = 0.0
    solver = ConfigSolver(config)
    with pytest.raises(ConfigSolverError, match="Minimax timeout must be positive"):
        solver.solve()

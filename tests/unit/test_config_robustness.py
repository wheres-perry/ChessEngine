import pytest

from engine.config import EngineConfig, EvaluationConfig, SearchConfig
from engine.factory import create_engine_runtime


class TestConfigRobustness:
    """
    Rigorously test the configuration validation and factory logic to ensure
    dependencies are enforced and incompatibilities are caught.
    """

    def test_valid_alpha_beta_only(self):
        """Verify: Custom Core + Python Search (Alpha-Beta Only) is valid."""
        cfg = EngineConfig(
            search=SearchConfig(
                use_alpha_beta=True,
                use_move_ordering=False,
                use_transposition_table=False,
                use_tt_aging=False,
                use_hash_move_ordering=False,
                use_iid=False,
                use_check_extensions=False,
                use_pvs=False,
                use_quiescence_search=False,
                use_null_move_pruning=False,
                use_killer_moves=False,
                use_history_heuristic=False,
                use_countermove_heuristic=False,
                use_mvv_lva=False,
                use_see_ordering=False,
                use_lmr=False,
                use_delta_pruning=False,
                use_see_pruning_in_qs=False,
                use_futility_pruning=False,
                use_extended_futility_pruning=False,
                use_reverse_futility_pruning=False,
                use_aspiration_windows=False,
            ),
        )
        # Should initialize without error
        assert cfg.search.use_alpha_beta is True
        assert cfg.search.use_move_ordering is False

        # Should result in a valid runtime
        runtime = create_engine_runtime(cfg)
        assert runtime is not None

    def test_invalid_killer_without_ordering(self):
        """Verify: Enabling Killer Moves without Move Ordering raises ValueError."""
        with pytest.raises(
            ValueError, match="Killer heuristic requires both move ordering"
        ):
            EngineConfig(
                search=SearchConfig(
                    use_alpha_beta=True,
                    use_move_ordering=False,  # Missing dependency
                    use_killer_moves=True,
                )
            )

    def test_invalid_pvs_without_ab(self):
        """Verify: PVS requires Alpha-Beta."""
        with pytest.raises(
            ValueError, match=r"Principal Variation Search.*requires alpha-beta"
        ):
            EngineConfig(
                search=SearchConfig(
                    use_alpha_beta=False,  # Missing dependency
                    use_pvs=True,
                )
            )

    def test_invalid_tt_aging_without_tt(self):
        """Verify: TT Aging requires TT."""
        with pytest.raises(ValueError, match="TT aging requires transposition table"):
            EngineConfig(
                search=SearchConfig(
                    use_transposition_table=False,  # Missing dependency
                    use_tt_aging=True,
                )
            )

    def test_full_optimization_stack(self):
        """Verify: A fully loaded configuration passes validation."""
        cfg = EngineConfig(
            search=SearchConfig(
                use_alpha_beta=True,
                use_move_ordering=True,
                use_transposition_table=True,
                use_tt_aging=True,
                use_pvs=True,
                use_quiescence_search=True,
                use_killer_moves=True,
                use_history_heuristic=True,
                use_countermove_heuristic=True,
                use_lmr=True,
                use_null_move_pruning=True,
            )
        )
        assert cfg is not None

    def test_eval_dependency_pawn_structure_requires_pst(self):
        """Verify: Pawn structure evaluation requires PST to be enabled."""
        with pytest.raises(
            ValueError,
            match="Pawn structure evaluation requires Piece-Square Tables",
        ):
            EngineConfig(
                evaluation=EvaluationConfig(
                    use_pst=False,
                    use_pawn_structure=True,
                )
            )

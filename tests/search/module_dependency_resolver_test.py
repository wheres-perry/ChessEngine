"""
Test suite for the module dependency resolver.

This module contains tests for the DependencyResolver class, which validates
feature dependencies and ensures configurations are internally consistent.
Tests cover:
- Valid configurations
- Zobrist-related dependencies
- Alpha-beta pruning dependencies
- Move ordering dependencies
- Search refinement dependencies
- Error handling and messages
"""

import pytest

from src.engine.config import EngineConfig, SearchConfig
from src.engine.module_dependency_resolver import (
    DependencyResolutionError,
    DependencyResolver,
)


class TestDependencyResolverBasics:
    """Test basic functionality of the dependency resolver."""

    def test_resolver_initialization(self):
        """Test that resolver initializes correctly with valid config."""
        config = EngineConfig()
        resolver = DependencyResolver(config)

        assert resolver.config is config
        assert resolver.search_config is config.minimax
        assert isinstance(resolver.search_config, SearchConfig)

    def test_resolve_valid_default_config(self):
        """Test that default config resolves successfully."""
        config = EngineConfig()
        resolver = DependencyResolver(config)

        result = resolver.resolve()

        assert isinstance(result, SearchConfig)
        assert result is config.minimax

    def test_resolve_returns_same_config_object(self):
        """Test that resolve returns the same SearchConfig object."""
        config = EngineConfig()
        resolver = DependencyResolver(config)

        result1 = resolver.resolve()
        result2 = resolver.resolve()

        assert result1 is result2
        assert result1 is config.minimax


class TestZobristDependencies:
    """Test Zobrist hashing related dependencies."""

    def test_transposition_table_requires_zobrist(self):
        """Test that TT without Zobrist raises error."""
        config = EngineConfig()
        config.minimax.use_transposition_table = True
        config.minimax.use_zobrist = False

        resolver = DependencyResolver(config)

        with pytest.raises(
            DependencyResolutionError,
            match="Transposition table requires Zobrist hashing to be enabled",
        ):
            resolver.resolve()

    def test_tt_aging_requires_zobrist(self):
        """Test that TT aging without Zobrist raises error."""
        config = EngineConfig()
        # Start with all disabled to isolate this dependency
        config.minimax.use_zobrist = False
        config.minimax.use_transposition_table = False
        config.minimax.use_tt_aging = True

        resolver = DependencyResolver(config)

        with pytest.raises(
            DependencyResolutionError,
            match="TT aging requires Zobrist hashing to be enabled",
        ):
            resolver.resolve()

    def test_zobrist_enabled_with_all_dependencies_met(self):
        """Test that Zobrist with all dependencies works."""
        config = EngineConfig()
        config.minimax.use_zobrist = True
        config.minimax.use_transposition_table = True
        config.minimax.use_tt_aging = True

        resolver = DependencyResolver(config)
        result = resolver.resolve()

        assert result.use_zobrist is True
        assert result.use_transposition_table is True
        assert result.use_tt_aging is True


class TestAlphaBetaDependencies:
    """Test alpha-beta pruning related dependencies."""

    def test_pvs_requires_alpha_beta(self):
        """Test that PVS without alpha-beta raises error."""
        config = EngineConfig()
        config.minimax.use_pvs = True
        config.minimax.use_alpha_beta = False

        resolver = DependencyResolver(config)

        with pytest.raises(
            DependencyResolutionError,
            match="Principal Variation Search \\(PVS\\) requires alpha-beta pruning",
        ):
            resolver.resolve()

    def test_aspiration_windows_requires_alpha_beta(self):
        """Test that aspiration windows without alpha-beta raises error."""
        config = EngineConfig()
        config.minimax.use_aspiration_windows = True
        config.minimax.use_alpha_beta = False

        resolver = DependencyResolver(config)

        # Will catch PVS error first (default is True) or aspiration error
        with pytest.raises(DependencyResolutionError):
            resolver.resolve()

    def test_null_move_pruning_requires_alpha_beta(self):
        """Test that null move pruning without alpha-beta raises error."""
        config = EngineConfig()
        config.minimax.use_null_move_pruning = True
        config.minimax.use_alpha_beta = False

        resolver = DependencyResolver(config)

        # Will catch some alpha-beta dependency error
        with pytest.raises(DependencyResolutionError):
            resolver.resolve()


class TestMoveOrderingDependencies:
    """Test move ordering related dependencies."""

    def test_lmr_requires_both_alpha_beta_and_move_ordering(self):
        """Test that LMR requires both alpha-beta and move ordering."""
        config = EngineConfig()
        config.minimax.use_lmr = True
        config.minimax.use_alpha_beta = True
        config.minimax.use_move_ordering = False

        resolver = DependencyResolver(config)

        with pytest.raises(
            DependencyResolutionError,
            match=(
                "Late Move Reduction \\(LMR\\) requires both "
                "alpha-beta pruning and move ordering"
            ),
        ):
            resolver.resolve()

    def test_hash_move_ordering_requires_move_ordering(self):
        """Test that hash move ordering requires move ordering."""
        config = EngineConfig()
        config.minimax.use_hash_move_ordering = True
        config.minimax.use_move_ordering = False

        resolver = DependencyResolver(config)

        # Will catch LMR or hash move ordering error
        with pytest.raises(DependencyResolutionError):
            resolver.resolve()

    def test_hash_move_ordering_requires_transposition_table(self):
        """Test that hash move ordering requires transposition table."""
        config = EngineConfig()
        config.minimax.use_hash_move_ordering = True
        config.minimax.use_move_ordering = True
        config.minimax.use_transposition_table = False

        resolver = DependencyResolver(config)

        with pytest.raises(
            DependencyResolutionError,
            match="Hash move ordering requires transposition table",
        ):
            resolver.resolve()

    def test_move_ordering_features_require_move_ordering(self):
        """Test that various move ordering features require move ordering enabled."""
        features = [
            "use_mvv_lva",
            "use_see_ordering",
            "use_killer_moves",
            "use_history_heuristic",
            "use_countermove_heuristic",
        ]

        for feature_name in features:
            config = EngineConfig()
            # Disable move ordering
            config.minimax.use_move_ordering = False
            # Enable the specific feature
            setattr(config.minimax, feature_name, True)

            resolver = DependencyResolver(config)

            # Should raise some dependency error (may be LMR or the specific feature)
            with pytest.raises(DependencyResolutionError):
                resolver.resolve()


class TestSearchRefinementDependencies:
    """Test search refinement related dependencies."""

    def test_iid_requires_iddfs(self):
        """Test that IID requires IDDFS."""
        config = EngineConfig()
        config.minimax.use_iid = True
        config.minimax.use_iddfs = False

        resolver = DependencyResolver(config)

        with pytest.raises(
            DependencyResolutionError,
            match="Internal Iterative Deepening \\(IID\\) requires IDDFS",
        ):
            resolver.resolve()

    def test_delta_pruning_requires_quiescence_search(self):
        """Test that delta pruning requires quiescence search."""
        config = EngineConfig()
        config.minimax.use_delta_pruning = True
        config.minimax.use_quiescence_search = False

        resolver = DependencyResolver(config)

        with pytest.raises(
            DependencyResolutionError,
            match="Delta pruning requires quiescence search",
        ):
            resolver.resolve()

    def test_see_pruning_in_qs_requires_quiescence_search(self):
        """Test that SEE pruning in QS requires quiescence search."""
        config = EngineConfig()
        config.minimax.use_see_pruning_in_qs = True
        config.minimax.use_quiescence_search = False

        resolver = DependencyResolver(config)

        # Will catch delta pruning error first (default True) or SEE pruning error
        with pytest.raises(DependencyResolutionError):
            resolver.resolve()


class TestValidConfigurations:
    """Test various valid configuration combinations."""

    def test_minimal_configuration(self):
        """Test a minimal configuration with core features only."""
        config = EngineConfig()
        # Disable most features
        config.minimax.use_alpha_beta = True
        config.minimax.use_zobrist = False
        config.minimax.use_transposition_table = False
        config.minimax.use_tt_aging = False
        config.minimax.use_iddfs = True
        config.minimax.use_move_ordering = False
        config.minimax.use_pvs = False
        config.minimax.use_lmr = False
        config.minimax.use_aspiration_windows = False
        config.minimax.use_null_move_pruning = False
        config.minimax.use_hash_move_ordering = False
        config.minimax.use_mvv_lva = False
        config.minimax.use_see_ordering = False
        config.minimax.use_killer_moves = False
        config.minimax.use_history_heuristic = False
        config.minimax.use_countermove_heuristic = False
        config.minimax.use_iid = False

        resolver = DependencyResolver(config)
        result = resolver.resolve()

        assert isinstance(result, SearchConfig)
        assert result.use_alpha_beta is True
        assert result.use_zobrist is False

    def test_maximal_configuration(self):
        """Test maximal configuration with all compatible features enabled."""
        config = EngineConfig()
        # Default config should have all features properly configured
        resolver = DependencyResolver(config)
        result = resolver.resolve()

        assert isinstance(result, SearchConfig)
        # Verify some key features are enabled
        assert result.use_alpha_beta is True
        assert result.use_zobrist is True
        assert result.use_pvs is True
        assert result.use_move_ordering is True

    def test_zobrist_without_transposition_table(self):
        """Test that Zobrist can be used without TT."""
        config = EngineConfig()
        config.minimax.use_zobrist = True
        config.minimax.use_transposition_table = False
        config.minimax.use_tt_aging = False
        config.minimax.use_hash_move_ordering = False

        resolver = DependencyResolver(config)
        result = resolver.resolve()

        assert result.use_zobrist is True
        assert result.use_transposition_table is False

    def test_move_ordering_without_advanced_features(self):
        """Test move ordering with basic heuristics only."""
        config = EngineConfig()
        config.minimax.use_move_ordering = True
        config.minimax.use_hash_move_ordering = False
        config.minimax.use_mvv_lva = True
        config.minimax.use_killer_moves = True

        resolver = DependencyResolver(config)
        result = resolver.resolve()

        assert result.use_move_ordering is True
        assert result.use_mvv_lva is True


class TestDependencyChains:
    """Test complex dependency chains."""

    def test_hash_move_ordering_chain(self):
        """Test hash move ordering -> TT -> Zobrist dependency chain."""
        config = EngineConfig()
        config.minimax.use_hash_move_ordering = True
        config.minimax.use_move_ordering = True
        config.minimax.use_transposition_table = True
        config.minimax.use_zobrist = False  # Break the chain

        resolver = DependencyResolver(config)

        with pytest.raises(
            DependencyResolutionError,
            match="Transposition table requires Zobrist hashing to be enabled",
        ):
            resolver.resolve()

    def test_lmr_requires_both_dependencies(self):
        """Test that LMR requires both alpha-beta AND move ordering."""
        # Test missing alpha-beta
        config1 = EngineConfig()
        config1.minimax.use_lmr = True
        config1.minimax.use_alpha_beta = False
        config1.minimax.use_move_ordering = True

        resolver1 = DependencyResolver(config1)
        with pytest.raises(DependencyResolutionError):
            resolver1.resolve()

        # Test missing move ordering
        config2 = EngineConfig()
        config2.minimax.use_lmr = True
        config2.minimax.use_alpha_beta = True
        config2.minimax.use_move_ordering = False

        resolver2 = DependencyResolver(config2)
        with pytest.raises(DependencyResolutionError):
            resolver2.resolve()


class TestResolverBehavior:
    """Test resolver behavior and edge cases."""

    def test_multiple_resolvers_same_config(self):
        """Test that multiple resolvers work with the same config."""
        config = EngineConfig()
        resolver1 = DependencyResolver(config)
        resolver2 = DependencyResolver(config)

        result1 = resolver1.resolve()
        result2 = resolver2.resolve()

        assert result1 is result2
        assert result1 is config.minimax

    def test_resolver_does_not_modify_config(self):
        """Test that resolver does not modify the original config."""
        config = EngineConfig()
        original_pvs = config.minimax.use_pvs
        original_zobrist = config.minimax.use_zobrist

        resolver = DependencyResolver(config)
        resolver.resolve()

        assert config.minimax.use_pvs == original_pvs
        assert config.minimax.use_zobrist == original_zobrist

    def test_resolve_can_be_called_multiple_times(self):
        """Test that resolve can be called multiple times safely."""
        config = EngineConfig()
        resolver = DependencyResolver(config)

        result1 = resolver.resolve()
        result2 = resolver.resolve()
        result3 = resolver.resolve()

        assert result1 is result2 is result3

    def test_config_modification_after_resolver_creation(self):
        """Test that modifying config after resolver creation is reflected."""
        config = EngineConfig()
        resolver = DependencyResolver(config)

        # Modify config after resolver creation but before resolve
        config.minimax.use_pvs = False

        result = resolver.resolve()

        assert result.use_pvs is False

    def test_invalid_modification_is_caught(self):
        """Test that creating invalid config after resolver init is caught."""
        config = EngineConfig()
        resolver = DependencyResolver(config)

        # Create invalid dependency after resolver creation
        config.minimax.use_transposition_table = True
        config.minimax.use_zobrist = False

        with pytest.raises(DependencyResolutionError):
            resolver.resolve()


class TestErrorMessages:
    """Test that error messages are clear and helpful."""

    def test_zobrist_error_message(self):
        """Test Zobrist dependency error message."""
        config = EngineConfig()
        config.minimax.use_transposition_table = True
        config.minimax.use_zobrist = False

        resolver = DependencyResolver(config)

        with pytest.raises(DependencyResolutionError) as exc_info:
            resolver.resolve()

        assert "Zobrist" in str(exc_info.value)
        assert "Transposition table" in str(exc_info.value)

    def test_pvs_error_message(self):
        """Test PVS dependency error message."""
        config = EngineConfig()
        config.minimax.use_pvs = True
        config.minimax.use_alpha_beta = False

        resolver = DependencyResolver(config)

        with pytest.raises(DependencyResolutionError) as exc_info:
            resolver.resolve()

        assert "PVS" in str(exc_info.value) or "alpha-beta" in str(exc_info.value)

    def test_lmr_error_message(self):
        """Test LMR dependency error message."""
        config = EngineConfig()
        config.minimax.use_lmr = True
        config.minimax.use_move_ordering = False

        resolver = DependencyResolver(config)

        with pytest.raises(DependencyResolutionError) as exc_info:
            resolver.resolve()

        error_msg = str(exc_info.value)
        # Should mention LMR and its requirements
        assert (
            "LMR" in error_msg
            or "move ordering" in error_msg
            or "alpha-beta" in error_msg
        )

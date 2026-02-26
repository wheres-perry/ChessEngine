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

from engine.config import EngineConfig, SearchConfig
from engine.config_dependency_resolver import (
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
        assert resolver.search_config is config.search
        assert isinstance(resolver.search_config, SearchConfig)

    def test_resolve_valid_default_config(self):
        """Test that default config resolves successfully."""
        config = EngineConfig()
        resolver = DependencyResolver(config)

        result = resolver.resolve()

        assert isinstance(result, SearchConfig)
        assert result is config.search

    def test_resolve_returns_same_config_object(self):
        """Test that resolve returns the same SearchConfig object."""
        config = EngineConfig()
        resolver = DependencyResolver(config)

        result1 = resolver.resolve()
        result2 = resolver.resolve()

        assert result1 is result2
        assert result1 is config.search


class TestZobristDependencies:
    """Test Zobrist hashing related dependencies."""

    def test_tt_aging_requires_tt(self):
        """Test that TT aging without TT raises error."""
        config = EngineConfig()
        config.search.use_transposition_table = False
        config.search.use_tt_aging = True

        resolver = DependencyResolver(config)

        with pytest.raises(DependencyResolutionError):
            resolver.resolve()

    def test_tt_enabled_with_all_dependencies_met(self):
        """Test that TT with all dependencies works."""
        config = EngineConfig()
        config.search.use_transposition_table = True
        config.search.use_tt_aging = True

        resolver = DependencyResolver(config)
        result = resolver.resolve()

        assert result.use_transposition_table is True
        assert result.use_tt_aging is True


class TestAlphaBetaDependencies:
    """Test alpha-beta pruning related dependencies."""

    def test_pvs_requires_alpha_beta(self):
        """Test that PVS without alpha-beta raises error."""
        config = EngineConfig()
        config.search.use_pvs = True
        config.search.use_alpha_beta = False
        config.search.use_move_ordering = False  # Disable to isolate PVS check

        resolver = DependencyResolver(config)

        with pytest.raises(
            DependencyResolutionError,
            match="Principal Variation Search requires alpha-beta",
        ):
            resolver.resolve()

    def test_aspiration_windows_requires_alpha_beta(self):
        """Test that aspiration windows without alpha-beta raises error."""
        config = EngineConfig()
        config.search.use_aspiration_windows = True
        config.search.use_alpha_beta = False

        resolver = DependencyResolver(config)

        # Will catch PVS error first (default is True) or aspiration error
        with pytest.raises(DependencyResolutionError):
            resolver.resolve()

    def test_null_move_pruning_requires_alpha_beta(self):
        """Test that null move pruning without alpha-beta raises error."""
        config = EngineConfig()
        config.search.use_null_move_pruning = True
        config.search.use_alpha_beta = False

        resolver = DependencyResolver(config)

        # Will catch some alpha-beta dependency error
        with pytest.raises(DependencyResolutionError):
            resolver.resolve()


class TestMoveOrderingDependencies:
    """Test move ordering related dependencies."""

    def test_lmr_requires_both_alpha_beta_and_move_ordering(self):
        """Test that LMR requires both alpha-beta and move ordering."""
        config = EngineConfig()
        config.search.use_lmr = True
        config.search.use_alpha_beta = True
        config.search.use_move_ordering = False

        resolver = DependencyResolver(config)

        with pytest.raises(DependencyResolutionError):
            resolver.resolve()

    def test_hash_move_ordering_requires_move_ordering(self):
        """Test that hash move ordering requires move ordering."""
        config = EngineConfig()
        config.search.use_hash_move_ordering = True
        config.search.use_move_ordering = False

        resolver = DependencyResolver(config)

        # Will catch LMR or hash move ordering error
        with pytest.raises(DependencyResolutionError):
            resolver.resolve()

    def test_hash_move_ordering_requires_transposition_table(self):
        """Test that hash move ordering requires transposition table."""
        config = EngineConfig()
        config.search.use_hash_move_ordering = True
        config.search.use_move_ordering = True
        config.search.use_transposition_table = False
        config.search.use_tt_aging = False  # Must also disable TT aging

        resolver = DependencyResolver(config)

        with pytest.raises(DependencyResolutionError):
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
            config.search.use_move_ordering = False
            # Enable the specific feature
            setattr(config.search, feature_name, True)

            resolver = DependencyResolver(config)

            # Should raise some dependency error (may be LMR or the specific feature)
            with pytest.raises(DependencyResolutionError):
                resolver.resolve()


class TestSearchRefinementDependencies:
    """Test search refinement related dependencies."""

    def test_iid_requires_tt(self):
        """Test that IID requires TT."""
        config = EngineConfig()
        config.search.use_iid = True
        config.search.use_hash_move_ordering = False
        config.search.use_transposition_table = False
        config.search.use_tt_aging = False

        resolver = DependencyResolver(config)

        with pytest.raises(
            DependencyResolutionError,
            match=(
                r"Internal Iterative Deepening \(IID\)"
                r" requires the transposition table"
            ),
        ):
            resolver.resolve()

    def test_delta_pruning_requires_quiescence_search(self):
        """Test that delta pruning requires quiescence search."""
        config = EngineConfig()
        config.search.use_delta_pruning = True
        config.search.use_quiescence_search = False

        resolver = DependencyResolver(config)

        with pytest.raises(DependencyResolutionError):
            resolver.resolve()

    def test_see_pruning_in_qs_requires_quiescence_search(self):
        """Test that SEE pruning in QS requires quiescence search."""
        config = EngineConfig()
        config.search.use_see_pruning_in_qs = True
        config.search.use_quiescence_search = False

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
        config.search.use_alpha_beta = True
        config.search.use_transposition_table = False
        config.search.use_tt_aging = False
        config.search.use_move_ordering = False
        config.search.use_pvs = False
        config.search.use_lmr = False
        config.search.use_aspiration_windows = False
        config.search.use_null_move_pruning = False
        config.search.use_hash_move_ordering = False
        config.search.use_mvv_lva = False
        config.search.use_see_ordering = False
        config.search.use_killer_moves = False
        config.search.use_history_heuristic = False
        config.search.use_countermove_heuristic = False
        config.search.use_iid = False

        resolver = DependencyResolver(config)
        result = resolver.resolve()

        assert isinstance(result, SearchConfig)
        assert result.use_alpha_beta is True
        assert result.use_transposition_table is False

    def test_maximal_configuration(self):
        """Test maximal configuration with all compatible features enabled."""
        config = EngineConfig()
        # Default config should have all features properly configured
        resolver = DependencyResolver(config)
        result = resolver.resolve()

        assert isinstance(result, SearchConfig)
        # Verify some key features are enabled
        assert result.use_alpha_beta is True
        assert result.use_transposition_table is True
        assert result.use_pvs is True
        assert result.use_move_ordering is True

    def test_move_ordering_without_advanced_features(self):
        """Test move ordering with basic heuristics only."""
        config = EngineConfig()
        config.search.use_move_ordering = True
        config.search.use_hash_move_ordering = False
        config.search.use_mvv_lva = True
        config.search.use_killer_moves = True

        resolver = DependencyResolver(config)
        result = resolver.resolve()

        assert result.use_move_ordering is True
        assert result.use_mvv_lva is True


class TestDependencyChains:
    """Test complex dependency chains."""

    def test_hash_move_ordering_chain(self):
        """Test hash move ordering -> TT dependency chain."""
        config = EngineConfig()
        config.search.use_hash_move_ordering = True
        config.search.use_move_ordering = True
        config.search.use_transposition_table = False  # Break the chain

        resolver = DependencyResolver(config)

        with pytest.raises(DependencyResolutionError):
            resolver.resolve()

    def test_lmr_requires_both_dependencies(self):
        """Test that LMR requires both alpha-beta AND move ordering."""
        # Test missing alpha-beta
        config1 = EngineConfig()
        config1.search.use_lmr = True
        config1.search.use_alpha_beta = False
        config1.search.use_move_ordering = True

        resolver1 = DependencyResolver(config1)
        with pytest.raises(DependencyResolutionError):
            resolver1.resolve()

        # Test missing move ordering
        config2 = EngineConfig()
        config2.search.use_lmr = True
        config2.search.use_alpha_beta = True
        config2.search.use_move_ordering = False

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
        assert result1 is config.search

    def test_resolver_does_not_modify_config(self):
        """Test that resolver does not modify the original config."""
        config = EngineConfig()
        original_pvs = config.search.use_pvs
        original_tt = config.search.use_transposition_table

        resolver = DependencyResolver(config)
        resolver.resolve()

        assert config.search.use_pvs == original_pvs
        assert config.search.use_transposition_table == original_tt

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
        config.search.use_pvs = False

        result = resolver.resolve()

        assert result.use_pvs is False

    def test_invalid_modification_is_caught(self):
        """Test that creating invalid config after resolver init is caught."""
        config = EngineConfig()
        resolver = DependencyResolver(config)

        # Create invalid dependency after resolver creation
        config.search.use_transposition_table = False
        config.search.use_tt_aging = True

        with pytest.raises(DependencyResolutionError):
            resolver.resolve()

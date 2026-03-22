"""Benchmark search metrics across different engine configurations.

This module provides comprehensive benchmarking of search performance
including cold vs warm runs, multi-depth sweeps, and statistics collection.
"""

import statistics
import time
from datetime import datetime
from pathlib import Path

import pytest

from engine import create_engine_runtime
from engine.config import EngineConfig, SearchConfig
from tests.benchmarks.infrastructure import (
    BaselineManager,
    BenchmarkResult,
    SearchStats,
)

BENCHMARK_FENS = {
    "Start": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "KiwiPete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "Middlegame": "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
}


def get_configs():
    """Return the 'drop-in' module configurations."""
    # 1. Base Minimax
    base_cfg = EngineConfig(
        search=SearchConfig(
            use_alpha_beta=False,
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
            max_time=2.0,
        ),
        search_depth=1,
    )

    # 2. Full Optimized
    full_cfg = EngineConfig(
        search=SearchConfig(
            use_alpha_beta=True,
            use_move_ordering=True,
            use_transposition_table=True,
            use_pvs=True,
            use_quiescence_search=True,
            use_killer_moves=True,
            use_history_heuristic=True,
            use_lmr=True,
            use_null_move_pruning=True,
            max_time=2.0,
        ),
        search_depth=3,
    )

    return {
        "Base_Minimax": base_cfg,
        "Full_Optimized": full_cfg,
    }


@pytest.fixture
def baseline_manager(tmp_path):
    """Return a BaselineManager for the test session."""
    bm_dir = Path(".benchmarks")
    return BaselineManager(bm_dir)


def run_search_harness(config, fen, depth=None, warm=False, runtime_instance=None):
    """Run a search with the given configuration and position."""
    if runtime_instance and warm:
        runtime = runtime_instance
        # For warm runs, we might need to reset board position if it changed,
        # but keep searcher state (TT).
        # Adapter from_fen updates the board in place usually.
        runtime.board.from_fen(fen)
        # Note: If Hybrid, C++ searcher adapter makes a COPY
        # of the board at search time,
        # so state reuse is limited to the adapter instance (which might reset stats).
        # Native C++ searcher holds a reference to the board.
    else:
        runtime = create_engine_runtime(config, fen)
        runtime.searcher.reset()

    run_depth = depth if depth is not None else config.search_depth
    start_time = time.time()
    score, move = runtime.searcher.search(depth=run_depth)
    duration = time.time() - start_time

    return runtime, duration, score, move


def test_search_metrics_full_suite(benchmark, baseline_manager):
    """Run comprehensive search benchmark suite using the Factory.

    Includes cold vs warm runs, multiple iterations, C++ vs Python comparison,
    and hybrid backend verification.
    """
    configs = get_configs()
    iterations = 2
    depths = [1, 2]  # Multi-depth sweep

    for name, config in configs.items():
        fen_stats = {}

        for fen_name, fen in BENCHMARK_FENS.items():
            for depth in depths:
                # Cold Run Loop
                durations_cold = []
                nodes_cold = []
                scores_cold = []

                # Keep last stats to capture counters
                last_stats = None
                last_move = None

                for _ in range(iterations):
                    runtime, duration, score, move = run_search_harness(
                        config, fen, depth=depth, warm=False
                    )
                    durations_cold.append(duration)
                    scores_cold.append(score if score is not None else 0)
                    s = runtime.searcher.get_stats()
                    nodes_cold.append(s.nodes)
                    last_stats = s
                    last_move = move

                # Warm Run Loop
                # Not all backends benefit from warm runs equally
                # (e.g. Hybrid), but we run it.
                # Reuse last runtime
                runtime_warm = runtime
                durations_warm = []

                for _ in range(iterations):
                    # pass runtime_warm to reuse
                    runtime_warm, duration, _, move = run_search_harness(
                        config,
                        fen,
                        depth=depth,
                        warm=True,
                        runtime_instance=runtime_warm,
                    )
                    durations_warm.append(duration)

                avg_duration_cold = statistics.mean(durations_cold)
                avg_nodes_cold = statistics.mean(nodes_cold)

                time_stddev = (
                    statistics.stdev(durations_cold) if len(durations_cold) > 1 else 0.0
                )
                nodes_stddev = (
                    statistics.stdev(nodes_cold) if len(nodes_cold) > 1 else 0.0
                )
                score_variance = (
                    statistics.variance(scores_cold) if len(scores_cold) > 1 else 0.0
                )

                # Construct final stats using the last run's counters + averages
                stats = SearchStats(
                    nodes=int(avg_nodes_cold),
                    depth=depth,
                    seldepth=last_stats.seldepth,
                    time_to_depth=avg_duration_cold,
                    nps=int(avg_nodes_cold / avg_duration_cold)
                    if avg_duration_cold > 0
                    else 0,
                    beta_cutoffs=last_stats.beta_cutoffs,
                    fmbc_count=last_stats.first_move_cuts,
                    killer_cuts=last_stats.killer_cuts,
                    history_cuts=last_stats.history_cuts,
                    tt_hits=last_stats.tt_hits,
                    qsearch_nodes=last_stats.qsearch_nodes,
                    null_move_cuts=last_stats.null_move_cuts,
                    pvs_researches=getattr(last_stats, "pvs_researches", 0),
                    root_move_changes=getattr(last_stats, "root_move_changes", 0),
                    score=int(statistics.mean(scores_cold)),
                    score_variance=score_variance,
                    time_stddev=time_stddev,
                    nodes_stddev=nodes_stddev,
                    best_move=last_move or "",
                )
                stats.calculate_derived_metrics()
                # Key by FEN + Depth
                fen_stats[f"{fen_name}_d{depth}"] = stats

        # Save Results
        result = BenchmarkResult(
            config_name=name, timestamp=datetime.now().isoformat(), stats=fen_stats
        )
        baseline_manager.save_baseline(f"search_{name}", result.to_dict())

    # Example comparison
    # diff = baseline_manager.compare(...)
    # print(diff)

    # Use pytest-benchmark for one representative config
    hybrid_cfg = configs["Full_Optimized"]
    start_fen = BENCHMARK_FENS["Start"]

    def run_bench():
        rt = create_engine_runtime(hybrid_cfg, start_fen)
        rt.searcher.search(depth=1)  # Fast depth

    benchmark(run_bench)

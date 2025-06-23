import argparse
import logging
import os
import queue
import sys
import threading
import time

import chess
import src.engine.search.minimax as minimax
import src.engine.evaluators.simple_eval as simple_eval
from src.engine.config import EngineConfig, EvaluationConfig, MinimaxConfig


def handle_args():
    parser = argparse.ArgumentParser(description="Chess Engine")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity. -v for INFO, -vv for DEBUG.",
    )

    # Add new optimization flags

    parser.add_argument(
        "--no-zobrist",
        action="store_true",
        help="Disable Zobrist hashing and transposition tables",
    )
    parser.add_argument(
        "--no-iddfs",
        action="store_true",
        help="Disable Iterative Deepening Depth-First Search",
    )
    parser.add_argument(
        "--no-alpha-beta",
        action="store_true",
        help="Disable Alpha-Beta pruning",
    )
    parser.add_argument(
        "--no-move-ordering",
        action="store_true",
        help="Disable move ordering",
    )
    parser.add_argument(
        "--no-pvs",
        action="store_true",
        help="Disable Principal Variation Search",
    )
    parser.add_argument(
        "--no-lmr",
        action="store_true",
        help="Disable Late Move Reduction",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Search depth for benchmarks (default: 8)",
    )
    parser.add_argument(
        "--evaluator",
        choices=["simple", "complex", "mock"],
        default="simple",
        help="Type of position evaluator to use",
    )

    args = parser.parse_args()
    return args


class TimeoutException(Exception):
    pass


def run_minimax_with_timeout(mm, depth, result_queue, timeout_seconds):
    """Run minimax search in a separate thread with timeout."""
    try:
        score, move = mm.find_top_move(depth=depth)
        result_queue.put(("success", score, move))
    except Exception as e:
        result_queue.put(("error", str(e), None))


def benchmark_minimax():
    """Benchmark different Minimax configurations on multiple games."""
    logger = logging.getLogger(__name__)

    # Get command line arguments
    args = handle_args()
    depth = args.depth
    timeout_seconds = None  # No timeout for benchmark runs

    # Create a valid evaluation config based on args
    eval_config_kwargs = {"evaluator_type": args.evaluator}
    if args.evaluator != "complex":
        # For 'simple' and 'mock', disable all complex flags
        eval_config_kwargs["use_pst"] = False
        eval_config_kwargs["use_mobility"] = False
        eval_config_kwargs["use_pawn_structure"] = False
        eval_config_kwargs["use_king_safety"] = False
        # 'simple' uses material, 'mock' does not.
        eval_config_kwargs["use_material"] = args.evaluator == "simple"
    
    base_eval_config = EvaluationConfig(**eval_config_kwargs)


    # Define benchmark configurations, building from minimal to full-featured
    configs = [
        {
            "name": "Minimal (No Optimizations)",
            "config": EngineConfig(
                minimax=MinimaxConfig(
                    use_zobrist=False,
                    use_iddfs=False,
                    use_alpha_beta=False,
                    use_move_ordering=False,
                    use_pvs=False,
                    use_lmr=False,
                    use_tt_aging=False,
                ),
                evaluation=base_eval_config,
                search_depth=depth,
            ),
        },
        {
            "name": "Base + Alpha-Beta",
            "config": EngineConfig(
                minimax=MinimaxConfig(
                    use_alpha_beta=True,
                    use_zobrist=False,
                    use_iddfs=False,
                    use_move_ordering=False,
                    use_pvs=False,
                    use_lmr=False,
                    use_tt_aging=False,
                ),
                evaluation=base_eval_config,
                search_depth=depth,
            ),
        },
        {
            "name": "Base + AB + MO + PVS + LMR",
            "config": EngineConfig(
                minimax=MinimaxConfig(
                    use_alpha_beta=True,
                    use_move_ordering=True,
                    use_pvs=True,
                    use_lmr=True,
                    use_zobrist=False,
                    use_iddfs=False,
                    use_tt_aging=False,
                ),
                evaluation=base_eval_config,
                search_depth=depth,
            ),
        },
        {
            "name": "Search + Zobrist/TT",
            "config": EngineConfig(
                minimax=MinimaxConfig(
                    use_alpha_beta=True,
                    use_move_ordering=True,
                    use_pvs=True,
                    use_lmr=True,
                    use_zobrist=True,
                    use_tt_aging=False,  # Test without aging first
                    use_iddfs=False,
                ),
                evaluation=base_eval_config,
                search_depth=depth,
            ),
        }
        {
            "name": "Default (All Optimizations)",
            "config": EngineConfig(
                minimax=MinimaxConfig(
                    use_zobrist=True,
                    use_iddfs=False,
                    use_alpha_beta=True,
                    use_move_ordering=True,
                    use_pvs=True,
                    use_lmr=True,
                    use_tt_aging=True,
                ),
                evaluation=base_eval_config,
                search_depth=depth,
            ),
        },
    ]


    # Define static board positions (FENs)
    static_fens = [
        "8/P3n3/pp6/p3P3/k1P1p2n/Pp2p3/1P2PpBp/3b1K2 w - - 0 1",  # Starting position
        "2k5/p1p2p2/1pbp4/4pR2/8/PNPP1B1q/1P5P/R2QK1r1 w - - 9 26",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Complex middlegame
    ]
    test_boards = [chess.Board(fen) for fen in static_fens]
    num_games = len(test_boards)

    print("=" * 80)
    print(f"MINIMAX BENCHMARK - Depth {depth}, {num_games} Static Games, No Timeout")
    print("=" * 80)

    print(f"Testing {len(test_boards)} game positions:")
    for i, board in enumerate(test_boards):
        print(f"  Game {i+1}: {board.fen()[:50]}...")
    print()

    # Results storage
    results = {}

    # Run benchmark for each configuration
    for config_item in configs:
        config_name = config_item["name"]
        engine_config = config_item["config"]

        # Validate config before running
        try:
            engine_config._validate_config()
        except ValueError as e:
            print(f"Skipping invalid configuration '{config_name}': {e}")
            print("-" * 60)
            continue

        print(f"Testing: {config_name} ({engine_config})")
        print("-" * 60)

        results[config_name] = {
            "times": [],
            "moves": [],
            "scores": [],
            "total_time": 0,
            "avg_time": 0,
            "failed": 0,
            "timeouts": 0,
        }

        for i, board in enumerate(test_boards):
            try:
                evaluator = simple_eval.SimpleEval(board.copy())
                mm = minimax.Minimax(board.copy(), evaluator, config=engine_config)
                result_queue = queue.Queue()
                search_thread = threading.Thread(
                    target=run_minimax_with_timeout,
                    args=(mm, depth, result_queue, timeout_seconds),
                )

                start_time = time.time()
                search_thread.start()
                search_thread.join(timeout=timeout_seconds)
                end_time = time.time()
                search_time = end_time - start_time

                if search_thread.is_alive():
                    results[config_name]["timeouts"] += 1
                    print(f"  Game {i+1}: TIMEOUT after {search_time:.1f}s")
                else:
                    try:
                        result_type, result_data, move = result_queue.get_nowait()
                        if result_type == "success":
                            score = result_data
                            results[config_name]["times"].append(search_time)
                            results[config_name]["moves"].append(str(move))
                            results[config_name]["scores"].append(score)
                            results[config_name]["total_time"] += search_time
                            score_display = (
                                f"{score:.2f}" if score is not None else "N/A"
                            )
                            print(
                                f"  Game {i+1}: Move={move}, Score={score_display}, Time={search_time:.3f}s"
                            )
                        else:
                            results[config_name]["failed"] += 1
                            print(f"  Game {i+1}: FAILED - {result_data}")
                    except queue.Empty:
                        results[config_name]["failed"] += 1
                        print(f"  Game {i+1}: FAILED - No result returned")
            except Exception as e:
                logger.error(f"Error in {config_name} for game {i+1}: {e}")
                results[config_name]["failed"] += 1
                print(f"  Game {i+1}: FAILED - {e}")

        if results[config_name]["times"]:
            results[config_name]["avg_time"] = results[config_name]["total_time"] / len(
                results[config_name]["times"]
            )
        else:
            results[config_name]["avg_time"] = float("inf")
        print(f"  Total Time: {results[config_name]['total_time']:.3f}s")
        print(f"  Average Time: {results[config_name]['avg_time']:.3f}s")
        print(f"  Failed: {results[config_name]['failed']}")
        print()

    # Summary table
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"{'Configuration':<40} {'Avg Time (s)':<12} {'Total Time (s)':<14} {'Timeouts':<10} {'Failed':<8}"
    )
    print("-" * 80)

    # Sort by average time (put infinite times at the end)

    sorted_configs = sorted(
        configs,
        key=lambda x: (
            results[x["name"]]["avg_time"] == float("inf"),
            results[x["name"]]["avg_time"],
        ),
    )

    for config in sorted_configs:
        name = config["name"]
        avg_time = results[name]["avg_time"]
        total_time = results[name]["total_time"]
        timeouts = results[name]["timeouts"]
        failed = results[name]["failed"]

        avg_time_str = f"{avg_time:.3f}" if avg_time != float("inf") else "TIMEOUT"
        print(
            f"{name:<40} {avg_time_str:<12} {total_time:<14.3f} {timeouts:<10} {failed:<8}"
        )
    # Performance comparison (only for non-timeout results)

    valid_configs = [
        c for c in sorted_configs if results[c["name"]]["avg_time"] != float("inf")
    ]
    if len(valid_configs) > 1:
        fastest = valid_configs[0]["name"]
        slowest = valid_configs[-1]["name"]
        speedup = results[slowest]["avg_time"] / results[fastest]["avg_time"]

        print()
        print(f"Fastest: {fastest}")
        print(f"Slowest (completed): {slowest}")
        print(f"Speedup: {speedup:.2f}x")
    print("=" * 80)


def main():
    args = handle_args()
    verbosity = args.verbose

    # Set up logging based on verbosity level

    log_level = logging.WARNING
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if verbosity == 1:
        log_level = logging.INFO
    elif verbosity >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting script with verbosity level: {verbosity}")
    logger.debug(f"Command line arguments parsed: {args}")

    # Run the benchmark
    benchmark_minimax()


if __name__ == "__main__":
    main()

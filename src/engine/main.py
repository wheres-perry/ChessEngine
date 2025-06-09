import argparse
import io_utils.load_games as load_games
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import logging
import threading
import queue

import engine.evaluators.simple_eval as simple_eval
import engine.minimax as minimax
import time


def handle_args():
    parser = argparse.ArgumentParser(description="Chess Engine")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",  # Counts occurrences of -v
        default=0,  # Default verbosity level
        help="Increase output verbosity. -v for INFO, -vv for DEBUG.",
    )
    args = parser.parse_args()
    return args


class TimeoutException(Exception):
    pass


def run_minimax_with_timeout(mm, depth, result_queue, timeout_seconds):
    """Run minimax search in a separate thread with timeout."""
    try:
        score, move = mm.find_top_move(depth=depth)
        result_queue.put(('success', score, move))
    except Exception as e:
        result_queue.put(('error', str(e), None))


def benchmark_minimax():
    """Benchmark different Minimax configurations on multiple games."""
    logger = logging.getLogger(__name__)
    
    # Test configurations
    configs = [
        {
            'name': 'Default (Alpha-Beta + Move Ordering + PVS + Zobrist)',
            'params': {}
        },
        {
            'name': 'No Alpha-Beta',
            'params': {'use_alpha_beta': False}
        },
        {
            'name': 'No Move Ordering',
            'params': {'use_move_ordering': False}
        },
        {
            'name': 'No PVS',
            'params': {'use_pvs': False}
        },
        {
            'name': 'No Zobrist',
            'params': {'use_zobrist': False}
        },
        {
            'name': 'Alpha-Beta Only',
            'params': {'use_move_ordering': False, 'use_pvs': False, 'use_zobrist': False}
        },
        {
            'name': 'Minimal (No Optimizations)',
            'params': {'use_alpha_beta': False, 'use_move_ordering': False, 'use_pvs': False, 'use_zobrist': False}
        }
    ]
    
    depth = 6
    num_games = 3
    timeout_seconds = 100
    
    print("=" * 80)
    print(f"MINIMAX BENCHMARK - Depth {depth}, {num_games} Games, {timeout_seconds}s Timeout")
    print("=" * 80)
    
    # Generate test boards
    test_boards = []
    for i in range(num_games):
        try:
            board = load_games.random_board()
            if board:
                test_boards.append(board)
                logger.debug(f"Loaded test board {i+1}: {board.fen()}")
        except Exception as e:
            logger.error(f"Error loading test board {i+1}: {e}")
    
    if not test_boards:
        logger.error("No test boards could be loaded")
        return
    
    print(f"Testing {len(test_boards)} game positions:")
    for i, board in enumerate(test_boards):
        print(f"  Game {i+1}: {board.fen()[:50]}...")
    print()
    
    # Results storage
    results = {}
    
    # Run benchmark for each configuration
    for config in configs:
        config_name = config['name']
        config_params = config['params']
        
        print(f"Testing: {config_name}")
        print("-" * 60)
        
        results[config_name] = {
            'times': [],
            'moves': [],
            'scores': [],
            'total_time': 0,
            'avg_time': 0,
            'failed': 0,
            'timeouts': 0
        }
        
        for i, board in enumerate(test_boards):
            try:
                # Create evaluator for this board
                evaluator = simple_eval.SimpleEval(board.copy())
                
                # Create minimax with specified configuration
                mm = minimax.Minimax(board.copy(), evaluator, **config_params)
                
                # Set up threading-based timeout
                result_queue = queue.Queue()
                search_thread = threading.Thread(
                    target=run_minimax_with_timeout,
                    args=(mm, depth, result_queue, timeout_seconds)
                )
                
                # Time the search
                start_time = time.time()
                search_thread.start()
                search_thread.join(timeout=timeout_seconds)
                end_time = time.time()
                search_time = end_time - start_time
                
                if search_thread.is_alive():
                    # Timeout occurred
                    results[config_name]['timeouts'] += 1
                    print(f"  Game {i+1}: TIMEOUT after {search_time:.1f}s")
                    # Note: Thread will continue running in background, but we move on
                else:
                    # Thread completed, get result
                    try:
                        result_type, result_data, move = result_queue.get_nowait()
                        if result_type == 'success':
                            score = result_data
                            results[config_name]['times'].append(search_time)
                            results[config_name]['moves'].append(str(move))
                            results[config_name]['scores'].append(score)
                            results[config_name]['total_time'] += search_time
                            print(f"  Game {i+1}: Move={move}, Score={score:.2f}, Time={search_time:.3f}s")
                        else:
                            # Error occurred
                            results[config_name]['failed'] += 1
                            print(f"  Game {i+1}: FAILED - {result_data}")
                    except queue.Empty:
                        results[config_name]['failed'] += 1
                        print(f"  Game {i+1}: FAILED - No result returned")
                
            except Exception as e:
                logger.error(f"Error in {config_name} for game {i+1}: {e}")
                results[config_name]['failed'] += 1
                print(f"  Game {i+1}: FAILED - {e}")
        
        # Calculate averages
        if results[config_name]['times']:
            results[config_name]['avg_time'] = results[config_name]['total_time'] / len(results[config_name]['times'])
        else:
            results[config_name]['avg_time'] = float('inf')  # All timeouts or failures
        
        print(f"  Total Time: {results[config_name]['total_time']:.3f}s")
        print(f"  Average Time: {results[config_name]['avg_time']:.3f}s")
        print(f"  Timeouts: {results[config_name]['timeouts']}")
        print(f"  Failed: {results[config_name]['failed']}")
        print()
    
    # Summary table
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<40} {'Avg Time (s)':<12} {'Total Time (s)':<14} {'Timeouts':<10} {'Failed':<8}")
    print("-" * 80)
    
    # Sort by average time (put infinite times at the end)
    sorted_configs = sorted(configs, key=lambda x: (results[x['name']]['avg_time'] == float('inf'), results[x['name']]['avg_time']))
    
    for config in sorted_configs:
        name = config['name']
        avg_time = results[name]['avg_time']
        total_time = results[name]['total_time']
        timeouts = results[name]['timeouts']
        failed = results[name]['failed']
        
        avg_time_str = f"{avg_time:.3f}" if avg_time != float('inf') else "TIMEOUT"
        print(f"{name:<40} {avg_time_str:<12} {total_time:<14.3f} {timeouts:<10} {failed:<8}")
    
    # Performance comparison (only for non-timeout results)
    valid_configs = [c for c in sorted_configs if results[c['name']]['avg_time'] != float('inf')]
    if len(valid_configs) > 1:
        fastest = valid_configs[0]['name']
        slowest = valid_configs[-1]['name']
        speedup = results[slowest]['avg_time'] / results[fastest]['avg_time']
        
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
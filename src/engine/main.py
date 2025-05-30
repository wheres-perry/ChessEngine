import argparse
import io_utils.load_games as load_games
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import logging

import chess.pgn
import src.io_utils.to_tensor as to_tensor

import engine.evaluators.simple_eval as simple_eval
import engine.evaluators.simple_nn_eval as nn_eval
import engine.evaluators.deep_cnn_eval as deep_cnn_eval
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

    try:
        b = load_games.random_board()
    except Exception as e:
        logger.error(f"Error loading random board: {e}")
        return
    logger.debug(f"Random board loaded: {b}")

    if not b:
        logger.error("Failed to load a valid board.")
        return
    ev = simple_eval.SimpleEval(b)
    logger.debug(f"SimpleEval initialized with board: {b}")
    try:
        score = ev.basic_evaluate()
    except Exception as e:
        logger.error(f"Error evaluating board: {e}")
        return
    print(f"Simple eval score: {score}")

    # NN Eval
    model_path = 'data/models/simple_nn.pth'
    nn_evaluator = nn_eval.NN_Eval(b, model_path=model_path)
    nn_score = nn_evaluator.evaluate()
    print(f"NN eval score: {nn_score}")

    # Deep CNN Eval - Lightweight
    deep_cnn_light = deep_cnn_eval.DeepCNN_Eval(
        b, 
        architecture="lightweight",
        num_residual_blocks=2,
        base_channels=64
    )
    deep_cnn_score_light = deep_cnn_light.evaluate()
    print(f"Deep CNN (lightweight) eval score: {deep_cnn_score_light}")
    print(f"Deep CNN (lightweight) model info: {deep_cnn_light.get_model_info()}")

    # Deep CNN Eval - Full
    deep_cnn_full = deep_cnn_eval.DeepCNN_Eval(
        b,
        architecture="deep", 
        num_residual_blocks=4,
        base_channels=128
    )
    deep_cnn_score_full = deep_cnn_full.evaluate()
    print(f"Deep CNN (full) eval score: {deep_cnn_score_full}")
    print(f"Deep CNN (full) model info: {deep_cnn_full.get_model_info()}")

    if b:
        try:
            # Using Simple Eval without Zobrist
            start_time = time.time()
            mm_simple = minimax.Minimax(b, ev, use_zobrist=False)
            score_simple, move_simple = mm_simple.find_top_move(depth=6)
            time_simple = time.time() - start_time
            print(f"Simple Eval Minimax (no Zobrist): Best Move {move_simple}, Score: {score_simple}, Time: {time_simple:.2f}s")

            # Using Simple Eval with Zobrist
            start_time = time.time()
            mm_simple_z = minimax.Minimax(b, ev, use_zobrist=True)
            score_simple_z, move_simple_z = mm_simple_z.find_top_move(depth=6)
            time_simple_z = time.time() - start_time
            print(f"Simple Eval Minimax (with Zobrist): Best Move {move_simple_z}, Score: {score_simple_z}, Time: {time_simple_z:.2f}s")

            # Using NN Eval without Zobrist
            start_time = time.time()
            mm_nn = minimax.Minimax(b, nn_evaluator, use_zobrist=False)
            score_nn, move_nn = mm_nn.find_top_move(depth=6)
            time_nn = time.time() - start_time
            print(f"NN Eval Minimax (no Zobrist): Best Move {move_nn}, Score: {score_nn}, Time: {time_nn:.2f}s")

            # Using NN Eval with Zobrist
            start_time = time.time()
            mm_nn_z = minimax.Minimax(b, nn_evaluator, use_zobrist=True)
            score_nn_z, move_nn_z = mm_nn_z.find_top_move(depth=6)
            time_nn_z = time.time() - start_time
            print(f"NN Eval Minimax (with Zobrist): Best Move {move_nn_z}, Score: {score_nn_z}, Time: {time_nn_z:.2f}s")

            print(b)
            print("PGN of current position:")
            print(chess.pgn.Game.from_board(b))

        except Exception as e:
            logger.error(f"Error evaluating board: {e}")
            return

    if b:
        try:
            tensor = to_tensor.create_tensor(b)
            print(f"Tensor shape: {tensor.shape}")
            print(tensor)
        except Exception as e:
            logger.error(f"Error creating tensor: {e}")
            return


if __name__ == "__main__":
    main()
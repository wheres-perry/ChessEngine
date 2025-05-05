import random
import chess
import eval
import load_games
import argparse
import logging


b = load_games.random_board()
e = eval.SimpleEval(b)

print(b)


def handle_args():
    parser = argparse.ArgumentParser(description="Chess Engine")
    parser.add_argument(
        '-v', '--verbose',
        action='count',  # Counts occurrences of -v
        default=0,       # Default verbosity level
        help="Increase output verbosity. -v for INFO, -vv for DEBUG."
    )
    args = parser.parse_args()
    return args


def main():

    args = handle_args()
    verbosity = args.verbose

    # Set up logging based on verbosity level
    log_level = logging.WARNING
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

    e = eval.SimpleEval(b)
    logger.debug(f"SimpleEval initialized with board: {b}")
    try:
        score = e.basic_evaluate()
    except Exception as e:
        logger.error(f"Error evaluating board: {e}")
        return

    print(f"Score: {score}")


if __name__ == "__main__":
    main()

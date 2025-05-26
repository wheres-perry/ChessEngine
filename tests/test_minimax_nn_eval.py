import chess
import logging
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engine.minimax import Minimax
from src.engine.evaluators.simple_nn_eval import NN_Eval, ChessNN

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s') 
logger = logging.getLogger(__name__)

def test_minimax_with_nn_eval():
    """
    Tests the Minimax algorithm with NN_Eval to a specified depth.
    """
    fen = "5r2/1pB2rbk/6pn/4n2q/P3B3/1P5P/4R1P1/2Q2R1K b - - 3 33"
    board = chess.Board(fen)
    logger.info(f"Board initialized with FEN: {fen}") 
    

    model_path = os.path.join(project_root, "notebooks", "model_training", "best_chess_nn.pth")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}. Please ensure the model is trained and the path is correct.")
        logger.error("Skipping test_minimax_with_nn_eval_depth_10.")

        return

    logger.info(f"Loading NN model from: {model_path}")
    try:
        nn_evaluator = NN_Eval(board=board, model_path=model_path)
        logger.info("NN_Eval initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize NN_Eval: {e}")
        return

    minimax_searcher = Minimax(board=board, evaluator=nn_evaluator)
    
    depth = 7
    logger.info(f"Starting Minimax search with NN_Eval to depth {depth} for {'White' if board.turn else 'Black'}'s turn.") 
    
    start_time = time.time()
    
    try:
        best_score, best_move = minimax_searcher.find_top_move(depth=depth)
        end_time = time.time()
        
        time_taken = end_time - start_time
        print(f"Search completed in {time_taken:.2f} seconds.") 
        logger.info(f"Search completed in {time_taken:.2f} seconds.") 
        logger.info(f"Total nodes visited: {minimax_searcher.node_count}") 
        if best_move:
            print(f"Best move found: {board.san(best_move)} with score: {best_score:.4f}")
            logger.info(f"Best move found: {board.san(best_move)} with score: {best_score:.4f}") 
        else:
            print(f"No legal moves found or game already over. Score: {best_score:.4f}")
            logger.info(f"No legal moves found or game already over. Score: {best_score:.4f}") 
            if board.is_checkmate():
                print(f"Checkmate! {'Black' if board.turn else 'White'} wins.")
                logger.info(f"Checkmate! {'Black' if board.turn else 'White'} wins.") 
            elif board.is_stalemate():
                print("Stalemate!")
                logger.info("Stalemate!") 
            elif board.is_insufficient_material():
                print("Draw due to insufficient material.")
                logger.info("Draw due to insufficient material.") 
            elif board.is_seventyfive_moves():
                print("Draw due to 75-move rule.")
                logger.info("Draw due to 75-move rule.") 
            elif board.is_fivefold_repetition():
                print("Draw due to fivefold repetition.")
                logger.info("Draw due to fivefold repetition.") 


    except Exception as e:
        logger.error(f"An error occurred during Minimax search: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Running Minimax with NN_Eval test") 
    print("Running Minimax with NN_Eval test.") 
    test_minimax_with_nn_eval()
    logger.info("Test finished.") 
    print("Test finished.") 
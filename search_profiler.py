import logging
import chess
import chess.engine
import os
from pathlib import Path
import src.engine.search.minimax as minimax
import src.engine.evaluators.mock_eval as mock_eval
from src.engine.config import EngineConfig, MinimaxConfig

logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EngineProfiler:
    """Simplified chess engine profiler with node count tracking only."""
    
    def __init__(self):
        self.depth = 6
        self.test_board = chess.Board()
        self.config_totals = {}
        
        # 2 configurations to test against; all optimizations or none
        self.configs = {
            "Base (No Optimizations)": MinimaxConfig(
                use_zobrist=False, use_alpha_beta=False, use_move_ordering=False,
                use_iddfs=False, use_pvs=False, use_lmr=False, use_tt_aging=False
            ),
            "All Optimizations": MinimaxConfig(
                use_zobrist=True, use_alpha_beta=True, use_move_ordering=True,
                use_iddfs=True, use_pvs=True, use_lmr=True, use_tt_aging=True
            )
        }
        
        # Initialize running totals for each config
        for config_name in self.configs:
            self.config_totals[config_name] = 0

    def add_config(self, name: str, conf: MinimaxConfig):
        self.configs[name] = conf
        self.config_totals[name] = 0

    def run_board(self, name: str, config: MinimaxConfig, game_length: int) -> dict:
        """Run a single benchmark test."""
        logger.info(f"Testing {name}")
        
        board = self.test_board.copy()
        evaluator = mock_eval.MockEval(board)
        engine = minimax.Minimax(board, evaluator, EngineConfig(minimax=config))
        
        try:
            with chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish") as sf_engine:
                for _ in range(game_length):
                    # Check if game is over
                    if board.is_game_over():
                        logger.info(f"Game ended: {board.result()}")
                        break
                        
                    score, move = engine.find_top_move(depth=self.depth)
                    if move is not None:
                        board.push(move)
                    else:
                        logger.warning(f"Engine returned None move for {name}")
                        break
                    
                    # Check again after engine move
                    if board.is_game_over():
                        logger.info(f"Game ended after engine move: {board.result()}")
                        break
                    
                    response = sf_engine.play(board, chess.engine.Limit(time=5.0))
                    if response.move is not None:
                        board.push(response.move)
                    else:
                        logger.warning(f"Stockfish returned None move for {name}")
                        break
                    
                # Get node count after the loop, with default fallback
                node_count = getattr(engine, 'node_count', 0)
                self.config_totals[name] += node_count
                
            return {
                'name': name,
                'success': True,
                'nodes': node_count,
            }
        except Exception as e:
            logger.error(f"Error in {name}: {e}")
            return {'name': name, 'success': False, 'nodes': 0, 'error': str(e)}

    def run_board_all_configs(self, game_length: int):
        """Run all benchmarks and print results."""
        logger.info(f"Starting benchmark suite (depth={self.depth}) for position: {self.test_board.fen()}")
        for name, config in self.configs.items():
            print(f"Starting eval with \"{name}\" config")
            self.run_board(name, config, game_length) 
        

    def profile(self, board_list, game_length: int):
        """Run benchmarks on multiple chess boards."""
        # Game length signifies how back and forth moves, important for displaying benefits of zobrists
        for config_name in self.configs:
            self.config_totals[config_name] = 0
        
        for i, board in enumerate(board_list):            
            self.test_board = board
            print(f"Starting evaluation for board {i}")
            self.run_board_all_configs(game_length=game_length)
        
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY - Total nodes across all boards:")
        print(f"{'='*80}")
        for config_name, total in self.config_totals.items():
            print(f"{config_name:<35} {total:,} nodes")

def main():
    """Main function to run the benchmarks."""
    print("Chess Engine Performance Profiler")
    print("=" * 50)
    print("Testing config against all optimizations and no optimizations")
    print("This will take a few minutes...\n")
    
    fen_path = Path("data/raw/example_fens")
    fens = ["M01.fen", "HardM2.fen", "M10.fen"]
    fen_files = [fen_path / fen for fen in fens]
    boards = []
    for fen_file in fen_files:
        with open(fen_file, 'r') as f:
            fen = f.read().strip()
            boards.append(chess.Board(fen))

    profiler = EngineProfiler()
    profiler.add_config(
        "Test Config",
        MinimaxConfig(
            use_zobrist=True, use_alpha_beta=True, use_move_ordering=True,
            use_iddfs=True, use_pvs=True, use_lmr=True, use_tt_aging=True
        )
    )
    
    profiler.profile(boards, game_length=2)

if __name__ == "__main__":
    main()

import logging
import chess
import chess.engine
import os
from pathlib import Path
import src.engine.search.minimax as minimax
import src.engine.evaluators.simple_eval as simple_eval
from src.engine.config import EngineConfig, MinimaxConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_fen_boards(fen_dir="data/raw/example_fens"):
    """Load chess boards from .fen files in the specified directory."""
    boards = []
    fen_path = Path(fen_dir)
    
    if not fen_path.exists():
        logger.warning(f"FEN directory {fen_path} does not exist")
        return boards
    
    # Only process M01.fen through M06.fen
    for i in range(7, 12):
        fen_file = fen_path / f"M{i:02d}.fen"
        if fen_file.exists():
            try:
                with open(fen_file, 'r') as f:
                    fen_string = f.read().strip()
                    board = chess.Board(fen_string)
                    boards.append(board)
                    logger.info(f"Loaded board from {fen_file.name}")
            except Exception as e:
                logger.error(f"Error loading {fen_file.name}: {e}")
        else:
            logger.warning(f"File {fen_file.name} not found")
    
    return boards

class EngineProfiler:
    """Simplified chess engine profiler with node count tracking only."""
    
    def __init__(self):
        self.depth = 7
        self.test_board = chess.Board("8/P3n3/pp6/p3P3/k1P1p2n/Pp2p3/1P2PpBp/3b1K2 w - - 0 1")
        self.config_totals = {}
        
        # 4 key configurations to isolate performance issues
        self.configs = {
            "Base (No Optimizations)": MinimaxConfig(
                use_zobrist=False, use_alpha_beta=False, use_move_ordering=False,
                use_iddfs=False, use_pvs=False, use_lmr=False, use_tt_aging=False
            ),
            "Alpha-Beta Only": MinimaxConfig(
                use_zobrist=False, use_alpha_beta=True, use_move_ordering=False,
                use_iddfs=False, use_pvs=False, use_lmr=False, use_tt_aging=False
            ),
            "All Optimizations (No TT Aging)": MinimaxConfig(
                use_zobrist=True, use_alpha_beta=True, use_move_ordering=True,
                use_iddfs=True, use_pvs=True, use_lmr=True, use_tt_aging=False
            ),
            "All Optimizations": MinimaxConfig(
                use_zobrist=True, use_alpha_beta=True, use_move_ordering=True,
                use_iddfs=True, use_pvs=True, use_lmr=True, use_tt_aging=True
            ),
        }
        
        # Initialize running totals for each config
        for config_name in self.configs:
            self.config_totals[config_name] = 0

    def run_benchmark(self, name: str, config: MinimaxConfig, game_length: int) -> dict:
        """Run a single benchmark test."""
        logger.info(f"Testing {name}")
        
        board = self.test_board.copy()
        evaluator = simple_eval.SimpleEval(board)
        engine = minimax.Minimax(board, evaluator, EngineConfig(minimax=config))
        
        try:
            with chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish") as sf_engine:
                for _ in range(game_length):
                    score, move = engine.find_top_move(depth=self.depth)
                    board.push(move)
                    response = sf_engine.play(board, chess.engine.Limit(time=5.0))
                    board.push(move)
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

    def run_all(self, game_length: int):
        """Run all benchmarks and print results."""
        logger.info(f"Starting benchmark suite (depth={self.depth}) for position: {self.test_board.fen()}")
        
        results = [self.run_benchmark(name, config, game_length) for name, config in self.configs.items()]
        

    def run_multiple_games(self, board_list, game_length: int):
        """Run benchmarks on multiple chess boards."""
        # Reset totals
        for config_name in self.configs:
            self.config_totals[config_name] = 0
        
        for i, board in enumerate(board_list, 1):            
            self.test_board = board
            self.run_all(game_length=game_length)
        
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY - Total nodes across all boards:")
        print(f"{'='*80}")
        for config_name, total in self.config_totals.items():
            print(f"{config_name:<35} {total:,} nodes")

def main():
    """Main function to run the benchmarks."""
    print("Chess Engine Performance Profiler")
    print("=" * 50)
    print("Testing 4 key configurations at depth 7")
    print("This will take a few minutes...\n")
    
    profiler = EngineProfiler()
    
    board_list = load_fen_boards()
    
    if board_list:
        profiler.run_multiple_games(board_list, game_length=3)
    else:
        raise ValueError("No FEN files found. Unable to run benchmarks.")

if __name__ == "__main__":
    main()

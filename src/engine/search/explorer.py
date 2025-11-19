"""
Modular Explorer search engine.

Alternative search implementation that can be configured with various
optimizations while respecting Tree 1 (Move Exploration) dependencies.

Similar to Minimax but potentially with different search strategies.
"""

import logging

from engine._core import chess_engine_core as chess
from src.engine.config import EngineConfig
from src.engine.config_dependency_resolver import DependencyResolver
from src.engine.evaluators.base_evaluator import BaseEvaluator
from src.engine.search.move_ordering import MoveOrderer
from src.engine.search.transposition_table import TranspositionTable
from src.engine.search.zobrist import Zobrist

logger = logging.getLogger(__name__)

# pylint: disable=too-many-instance-attributes


class Explorer:
    """
    Modular explorer search engine.

    Supports adaptive control flow based on configuration flags:
    - Pure minimax when use_alpha_beta is False
    - Alpha-beta pruning when enabled
    - Iterative deepening when use_iddfs is True
    - Transposition tables when use_zobrist and use_transposition_table are True
    - Move ordering when use_move_ordering is True

    Similar to Minimax but provides an alternative implementation for
    experimentation and comparison.
    """

    NEG_INF = float("-inf")
    POS_INF = float("inf")
    DEFAULT_TT_SIZE = 10000
    TIME_CHECK_INTERVAL = 10000

    def __init__(
        self,
        board: chess.Board,
        evaluator: BaseEvaluator,
        config: EngineConfig,
    ) -> None:
        """
        Initialize the modular explorer search engine.

        Args:
            board: Chess board to search from
            evaluator: Position evaluator (BaseEvaluator subclass)
            config: Engine configuration (validated for dependencies)
        """
        # Resolve and validate dependencies
        resolver = DependencyResolver(config)
        try:
            self.config = resolver.resolve()
        except Exception as e:
            logger.error("Error resolving dependencies: %s", e)
            raise

        # Store full engine config for access to evaluation config
        self.engine_config = config

        # Core components
        self.board = board
        self.evaluator = evaluator

        # Extract search configuration flags
        self.use_minimax = self.config.use_minimax
        self.use_alpha_beta = self.config.use_alpha_beta
        self.use_iddfs = self.config.use_iddfs
        self.use_move_ordering = self.config.use_move_ordering
        self.use_transposition_table = self.config.use_transposition_table
        self.use_zobrist = self.config.use_zobrist

        # Initialize Zobrist hashing and transposition table (if enabled)
        self.zobrist: Zobrist | None = None
        self.transposition_table: TranspositionTable | None = None
        if self.use_zobrist and self.use_transposition_table:
            self.zobrist = Zobrist()
            self.transposition_table = TranspositionTable(
                self.DEFAULT_TT_SIZE, use_tt_aging=self.config.use_tt_aging
            )
            # Initialize hash for the starting position
            self.zobrist.hash_board(self.board)

        # Initialize move orderer (if enabled)
        self.move_orderer: MoveOrderer | None = None
        if self.use_move_ordering:
            self.move_orderer = MoveOrderer(
                self.board, self.config, self.zobrist, self.transposition_table
            )

        # Search state
        self.nodes_searched = 0
        self.time_up = False

    def search(self, depth: int = 1) -> tuple[float | None, chess.Move | None]:
        """
        Search for the best move using modular exploration.

        Args:
            depth: Maximum search depth

        Returns:
            Tuple of (evaluation_score, best_move)
        """
        if not self.use_minimax:
            logger.warning("Minimax is disabled, cannot search")
            return None, None

        # Initialize search
        self.nodes_searched = 0
        self.time_up = False

        # Choose search strategy based on configuration
        if self.use_iddfs and depth > 1:
            return self._iterative_deepening_search(depth)
        return self._fixed_depth_search(depth)

    def _iterative_deepening_search(
        self, max_depth: int
    ) -> tuple[float | None, chess.Move | None]:
        """Iterative deepening search from depth 1 to max_depth."""
        best_score: float | None = None
        best_move: chess.Move | None = None

        for current_depth in range(1, max_depth + 1):
            score, move = self._fixed_depth_search(current_depth)

            if move is not None:
                best_score = score
                best_move = move

            logger.debug("Depth %s: score=%s, move=%s", current_depth, score, move)

        return best_score, best_move

    def _fixed_depth_search(self, depth: int) -> tuple[float, chess.Move | None]:
        """Fixed-depth search returning score and best move."""
        # Initialize alpha-beta window
        alpha = self.NEG_INF
        beta = self.POS_INF

        # Get all legal moves
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return 0.0, None

        # Order moves (if enabled)
        if self.use_move_ordering and self.move_orderer:
            ordered_moves = self.move_orderer.order_moves(legal_moves, depth=depth)
        else:
            ordered_moves = legal_moves

        # Track best move
        best_move: chess.Move | None = None
        is_maximizing = self.board.turn == chess.WHITE
        best_score = self.NEG_INF if is_maximizing else self.POS_INF

        # Search each move
        for move in ordered_moves:
            # Make the move
            self.board.push(move)

            # Recursively search
            score = self._search_recursive(
                depth - 1, alpha, beta, maximizing_player=is_maximizing
            )

            # Undo the move
            self.board.pop()

            # Update best move
            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)

            # Alpha-beta cutoff (if enabled)
            if self.use_alpha_beta and alpha >= beta:
                break

        return best_score, best_move

    def _search_recursive(
        self, depth: int, alpha: float, beta: float, maximizing_player: bool
    ) -> float:
        """Recursive search with modular control flow (minimax/alpha-beta)."""
        # Count nodes
        self.nodes_searched += 1

        # Base case: depth 0 or game over
        if depth == 0 or self.board.is_game_over():
            return self.evaluator.evaluate()

        # Check transposition table (if enabled)
        if self.use_transposition_table and self.zobrist and self.transposition_table:
            position_hash = self.zobrist.get_current_hash()
            if position_hash is not None:
                tt_score = self.transposition_table.lookup(
                    position_hash, depth, alpha, beta
                )
                if tt_score is not None:
                    return tt_score

        # Get and order moves
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return self.evaluator.evaluate()

        if self.use_move_ordering and self.move_orderer:
            ordered_moves = self.move_orderer.order_moves(legal_moves, depth=depth)
        else:
            ordered_moves = legal_moves

        # Search moves
        return self._minimax_search(
            ordered_moves, depth, alpha, beta, maximizing_player
        )

    def _minimax_search(
        self,
        moves: list[chess.Move],
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
    ) -> float:
        """Core minimax logic extracted to reduce complexity."""
        if maximizing_player:
            max_eval = self.NEG_INF
            for move in moves:
                self.board.push(move)
                eval_score = self._search_recursive(
                    depth - 1, alpha, beta, maximizing_player=False
                )
                self.board.pop()

                max_eval = max(max_eval, eval_score)
                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            return max_eval

        min_eval = self.POS_INF
        for move in moves:
            self.board.push(move)
            eval_score = self._search_recursive(
                depth - 1, alpha, beta, maximizing_player=True
            )
            self.board.pop()

            min_eval = min(min_eval, eval_score)
            if self.use_alpha_beta:
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
        return min_eval

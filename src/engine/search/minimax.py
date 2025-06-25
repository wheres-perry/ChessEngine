import chess
import logging
import time

from src.engine.config import EngineConfig
from src.engine.constants import PIECE_VALUES
from src.engine.evaluators.eval import Eval
from src.engine.search.transposition_table import TranspositionTable
from src.engine.search.zobrist import Zobrist
from src.engine.search.move_ordering import MoveOrderer

logger = logging.getLogger(__name__)

class Minimax:
    """
    Minimax search engine with alpha-beta pruning, transposition tables, and iterative deepening.
    Implements a chess engine that searches for the best move using minimax algorithm
    with various optimizations including move ordering, zobrist hashing, and time management.
    """

    NEG_INF = float("-inf")
    POS_INF = float("inf")
    DEFAULT_TT_SIZE = 10000
    TIME_CHECK_INTERVAL = 10000
    
    # LMR parameters
    LMR_MIN_DEPTH = 2
    LMR_MIN_MOVES = 2
    LMR_REDUCTION = 2

    def __init__(
        self,
        board: chess.Board,
        evaluator: Eval,
        config: EngineConfig,
    ):
        """
        Initialize the minimax search engine.

        Args:
            board: Chess board position to search from
            evaluator: Position evaluation function
            config: Engine configuration (assumed to be pre-validated)
        """
        # Extract 
        minimax_config = config.minimax
        self.use_zobrist = minimax_config.use_zobrist
        self.use_iddfs = minimax_config.use_iddfs
        self.use_alpha_beta = minimax_config.use_alpha_beta
        self.use_move_ordering = minimax_config.use_move_ordering
        self.use_pvs = minimax_config.use_pvs
        self.use_tt_aging = minimax_config.use_tt_aging
        self.use_lmr = minimax_config.use_lmr
        self.max_time = minimax_config.max_time

        # Initialize Zobrist hashing and transposition table
        if self.use_zobrist:
            self.zobrist = Zobrist()
            self.transposition_table = TranspositionTable(
                self.DEFAULT_TT_SIZE, use_tt_aging=self.use_tt_aging
            )
        else:
            self.zobrist = None
            self.transposition_table = None

        self.board = board
        self.evaluator = evaluator
        
        # Initialize the move orderer
        self.move_orderer = MoveOrderer(
            self.board, 
            self.zobrist, 
            self.transposition_table
        )

        # Initialize hash for the starting position
        if self.zobrist:
            self.zobrist.hash_board(self.board)

        # Initialize hash stack for efficient incremental updates
        self.hash_stack = []
        self.pv_move = None  # Current PV move for this position

    def find_top_move(self, depth: int = 1) -> tuple[None | float, None | chess.Move]:
        """
        Find the best move for the current position.

        Args:
            depth: Maximum search depth

        Returns:
            Tuple of (evaluation_score, best_move)
        """
        # FIXED: Use consistent node counting
        self.nodes_searched = 0
        self.node_count = 0  # For backward compatibility with tests
        self.time_up = False
        self.start_time = time.time()
        self.best_move_first = None

        # FIXED: Only increment age ONCE per search, not per depth
        if self.use_zobrist and self.transposition_table and self.use_tt_aging:
            self.transposition_table.increment_age()

        if self.use_iddfs and depth > 1:
            result = self._iterative_deepening(depth)
        else:
            result = self._search_fixed_depth(depth)
        
        # FIXED: Update node_count for test compatibility
        self.node_count = self.nodes_searched
        return result

    def _check_time_limit(self) -> bool:
        """Check if the allocated search time has been exceeded."""
        if (
            self.max_time
            and self.start_time
            and time.time() - self.start_time >= self.max_time
        ):
            self.time_up = True
            return True
        return False

    def _iterative_deepening(
        self, max_depth: int
    ) -> tuple[None | float, None | chess.Move]:
        """
        Perform iterative deepening search from depth 1 to max_depth.
        Each iteration provides a better move estimate and enables early termination
        when time runs out while maintaining the best move found so far.

        Args:
            max_depth: Maximum depth to search to

        Returns:
            Tuple of (best_score, best_move) from deepest completed iteration
        """
        best_score = None
        best_move = None

        for current_depth in range(1, max_depth + 1):
            if self._check_time_limit():
                break

            score, move = self._search_fixed_depth(current_depth)

            if self.time_up:
                break

            if move is not None:
                best_score = score
                best_move = move
                self.best_move_first = move

        return best_score, best_move

    def _search_fixed_depth(self, depth: int) -> tuple[float, None | chess.Move]:
        """
        Search to a fixed depth and return the best move.

        Args:
            depth: Depth to search to

        Returns:
            (score, best_move) tuple
        """
        # Only reset nodes if not using IDDFS (to allow accumulation)
        if not self.use_iddfs:
            self.nodes_searched = 0
        
        self.start_time = time.time()

        # REMOVED: This was clearing TT every search, destroying performance  
        # if self.use_zobrist and self.transposition_table and self.transposition_table.use_tt_aging:
        #     self.transposition_table.reset_age()

        # Initial hash for the root position
        if self.zobrist:
            self.zobrist.hash_board(self.board)

        alpha = self.NEG_INF
        beta = self.POS_INF

        # Get all legal moves
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return 0.0, None

        # Track best move
        best_move = None
        best_score = self.NEG_INF if self.board.turn == chess.WHITE else self.POS_INF

        # Order moves at the root
        ordered_moves = self.order_moves(legal_moves)

        # Search each move
        for m in ordered_moves:
            # Store current hash before making move
            if self.zobrist:
                current_hash = self.zobrist.make_move_hash(self.board, m)
                self.hash_stack.append(self.zobrist.get_current_hash())

            # Make the move
            self.board.push(m)

            # Update hash after move
            if self.zobrist:
                self.zobrist.set_current_hash(current_hash)

            # Search from this new position
            score = self.minimax_alpha_beta(depth - 1, alpha, beta, self.board.turn == chess.WHITE)

            # Save best move
            if self.board.turn == chess.BLACK:
                if score > best_score:  # Maximizing player (White)
                    best_score = score
                    best_move = m
            else:
                if score < best_score:  # Minimizing player (Black)  
                    best_score = score
                    best_move = m

            # Undo the move
            self.board.pop()

            # Restore the hash from stack
            if self.zobrist:
                restored_hash = self.hash_stack.pop()
                self.zobrist.set_current_hash(restored_hash)

        # Store the best move in the transposition table
        if self.zobrist and self.transposition_table:
            position_hash = self.zobrist.get_current_hash()
            if position_hash is not None:
                self.transposition_table.store(position_hash, depth, best_score, alpha, beta, alpha, best_move)

        return best_score, best_move

    def order_moves(self, moves: list[chess.Move]) -> list[chess.Move]:
        """
        Order moves to improve alpha-beta pruning efficiency by delegating to the move orderer.
        
        Args:
            moves: List of legal moves to order

        Returns:
            Ordered list of moves
        """
        if not self.use_move_ordering:
            return moves
        
        return self.move_orderer.order_moves(moves)

    def minimax_alpha_beta(self, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
        """
        Minimax search with alpha-beta pruning, transposition tables, and move ordering.

        Args:
            depth: Current search depth
            alpha: Alpha value (best already explored option for maximizer)
            beta: Beta value (best already explored option for minimizer)
            maximizing_player: Whether the current player is maximizing

        Returns:
            Best evaluation score for the current player
        """
        # Node counting and time limit checks
        self.nodes_searched += 1

        if self.nodes_searched % self.TIME_CHECK_INTERVAL == 0:
            if self._check_time_limit():
                return 0.0

        # Handle base cases
        if depth == 0 or self.board.is_game_over():
            return self.evaluator.evaluate()

        # Check transposition table
        tt_hit = False
        tt_move = None
        position_hash = None

        if self.zobrist and self.transposition_table:
            position_hash = self.zobrist.get_current_hash()
            if position_hash is not None:
                tt_score = self.transposition_table.lookup(position_hash, depth, alpha, beta)
                tt_move = self.transposition_table.get_best_move(position_hash)
                if tt_score is not None:
                    return tt_score

        # Get and order legal moves
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return self.evaluator.evaluate()

        ordered_moves = self.order_moves(legal_moves)

        # Initialize best move
        best_move = None
        original_alpha = alpha  # Save for TT entry type

        if maximizing_player:
            max_eval = self.NEG_INF

            for i, m in enumerate(ordered_moves):
                if self._check_time_limit():
                    break

                # Store current hash before making move
                if self.zobrist:
                    new_hash = self.zobrist.make_move_hash(self.board, m)
                    self.hash_stack.append(self.zobrist.get_current_hash())

                # Make the move
                self.board.push(m)

                # Update hash after move
                if self.zobrist:
                    self.zobrist.set_current_hash(new_hash)

                # Determine search depth
                search_depth = depth - 1

                # Late Move Reduction
                if (self.use_lmr and depth >= self.LMR_MIN_DEPTH and
                    i >= self.LMR_MIN_MOVES and
                    not self.board.is_capture(m) and not self.board.is_en_passant(m) and
                    not self.board.is_check() and not self._gives_check_fast(m)):
                    
                    reduced_depth = max(1, search_depth - self.LMR_REDUCTION)
                    eval_score = self.minimax_alpha_beta(reduced_depth, alpha, beta, False)
                    
                    # Re-search if promising
                    if eval_score > alpha:
                        eval_score = self.minimax_alpha_beta(search_depth, alpha, beta, False)
                else:
                    # Principal Variation Search
                    if self.use_pvs and i > 0:
                        # Search with zero window to see if we can improve alpha
                        eval_score = self.minimax_alpha_beta(search_depth, alpha, alpha + 1e-10, False)
                        
                        # Re-search with full window if better than alpha
                        if alpha < eval_score < beta:
                            eval_score = self.minimax_alpha_beta(search_depth, alpha, beta, False)
                    else:
                        # Regular alpha-beta
                        eval_score = self.minimax_alpha_beta(search_depth, alpha, beta, False)

                # Undo the move
                self.board.pop()

                # Restore the hash from stack
                if self.zobrist:
                    restored_hash = self.hash_stack.pop()
                    self.zobrist.set_current_hash(restored_hash)

                # Update max evaluation and best move
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = m

                # Update alpha
                alpha = max(alpha, max_eval)

                # Alpha-beta pruning
                if self.use_alpha_beta and alpha >= beta:
                    break

            # Store in transposition table
            if self.zobrist and self.transposition_table and position_hash is not None:
                self._store_tt_entry(position_hash, depth, max_eval, alpha, beta, original_alpha, best_move)

            return max_eval

        else:
            min_eval = self.POS_INF

            for i, m in enumerate(ordered_moves):
                if self._check_time_limit():
                    break

                # Store current hash before making move
                if self.zobrist:
                    new_hash = self.zobrist.make_move_hash(self.board, m)
                    self.hash_stack.append(self.zobrist.get_current_hash())

                # Make the move
                self.board.push(m)

                # Update hash after move
                if self.zobrist:
                    self.zobrist.set_current_hash(new_hash)

                # Determine search depth
                search_depth = depth - 1

                # Late Move Reduction
                if (self.use_lmr and depth >= self.LMR_MIN_DEPTH and
                    i >= self.LMR_MIN_MOVES and
                    not self.board.is_capture(m) and not self.board.is_en_passant(m) and
                    not self.board.is_check() and not self._gives_check_fast(m)):
                    
                    reduced_depth = max(1, search_depth - self.LMR_REDUCTION)
                    eval_score = self.minimax_alpha_beta(reduced_depth, beta - 1e-10, beta, True)
                    
                    # Re-search if promising
                    if eval_score < beta:
                        eval_score = self.minimax_alpha_beta(search_depth, alpha, beta, True)
                else:
                    # Principal Variation Search
                    if self.use_pvs and i > 0:
                        # Search with zero window to see if we can improve beta
                        eval_score = self.minimax_alpha_beta(search_depth, beta - 1e-10, beta, True)
                        
                        # Re-search with full window if better than beta
                        if alpha < eval_score < beta:
                            eval_score = self.minimax_alpha_beta(search_depth, alpha, beta, True)
                    else:
                        # Regular alpha-beta
                        eval_score = self.minimax_alpha_beta(search_depth, alpha, beta, True)

                # Undo the move
                self.board.pop()

                # Restore the hash from stack
                if self.zobrist:
                    restored_hash = self.hash_stack.pop()
                    self.zobrist.set_current_hash(restored_hash)

                # Update min evaluation and best move
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = m

                # Update beta
                beta = min(beta, min_eval)

                # Alpha-beta pruning
                if self.use_alpha_beta and alpha >= beta:
                    break

            # Store in transposition table
            if self.zobrist and self.transposition_table and position_hash is not None:
                self._store_tt_entry(position_hash, depth, min_eval, alpha, beta, original_alpha, best_move)

            return min_eval

    def _store_tt_entry(
        self,
        hash_val: int,
        depth: int,
        score: float,
        alpha: float,
        beta: float,
        original_alpha: float,
        best_move: chess.Move | None = None,
    ) -> None:
        """Store an entry in the transposition table with the best move."""
        if self.transposition_table:
            self.transposition_table.store(hash_val, depth, score, alpha, beta, original_alpha, best_move)
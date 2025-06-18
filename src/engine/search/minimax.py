import logging
import time

import chess
from src.engine.config import EngineConfig
from src.engine.constants import PIECE_VALUES
from src.engine.evaluators.eval import Eval

from src.engine.search.transposition_table import TranspositionTable
from src.engine.search.zobrist import Zobrist

logger = logging.getLogger(__name__)


class Minimax:
    """
    Minimax search engine with alpha-beta pruning, transposition tables, and iterative deepening.

    Implements a chess engine that searches for the best move using minimax algorithm
    with various optimizations including move ordering, zobrist hashing, and time management.
    """

    NEG_INF = float("-inf")
    POS_INF = float("inf")
    DEFAULT_TT_SIZE = 100000
    TIME_CHECK_INTERVAL = 1000
    MVV_LVA_MULTIPLIER = 10

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
            config: Engine configuration
        """
        # Extract minimax config

        minimax_config = config.minimax

        self.use_zobrist = minimax_config.use_zobrist
        self.use_iddfs = minimax_config.use_iddfs
        self.use_alpha_beta = minimax_config.use_alpha_beta
        self.use_move_ordering = minimax_config.use_move_ordering
        self.use_pvs = minimax_config.use_pvs
        self.use_tt_aging = minimax_config.use_tt_aging
        self.max_time = minimax_config.max_time

        # Initialize other members

        self.start_time = None
        self.time_up = False
        self.best_move_first = None

        # PVS requires alpha-beta; enforce this

        self.use_pvs = self.use_pvs and self.use_alpha_beta
        if minimax_config.use_pvs and not self.use_alpha_beta:
            logger.warning(
                "PVS requires alpha-beta pruning. Disabling PVS since use_alpha_beta is False."
            )
        # Initialize Zobrist hashing and transposition table

        if self.use_zobrist:
            self.zobrist = Zobrist()
            self.transposition_table = TranspositionTable(
                self.DEFAULT_TT_SIZE, self.use_tt_aging
            )
        else:
            self.zobrist = None
            self.transposition_table = None

        self.board = board
        self.evaluator = evaluator

        # Initialize hash for the starting position

        if self.zobrist:
            self.zobrist.hash_board(self.board)

    def find_top_move(self, depth: int = 1) -> tuple[None | float, None | chess.Move]:
        """
        Find the best move for the current position.

        Args:
            depth: Maximum search depth

        Returns:
            Tuple of (evaluation_score, best_move)
        """
        self.node_count = 0
        self.time_up = False
        self.start_time = time.time()
        self.best_move_first = None

        if self.use_iddfs and depth > 1:
            return self._iterative_deepening(depth)
        else:
            return self._search_fixed_depth(depth)

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
        if best_move is None:
            logger.warning("No valid moves found in iterative deepening")
        return best_score, best_move

    def _search_fixed_depth(self, depth: int) -> tuple[float, None | chess.Move]:
        """
        Search all legal moves at the root position to the specified depth.

        Args:
            depth: Fixed depth to search to

        Returns:
            Tuple of (best_score, best_move)
        """
        maximizing_player = self.board.turn
        alpha = float(self.NEG_INF)
        beta = float(self.POS_INF)
        best_move: None | chess.Move = None

        if maximizing_player:
            best_score = float(self.NEG_INF)
        else:
            best_score = float(self.POS_INF)
        legal_moves = list(self.board.legal_moves)

        # Apply move ordering if enabled

        if self.use_move_ordering:
            ordered_moves = self.order_moves(legal_moves)
        else:
            ordered_moves = legal_moves
        for i, m in enumerate(ordered_moves):
            if self._check_time_limit():
                break
            self.board.push(m)

            # Use PVS for non-first moves if enabled

            if self.use_pvs and i > 0 and self.use_alpha_beta:
                # First try with a null window around alpha

                score = self.minimax_alpha_beta(
                    depth - 1, alpha, alpha + 0.00001, not maximizing_player
                )

                # If the score might be better than alpha, do a full re-search

                if score > alpha and score < beta:
                    score = self.minimax_alpha_beta(
                        depth - 1, alpha, beta, not maximizing_player
                    )
            else:
                # Regular minimax or alpha-beta search

                score = self.minimax_alpha_beta(
                    depth - 1, alpha, beta, not maximizing_player
                )
            self.board.pop()

            if maximizing_player:
                if score > best_score:
                    best_score = score
                    best_move = m
                if self.use_alpha_beta:
                    alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = m
                if self.use_alpha_beta:
                    beta = min(beta, score)
            # Alpha-beta pruning

            if self.use_alpha_beta and beta <= alpha:
                break
        return best_score, best_move

    def order_moves(self, moves: list[chess.Move]) -> list[chess.Move]:
        """
        Order moves for optimal alpha-beta pruning efficiency.

        Prioritizes: 1) Principal variation move 2) Captures (by MVV-LVA)
        3) Checks 4) Quiet moves

        Args:
            moves: List of legal moves to order

        Returns:
            Ordered list of moves with best moves first
        """
        # If move ordering is disabled, return the original list

        if not self.use_move_ordering:
            return list(moves)
        pv_moves = []
        checks = []
        captures = []
        other_moves = []

        def capture_score(move: chess.Move) -> float:
            """Calculate MVV-LVA score for capture ordering."""
            to_square = move.to_square
            from_square = move.from_square

            if self.board.is_en_passant(move):
                victim_value = PIECE_VALUES.get(chess.PAWN, 0)
            else:
                victim_piece = self.board.piece_at(to_square)
                victim_value = (
                    PIECE_VALUES.get(victim_piece.piece_type, 0) if victim_piece else 0
                )
            aggressor_piece = self.board.piece_at(from_square)
            aggressor_value = (
                PIECE_VALUES.get(aggressor_piece.piece_type, 0)
                if aggressor_piece
                else 0
            )

            return victim_value * self.MVV_LVA_MULTIPLIER - aggressor_value

        for move in moves:
            if self.best_move_first and move == self.best_move_first:
                pv_moves.append(move)
            elif self.board.is_capture(move):
                captures.append(move)
            elif self.board.gives_check(move):
                checks.append(move)
            else:
                other_moves.append(move)
        captures.sort(key=capture_score, reverse=True)

        return pv_moves + captures + checks + other_moves

    def minimax_alpha_beta(
        self, depth: int, alpha: float, beta: float, maximizing_player: bool
    ) -> float:
        """
        Recursive minimax search with alpha-beta pruning and transposition tables.

        Args:
            depth: Remaining search depth
            alpha: Best score for maximizing player (lower bound)
            beta: Best score for minimizing player (upper bound)
            maximizing_player: True if current player is maximizing

        Returns:
            Best evaluation score for the current position
        """
        if self.node_count % self.TIME_CHECK_INTERVAL == 0 and self._check_time_limit():
            return 0.0
        original_alpha = alpha
        hash_val: None | int = None

        # Transposition table lookup

        if self.use_zobrist and self.zobrist and self.transposition_table:
            hash_val = self.zobrist.get_current_hash()
            if hash_val is None:
                hash_val = self.zobrist.hash_board(self.board)
            tt_score = self.transposition_table.lookup(hash_val, depth, alpha, beta)
            if tt_score is not None:
                return tt_score
        self.node_count += 1

        # Terminal node evaluation

        if depth == 0 or self.board.is_game_over():
            if self.board.is_checkmate():
                score = (
                    float(self.NEG_INF) if maximizing_player else float(self.POS_INF)
                )
            elif self.board.is_stalemate():
                score = 0.0
            else:
                score = self.evaluator.evaluate()
            if hash_val is not None and self.transposition_table:
                self.transposition_table.store(
                    hash_val, depth, score, alpha, beta, original_alpha
                )
            return score
        legal_moves = list(self.board.legal_moves)

        # Apply move ordering if enabled

        if self.use_move_ordering:
            ordered_moves = self.order_moves(legal_moves)
        else:
            ordered_moves = legal_moves
        # Minimax with optional alpha-beta pruning

        if maximizing_player:
            max_eval = float(self.NEG_INF)

            for i, m in enumerate(ordered_moves):
                if self._check_time_limit():
                    break
                # Store state for incremental hashing

                old_castling_rights = self.board.castling_rights
                old_ep_square = self.board.ep_square

                # Get move information before making the move

                captured_piece = self.board.piece_at(m.to_square)
                captured_piece_type = (
                    captured_piece.piece_type if captured_piece else None
                )
                was_ep = self.board.is_en_passant(m)
                ks_castle = self.board.is_kingside_castling(m)
                qs_castle = self.board.is_queenside_castling(m)

                self.board.push(m)

                # Update hash incrementally if using Zobrist

                if self.zobrist:
                    self.zobrist.update_hash_for_move(
                        self.board,
                        m,
                        old_castling_rights,
                        old_ep_square,
                        captured_piece_type,
                        was_ep,
                        ks_castle,
                        qs_castle,
                    )
                # Use PVS for non-first moves if enabled

                if self.use_pvs and i > 0 and self.use_alpha_beta and depth > 1:
                    # First try with a null window around alpha

                    eval_score = self.minimax_alpha_beta(
                        depth - 1, alpha, alpha + 0.00001, False
                    )

                    # If the score might be better than alpha, do a full re-search

                    if eval_score > alpha and eval_score < beta:
                        eval_score = self.minimax_alpha_beta(
                            depth - 1, alpha, beta, False
                        )
                else:
                    # Regular minimax or alpha-beta search

                    eval_score = self.minimax_alpha_beta(depth - 1, alpha, beta, False)
                self.board.pop()

                # Restore hash after undoing move

                if self.zobrist:
                    self.zobrist.hash_board(
                        self.board
                    )  # For now, recalculate after pop
                max_eval = max(max_eval, eval_score)

                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Beta cutoff (only if alpha-beta is enabled)
            if hash_val is not None and self.transposition_table:
                self.transposition_table.store(
                    hash_val, depth, max_eval, alpha, beta, original_alpha
                )
            return max_eval
        else:
            min_eval = float(self.POS_INF)

            for i, m in enumerate(ordered_moves):
                if self._check_time_limit():
                    break
                # Store state for incremental hashing

                old_castling_rights = self.board.castling_rights
                old_ep_square = self.board.ep_square

                # Get move information before making the move

                captured_piece = self.board.piece_at(m.to_square)
                captured_piece_type = (
                    captured_piece.piece_type if captured_piece else None
                )
                was_ep = self.board.is_en_passant(m)
                ks_castle = self.board.is_kingside_castling(m)
                qs_castle = self.board.is_queenside_castling(m)

                self.board.push(m)

                # Update hash incrementally if using Zobrist

                if self.zobrist:
                    self.zobrist.update_hash_for_move(
                        self.board,
                        m,
                        old_castling_rights,
                        old_ep_square,
                        captured_piece_type,
                        was_ep,
                        ks_castle,
                        qs_castle,
                    )
                # Use PVS for non-first moves if enabled

                if self.use_pvs and i > 0 and self.use_alpha_beta and depth > 1:
                    # First try with a null window around beta

                    eval_score = self.minimax_alpha_beta(
                        depth - 1, beta - 0.00001, beta, True
                    )

                    # If the score might be better than beta, do a full re-search

                    if eval_score < beta and eval_score > alpha:
                        eval_score = self.minimax_alpha_beta(
                            depth - 1, alpha, beta, True
                        )
                else:
                    # Regular minimax or alpha-beta search

                    eval_score = self.minimax_alpha_beta(depth - 1, alpha, beta, True)
                self.board.pop()

                # Restore hash after undoing move

                if self.zobrist:
                    self.zobrist.hash_board(
                        self.board
                    )  # For now, recalculate after pop
                min_eval = min(min_eval, eval_score)

                if self.use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha cutoff (only if alpha-beta is enabled)
            if hash_val is not None and self.transposition_table:
                self.transposition_table.store(
                    hash_val, depth, min_eval, alpha, beta, original_alpha
                )
            return min_eval

    def _store_tt_entry(
        self,
        hash_val: int,
        depth: int,
        score: float,
        alpha: float,
        beta: float,
        original_alpha: float,
    ) -> None:
        """
        Store position evaluation in transposition table with proper bound classification.

        Classifies entries as exact values, upper bounds (fail-low), or lower bounds
        (fail-high) based on alpha-beta search results.

        Args:
            hash_val: Zobrist hash of the position
            depth: Search depth for this entry
            score: Evaluation score
            alpha: Current alpha value
            beta: Current beta value
            original_alpha: Alpha value at start of search
        """
        if not self.use_zobrist or hash_val is None or not self.transposition_table:
            return
        # Delegate to the TranspositionTable's store method, which correctly
        # handles entry type determination and storage logic.

        self.transposition_table.store(
            hash_val=hash_val,
            depth=depth,
            score=score,
            alpha=alpha,
            beta=beta,
            original_alpha=original_alpha,
        )

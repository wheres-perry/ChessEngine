import logging

import chess
from src.engine.evaluators.eval import Eval
from src.engine.zobrist import Zobrist

logger = logging.getLogger(__name__)


class Minimax:
    # TODO: Store calculated line, can work up backwards from that node rather than top down approach.

    board: chess.Board
    evaluator: Eval
    node_count: int
    score: float
    alpha: float
    beta: float

    def __init__(self, board: chess.Board, evaluator: Eval, use_zobrist: bool = True):
        self.use_zobrist = use_zobrist
        if use_zobrist:
            self.zobrist = Zobrist()
            self.transposition_table = {}
        logger.debug("Minimax initialized with board: %s", board)
        self.board = board
        self.evaluator = evaluator

    def _check_time_limit(self) -> bool:
        """Check if time limit has been exceeded."""
        if self.max_time and self.start_time and time.time() - self.start_time >= self.max_time:
            self.time_up = True
            return True
        return False

    # Returns a tuple of score(float) and move

    def find_top_move(self, depth: int = 1) -> tuple[float, chess.Move | None]:
        logger.info("Finding best move for ", self.board.turn.__str__())
        self.node_count = 0
        self.time_up = False
        self.start_time = time.time()
        
        if self.use_iddfs and depth > 1:
            return self._iterative_deepening(depth)
        else:
            return self._search_fixed_depth(depth)
    
    def _iterative_deepening(self, max_depth: int) -> tuple[float, chess.Move | None]:
        best_score = None
        best_move = None
        
        # Start from depth 1 and increase gradually
        for current_depth in range(1, max_depth + 1):
            # Check if we've run out of time before starting this depth
            if self._check_time_limit():
                logger.info(f"Time limit reached before starting depth {current_depth}")
                break
                
            score, move = self._search_fixed_depth(current_depth)
            
            # If time is up during the search, don't use this result
            if self.time_up:
                logger.info(f"Time limit reached during depth {current_depth} search")
                break
                
            # Update best move - always accept the first valid result
            if move is not None:
                best_score = score
                best_move = move
                
            logger.info(f"Completed depth {current_depth} search: best move {move} with score {score}")
            
        if best_move is None:
            logger.warning("No valid moves found in iterative deepening")
            
        return best_score, best_move
    
    def _search_fixed_depth(self, depth: int) -> tuple[float, chess.Move | None]:
        maximizing_player = self.board.turn
        alpha = float(-2147483648)
        beta = float(2147483647)
        best_move: chess.Move | None = None

        if maximizing_player:
            best_score = float(-2147483648)
        else:
            best_score = float(2147483647)
        for m in self.order_moves(list(self.board.legal_moves)):
            # Check if we've run out of time
            if self._check_time_limit():
                break
                
            self.board.push(m)
            score = self.minimax_alpha_beta(
                depth - 1, alpha, beta, not maximizing_player
            )
            self.board.pop()

            if maximizing_player:
                if score > best_score:
                    best_score = score
                    best_move = m
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = m
                beta = min(beta, score)
        logger.info("Best move found: %s with score %f", best_move, best_score)
        logger.info("Total nodes visited: %d", self.node_count)
        return best_score, best_move

    # uses MVV-LVA (Most Valuable Victim - Least Valuable Aggressor) to order moves

    def order_moves(self, moves: list[chess.Move]) -> list[chess.Move]:
        logger.debug("Ordering %d moves", len(moves))

        ordered_moves = []

        # First, separate moves into different categories

        checks = []
        captures = []
        other_moves = []

        for move in moves:

            # Check if the move is a capture

            if self.board.is_capture(move):
                captures.append(move)
            else:

                # Make the move to see if it gives check

                self.board.push(move)
                gives_check = self.board.is_check()
                self.board.pop()

                if gives_check:
                    checks.append(move)
                else:
                    other_moves.append(move)
        # Sort captures by MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
        # Higher score means better capture

        def capture_score(move):
            from .constants import PIECE_VALUES

            # Get the captured piece value (victim)

            to_square = move.to_square
            victim_piece = self.board.piece_at(to_square)
            victim_value = (
                PIECE_VALUES.get(victim_piece.piece_type, 0) if victim_piece else 0
            )

            # Get the moving piece value (aggressor)

            from_square = move.from_square
            aggressor_piece = self.board.piece_at(from_square)
            aggressor_value = (
                PIECE_VALUES.get(aggressor_piece.piece_type, 0)
                if aggressor_piece
                else 0
            )

            # MVV-LVA score: higher victim value and lower aggressor value is better

            return victim_value * 10 - aggressor_value

        # Sort captures by MVV-LVA score in descending order

        captures.sort(key=capture_score, reverse=True)

        # Combine the ordered moves: checks first, then captures, then other moves

        ordered_moves = checks + captures + other_moves

        logger.debug(
            "Move ordering complete: %d checks, %d captures, %d other moves",
            len(checks),
            len(captures),
            len(other_moves),
        )

        return ordered_moves

    def minimax_alpha_beta(
        self, depth: int, alpha: float, beta: float, maximizing_player: bool
    ) -> float:
        # Check time limit periodically to avoid going over
        if self.node_count % 1000 == 0 and self._check_time_limit():
            return 0  # Return neutral score if time is up
            
        hash_val = None
        tt_hit = False
        
        # Try to retrieve from transposition table if using Zobrist
        if self.use_zobrist:
            hash_val = self.zobrist.hash_board(self.board)
            tt_entry = self.transposition_table.get(hash_val)
            if tt_entry and tt_entry["depth"] >= depth:
                entry_type = tt_entry.get("type", "exact")
                score = tt_entry["score"]
                
                if entry_type == "exact":
                    return score
                elif entry_type == "lower" and score >= beta:
                    return beta
                elif entry_type == "upper" and score <= alpha:
                    return alpha

        self.node_count += 1
        logger.debug(
            "Minimax called with depth: %d, alpha: %f, beta: %f, maximizing_player: %s, node %d",
            depth,
            alpha,
            beta,
            maximizing_player,
            self.node_count,
        )

        # Base case: if depth is 0 or game is over, return evaluation score

        if depth == 0 or self.board.is_game_over():
            if self.board.is_checkmate():
                logger.debug("Found checkmate at depth: %d", depth)
                score = -2147483648 if maximizing_player else 2147483647
            elif self.board.is_stalemate():
                logger.debug("Found stalemate at depth: %d", depth)
                score = 0
            else:
                # Depth limit reached, evaluate the position statically

                score = self.evaluator.evaluate()
                logger.debug("Leaf node reached with score: %f", score)
            
            # Store in transposition table if using Zobrist - only check size limit occasionally
            if self.use_zobrist and hash_val is not None:
                # Only check size limit every 100 entries to reduce overhead
                if len(self.transposition_table) < self.max_tt_entries or self.node_count % 100 != 0:
                    self.transposition_table[hash_val] = {"depth": depth, "score": score}
                
            return score

        # Dummy player is maximizing
        if maximizing_player:
            max_eval = float(-2147483648)

            for m in self.order_moves(list(self.board.legal_moves)):
                # Check time limit before each move
                if self._check_time_limit():
                    break
                    
                self.board.push(m)
                eval = self.minimax_alpha_beta(depth - 1, alpha, beta, False)
                self.board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)

                # prune
                if beta <= alpha:
                    logger.debug(
                        f"Pruning (Max): depth={depth}, move={m}, alpha={alpha}, beta={beta}"
                    )
                    break
            
            logger.info("Maximizing player evaluation: %f", max_eval)
            # Store result in transposition table - reduce size checking overhead
            if self.use_zobrist and hash_val is not None:
                if len(self.transposition_table) < self.max_tt_entries or self.node_count % 100 != 0:
                    self.transposition_table[hash_val] = {"depth": depth, "score": max_eval}
            return max_eval

        # Dummy player is minimizing
        else:
            min_eval = float(2147483647)
            for m in self.order_moves(list(self.board.legal_moves)):
                # Check time limit before each move
                if self._check_time_limit():
                    break
                    
                self.board.push(m)
                eval = self.minimax_alpha_beta(depth - 1, alpha, beta, True)
                self.board.pop()

                min_eval = min(min_eval, eval)
                beta = min(beta, eval)

                # prune
                if beta <= alpha:
                    logger.debug(
                        f"Pruning (Min): depth={depth}, move={m}, alpha={alpha}, beta={beta}"
                    )
                    break
            
            logger.info("Minimizing player evaluation: %f", min_eval)
            # Store result in transposition table - reduce size checking overhead
            if self.use_zobrist and hash_val is not None:
                if len(self.transposition_table) < self.max_tt_entries or self.node_count % 100 != 0:
                    self.transposition_table[hash_val] = {"depth": depth, "score": min_eval}
            return min_eval

import logging

import chess
from src.engine.evaluators.eval import Eval

logger = logging.getLogger(__name__)


class Minimax:
    # TODO: Store calculated line, can work up backwards from that node rather than top down approach.

    board: chess.Board
    evaluator: Eval
    node_count: int
    score: float
    alpha: float
    beta: float

    def __init__(self, board: chess.Board, evaluator: Eval):
        logger.debug("Minimax initialized with board: %s", board)
        self.board = board
        self.evaluator = evaluator

    # Returns a tuple of score(float) and move

    def find_top_move(self, depth: int = 1) -> tuple[float, chess.Move | None]:
        logger.info("Finding best move for ", self.board.turn.__str__())
        self.node_count = 0
        maximizing_player = self.board.turn
        alpha = float(-2147483648)
        beta = float(2147483647)
        best_move: chess.Move | None = None

        if maximizing_player:
            best_score = float(-2147483648)
        else:
            best_score = float(2147483647)
        for m in self.order_moves(list(self.board.legal_moves)):
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
            return score
        # Dummy player is maximizing
        # This player is trying to maximize the score

        if maximizing_player:
            max_eval = float(-2147483648)

            for m in self.order_moves(list(self.board.legal_moves)):
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
            return max_eval
        # Dummy player is minimizing
        # This player is trying to minimize the score

        else:
            min_eval = float(2147483647)
            for m in self.order_moves(list(self.board.legal_moves)):
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
            return min_eval

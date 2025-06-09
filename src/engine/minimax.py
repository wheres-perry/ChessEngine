import logging
import time
from typing import Any

import chess
from src.engine.constants import PIECE_VALUES
from src.engine.evaluators.eval import Eval
from src.engine.zobrist import Zobrist

logger = logging.getLogger(__name__)


class Minimax:
    NEGATIVE_INFINITY = -2147483648
    POSITIVE_INFINITY = 2147483647
    DEFAULT_TT_SIZE = 100000
    TIME_CHECK_INTERVAL = 1000
    TT_SIZE_CHECK_INTERVAL = 100
    MVV_LVA_MULTIPLIER = 10

    board: chess.Board
    evaluator: Eval
    node_count: int
    score: float
    alpha: float
    beta: float
    use_zobrist: bool
    use_iddfs: bool
    max_time: float | None
    start_time: float | None
    time_up: bool
    zobrist: Zobrist | None
    transposition_table: dict[int, dict[str, Any]]
    max_tt_entries: int

    def __init__(
        self,
        board: chess.Board,
        evaluator: Eval,
        use_zobrist: bool = True,
        use_iddfs: bool = True,
        max_time: float | None = None,
    ):
        self.use_zobrist = use_zobrist
        self.use_iddfs = use_iddfs
        self.max_time = max_time
        self.start_time = None
        self.time_up = False

        if use_zobrist:
            self.zobrist = Zobrist()
            self.transposition_table = {}
            self.max_tt_entries = self.DEFAULT_TT_SIZE
        else:
            self.zobrist = None
            self.transposition_table = {}
            self.max_tt_entries = 0
        logger.debug("Minimax initialized with board: %s", board)
        self.board = board
        self.evaluator = evaluator

    def _check_time_limit(self) -> bool:
        """Check if time limit has been exceeded."""
        if (
            self.max_time
            and self.start_time
            and time.time() - self.start_time >= self.max_time
        ):
            self.time_up = True
            return True
        return False

    def find_top_move(self, depth: int = 1) -> tuple[float | None, chess.Move | None]:
        logger.info("Finding best move for ", self.board.turn.__str__())
        self.node_count = 0
        self.time_up = False
        self.start_time = time.time()

        if self.use_iddfs and depth > 1:
            return self._iterative_deepening(depth)
        else:
            return self._search_fixed_depth(depth)

    def _iterative_deepening(
        self, max_depth: int
    ) -> tuple[float | None, chess.Move | None]:
        best_score = None
        best_move = None

        for current_depth in range(1, max_depth + 1):
            if self._check_time_limit():
                logger.info(f"Time limit reached before starting depth {current_depth}")
                break
            score, move = self._search_fixed_depth(current_depth)

            if self.time_up:
                logger.info(f"Time limit reached during depth {current_depth} search")
                break
            if move is not None:
                best_score = score
                best_move = move
            logger.info(
                f"Completed depth {current_depth} search: best move {move} with score {score}"
            )
        if best_move is None:
            logger.warning("No valid moves found in iterative deepening")
        return best_score, best_move

    def _search_fixed_depth(self, depth: int) -> tuple[float, chess.Move | None]:
        maximizing_player = self.board.turn
        alpha = float(self.NEGATIVE_INFINITY)
        beta = float(self.POSITIVE_INFINITY)
        best_move: chess.Move | None = None

        if maximizing_player:
            best_score = float(self.NEGATIVE_INFINITY)
        else:
            best_score = float(self.POSITIVE_INFINITY)
        for m in self.order_moves(list(self.board.legal_moves)):
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

    def order_moves(self, moves: list[chess.Move]) -> list[chess.Move]:
        logger.debug("Ordering %d moves", len(moves))

        checks = []
        captures = []
        other_moves = []

        for move in moves:
            if self.board.is_capture(move):
                captures.append(move)
            else:
                self.board.push(move)
                gives_check = self.board.is_check()
                self.board.pop()

                if gives_check:
                    checks.append(move)
                else:
                    other_moves.append(move)

        def capture_score(move: chess.Move) -> float:
            to_square = move.to_square
            victim_piece = self.board.piece_at(to_square)
            victim_value = (
                PIECE_VALUES.get(victim_piece.piece_type, 0) if victim_piece else 0
            )

            from_square = move.from_square
            aggressor_piece = self.board.piece_at(from_square)
            aggressor_value = (
                PIECE_VALUES.get(aggressor_piece.piece_type, 0)
                if aggressor_piece
                else 0
            )

            return victim_value * self.MVV_LVA_MULTIPLIER - aggressor_value

        captures.sort(key=capture_score, reverse=True)
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
        if self.node_count % self.TIME_CHECK_INTERVAL == 0 and self._check_time_limit():
            return 0.0
        hash_val: int | None = None

        if self.use_zobrist and self.zobrist:
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

        if depth == 0 or self.board.is_game_over():
            if self.board.is_checkmate():
                logger.debug("Found checkmate at depth: %d", depth)
                score = (
                    float(self.NEGATIVE_INFINITY)
                    if maximizing_player
                    else float(self.POSITIVE_INFINITY)
                )
            elif self.board.is_stalemate():
                logger.debug("Found stalemate at depth: %d", depth)
                score = 0.0
            else:
                score = self.evaluator.evaluate()
                logger.debug("Leaf node reached with score: %f", score)
            if self.use_zobrist and hash_val is not None:
                if (
                    len(self.transposition_table) < self.max_tt_entries
                    or self.node_count % self.TT_SIZE_CHECK_INTERVAL == 0
                ):
                    self.transposition_table[hash_val] = {
                        "depth": depth,
                        "score": score,
                    }
            return score
        if maximizing_player:
            max_eval = float(self.NEGATIVE_INFINITY)

            for m in self.order_moves(list(self.board.legal_moves)):
                if self._check_time_limit():
                    break
                self.board.push(m)
                eval_score = self.minimax_alpha_beta(depth - 1, alpha, beta, False)
                self.board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha:
                    logger.debug(
                        f"Pruning (Max): depth={depth}, move={m}, alpha={alpha}, beta={beta}"
                    )
                    break
            logger.info("Maximizing player evaluation: %f", max_eval)
            if self.use_zobrist and hash_val is not None:
                if (
                    len(self.transposition_table) < self.max_tt_entries
                    or self.node_count % self.TT_SIZE_CHECK_INTERVAL == 0
                ):
                    self.transposition_table[hash_val] = {
                        "depth": depth,
                        "score": max_eval,
                    }
            return max_eval
        else:
            min_eval = float(self.POSITIVE_INFINITY)

            for m in self.order_moves(list(self.board.legal_moves)):
                if self._check_time_limit():
                    break
                self.board.push(m)
                eval_score = self.minimax_alpha_beta(depth - 1, alpha, beta, True)
                self.board.pop()

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    logger.debug(
                        f"Pruning (Min): depth={depth}, move={m}, alpha={alpha}, beta={beta}"
                    )
                    break
            logger.info("Minimizing player evaluation: %f", min_eval)

            if self.use_zobrist and hash_val is not None:
                if (
                    len(self.transposition_table) < self.max_tt_entries
                    or self.node_count % self.TT_SIZE_CHECK_INTERVAL != 0
                ):
                    self.transposition_table[hash_val] = {
                        "depth": depth,
                        "score": min_eval,
                    }
            return min_eval

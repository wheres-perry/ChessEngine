import chess

from src.engine.config import EvaluationConfig
from src.engine.constants import EVAL_PIECES, PIECE_VALUES
from src.engine.evaluators.eval import *


def make_pst(table: list[int]) -> list[float]:
    """Flips the table for the full 64 squares and scales the values."""
    table_float = [v / 100.0 for v in table]
    return table_float + table_float[::-1]


mg_pawn_pst = make_pst(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5,
        10,
        10,
        -20,
        -20,
        10,
        10,
        5,
        5,
        -5,
        -10,
        0,
        0,
        -10,
        -5,
        5,
        0,
        0,
        0,
        20,
        20,
        0,
        0,
        0,
    ]
)
eg_pawn_pst = make_pst(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
    ]
)
mg_knight_pst = make_pst(
    [
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
        -40,
        -20,
        0,
        5,
        5,
        0,
        -20,
        -40,
        -30,
        5,
        10,
        15,
        15,
        10,
        5,
        -30,
        -30,
        0,
        15,
        20,
        20,
        15,
        0,
        -30,
    ]
)
eg_knight_pst = make_pst(
    [
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
    ]
)
mg_bishop_pst = make_pst(
    [
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
        -10,
        5,
        0,
        0,
        0,
        0,
        5,
        -10,
        -10,
        10,
        10,
        10,
        10,
        10,
        10,
        -10,
        -10,
        0,
        10,
        10,
        10,
        10,
        0,
        -10,
    ]
)
eg_bishop_pst = make_pst(
    [
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
    ]
)
mg_rook_pst = make_pst(
    [
        0,
        0,
        0,
        5,
        5,
        0,
        0,
        0,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
    ]
)
eg_rook_pst = make_pst(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5,
        10,
        10,
        10,
        10,
        10,
        10,
        5,
        5,
        10,
        10,
        10,
        10,
        10,
        10,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
)
mg_queen_pst = make_pst(
    [
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
        -10,
        0,
        5,
        0,
        0,
        0,
        0,
        -10,
        -10,
        5,
        5,
        5,
        5,
        5,
        0,
        -10,
        0,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
    ]
)
eg_queen_pst = make_pst(
    [
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
    ]
)
mg_king_pst = make_pst(
    [
        20,
        30,
        10,
        0,
        0,
        10,
        30,
        20,
        20,
        20,
        0,
        0,
        0,
        0,
        20,
        20,
        -10,
        -20,
        -20,
        -20,
        -20,
        -20,
        -20,
        -10,
        -20,
        -30,
        -30,
        -40,
        -40,
        -30,
        -30,
        -20,
    ]
)
eg_king_pst = make_pst(
    [
        -50,
        -30,
        -30,
        -30,
        -30,
        -30,
        -30,
        -50,
        -30,
        -30,
        0,
        0,
        0,
        0,
        -30,
        -30,
        -30,
        -10,
        20,
        30,
        30,
        20,
        -10,
        -30,
        -30,
        -10,
        30,
        40,
        40,
        30,
        -10,
        -30,
    ]
)

# Map piece types to their PSTs


PST = {
    chess.PAWN: (mg_pawn_pst, eg_pawn_pst),
    chess.KNIGHT: (mg_knight_pst, eg_knight_pst),
    chess.BISHOP: (mg_bishop_pst, eg_bishop_pst),
    chess.ROOK: (mg_rook_pst, eg_rook_pst),
    chess.QUEEN: (mg_queen_pst, eg_queen_pst),
    chess.KING: (mg_king_pst, eg_king_pst),
}


class ComplexEval(Eval):
    """
    A complex evaluator that considers material, piece-square tables,
    mobility, pawn structure, and king safety.
    The final score is from White's perspective (positive is good for White).
    """

    def __init__(self, board: chess.Board, config: EvaluationConfig):
        super().__init__(board)
        self.config = config
        self.pawn_files = {chess.WHITE: [0] * 8, chess.BLACK: [0] * 8}
        self._init_pawn_structure()

    def _init_pawn_structure(self):
        """Pre-calculates pawn file information."""
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                file_idx = chess.square_file(sq)
                self.pawn_files[piece.color][file_idx] += 1

    def evaluate(self) -> float:
        """
        Calculates the evaluation of the current board position.
        A positive score favors White, a negative score favors Black.
        """
        if self.board.is_checkmate():
            # If white is to move and is checkmated, black wins.

            return -9999 if self.board.turn == chess.WHITE else 9999
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0
        score = 0.0
        game_phase = self._get_game_phase()

        if self.config.use_material:
            score += self._evaluate_material()
        if self.config.use_pst:
            score += self._evaluate_pst(game_phase)
        if self.config.use_pawn_structure:
            score += self._evaluate_pawn_structure()
        if self.config.use_king_safety:
            score += self._evaluate_king_safety(game_phase)
        if self.config.use_mobility:
            score += self._evaluate_mobility()
        return score

    def _get_game_phase(self) -> float:
        """
        Calculates the game phase, from 1.0 (opening) to 0.0 (endgame).
        Based on the material on the board, excluding kings and pawns.
        """
        phase = 0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            phase += len(self.board.pieces(piece_type, chess.WHITE))
            phase += len(self.board.pieces(piece_type, chess.BLACK))
        # Total number of pieces to start is 2*2 Knights, 2*2 Bishops, 2*2 Rooks, 2*1 Queens = 12
        # We can scale it from 0 to 24 (max possible pieces)

        return min(phase / 24.0, 1.0)

    def _evaluate_material(self) -> float:
        """Evaluates material balance."""
        score = 0.0
        for piece_type in EVAL_PIECES:
            white_count = len(self.board.pieces(piece_type, chess.WHITE))
            black_count = len(self.board.pieces(piece_type, chess.BLACK))
            score += (white_count - black_count) * PIECE_VALUES[piece_type]
        return score

    def _evaluate_pst(self, game_phase: float) -> float:
        """Evaluates piece positions using interpolated PSTs."""
        score = 0.0
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if not piece:
                continue
            mg_pst, eg_pst = PST[piece.piece_type]

            # Interpolate between midgame and endgame tables

            pst_val = mg_pst[sq] * game_phase + eg_pst[sq] * (1 - game_phase)

            if piece.color == chess.WHITE:
                score += pst_val
            else:
                # For black, the square index must be flipped for the table lookup

                score -= pst_val
        return score

    def _evaluate_mobility(self) -> float:
        """
        Evaluates mobility (number of legal moves).
        A small weight is applied to this score.
        """
        MOBILITY_WEIGHT = 0.05

        # White's mobility

        self.board.turn = chess.WHITE
        white_mobility = self.board.legal_moves.count()

        # Black's mobility

        self.board.turn = chess.BLACK
        black_mobility = self.board.legal_moves.count()

        # Restore the original turn

        self.board.turn = not self.board.turn

        return (white_mobility - black_mobility) * MOBILITY_WEIGHT

    def _evaluate_pawn_structure(self) -> float:
        """Evaluates pawn structure for doubled and isolated pawns."""
        score = 0.0
        DOUBLED_PAWN_PENALTY = -0.25
        ISOLATED_PAWN_PENALTY = -0.15

        for color in [chess.WHITE, chess.BLACK]:
            color_penalty = 0.0
            for file_idx in range(8):
                pawn_count = self.pawn_files[color][file_idx]
                if pawn_count > 1:
                    color_penalty += pawn_count * DOUBLED_PAWN_PENALTY
                if pawn_count > 0:
                    left_file = max(0, file_idx - 1)
                    right_file = min(7, file_idx + 1)
                    is_isolated = (
                        self.pawn_files[color][left_file] == 0
                        and self.pawn_files[color][right_file] == 0
                    )
                    if is_isolated:
                        color_penalty += ISOLATED_PAWN_PENALTY
            if color == chess.WHITE:
                score += color_penalty
            else:
                score -= color_penalty
        return score

    def _evaluate_king_safety(self, game_phase: float) -> float:
        """Evaluates king safety based on pawn shield."""
        score = 0.0
        PAWN_SHIELD_BONUS = 0.4

        # King safety is less important in the endgame

        if game_phase < 0.3:
            return 0.0
        for color in [chess.WHITE, chess.BLACK]:
            king_sq = self.board.king(color)
            if king_sq is None:
                continue
            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)

            shield_bonus = 0.0

            # Find pawns in front of the king

            for file_offset in [-1, 0, 1]:
                check_file = king_file + file_offset
                if 0 <= check_file <= 7:
                    # Look for friendly pawns one rank ahead

                    pawn_sq = chess.square(
                        check_file, king_rank + (1 if color == chess.WHITE else -1)
                    )
                    if self.board.piece_at(pawn_sq) == chess.Piece(chess.PAWN, color):
                        shield_bonus += PAWN_SHIELD_BONUS
            # Apply the bonus, scaled by game phase

            shield_bonus *= game_phase
            if color == chess.WHITE:
                score += shield_bonus
            else:
                score -= shield_bonus
        return score

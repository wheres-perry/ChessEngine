"""Concrete evaluation components.

Each class is a self-contained heuristic that respects the ``EvalComponent``
interface.  When constructed with ``gsc=True`` (Game-Stage Conscious), the
component blends opening / middlegame / endgame weights using the game
phase value supplied by the composite evaluator.

All scores are in **centipawns** (1 pawn = 100 cp).

Components
----------
MaterialComponent    - raw material balance (always included)
PSTComponent         - piece-square table bonuses
PawnStructureComponent - doubled / isolated / passed pawn analysis
MobilityComponent    - legal-move count weighted by piece type
KingSafetyComponent  - pawn shield, open-file penalty, attack zone
"""

from __future__ import annotations

from engine._core import chess_engine_core as chess
from engine.evaluators.base import EvalComponent
from engine.evaluators.pst_tables import (
    PIECE_SQUARE_TABLES_EG,
    PIECE_SQUARE_TABLES_MG,
)


# --- Helpers ---
def _lerp(mg: float, eg: float, phase: float) -> float:
    """Linearly interpolate between middlegame and endgame values.

    *phase* 1.0 -> full middlegame, 0.0 -> full endgame.
    """
    return mg * phase + eg * (1.0 - phase)


_MATERIAL_CP: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


# --- Material (always-on baseline) ---
class MaterialComponent(EvalComponent):
    """Pure material balance in centipawns."""

    def score(self, board: chess.Board, phase: float) -> float:  # noqa: ARG002
        """Return material balance score in centipawns.

        Args:
            board: The current board position.
            phase: Game phase (unused for material).

        Returns:
            Positive score favors White, negative favors Black.
        """
        total = 0
        for piece_type, cp in _MATERIAL_CP.items():
            w = len(board.pieces(piece_type, chess.WHITE))
            b = len(board.pieces(piece_type, chess.BLACK))
            total += (w - b) * cp
        return float(total)


# --- Piece-Square Tables ---
class PSTComponent(EvalComponent):
    """Piece-square table evaluation.

    Without GSC: uses the middlegame tables only.
    With GSC:    interpolates MG <-> EG tables based on game phase.
    """

    def __init__(self, *, gsc: bool = False) -> None:
        self._gsc = gsc

    def score(self, board: chess.Board, phase: float) -> float:
        """Return piece-square table score in centipawns.

        Args:
            board: The current board position.
            phase: Game phase value in [0.0, 1.0] for interpolation.

        Returns:
            PST evaluation score (positive favors White).
        """
        total = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            mg = PIECE_SQUARE_TABLES_MG[piece.piece_type][square]

            if self._gsc:
                eg = PIECE_SQUARE_TABLES_EG[piece.piece_type][square]
                value = _lerp(mg, eg, phase)
            else:
                value = mg

            if piece.color == chess.WHITE:
                total += value
            else:
                total -= value

        return total


# --- Pawn Structure ---
_DOUBLED_PENALTY_CP = 20
_ISOLATED_PENALTY_CP = 25
_PASSED_BASE_CP = 10
_PASSED_PER_RANK_CP = 10

# GSC phase multipliers: [opening, endgame]
_PAWN_STRUCT_GSC_MG = 0.6
_PAWN_STRUCT_GSC_EG = 1.4


class PawnStructureComponent(EvalComponent):
    """Doubled, isolated, and passed-pawn evaluation.

    Without GSC: flat contribution.
    With GSC:    pawn structure matters *more* in the endgame
                 (_PAWN_STRUCT_GSC_EG) and *less* in the opening
                 (_PAWN_STRUCT_GSC_MG), interpolated by phase.
    """

    def __init__(self, *, gsc: bool = False) -> None:
        self._gsc = gsc

    def score(self, board: chess.Board, phase: float) -> float:
        """Return pawn structure evaluation in centipawns.

        Evaluates doubled, isolated, and passed pawns with phase weighting.

        Args:
            board: The current board position.
            phase: Game phase value in [0.0, 1.0] for interpolation.

        Returns:
            Pawn structure score (positive favors White).
        """
        raw = 0.0
        for color in (chess.WHITE, chess.BLACK):
            sign = 1.0 if color == chess.WHITE else -1.0
            pawns = board.pieces(chess.PAWN, color)

            for sq in pawns:
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)

                # Doubled pawns
                same_file = [p for p in pawns if chess.square_file(p) == file]
                if len(same_file) > 1:
                    raw -= _DOUBLED_PENALTY_CP * sign

                # Isolated pawns
                has_neighbour = False
                for adj in (file - 1, file + 1):
                    if 0 <= adj <= 7 and any(
                        chess.square_file(p) == adj for p in pawns
                    ):
                        has_neighbour = True
                        break
                if not has_neighbour:
                    raw -= _ISOLATED_PENALTY_CP * sign

                # Passed pawns
                if self._is_passed(board, sq, color):
                    advancement = rank if color == chess.WHITE else (7 - rank)
                    raw += (_PASSED_BASE_CP + _PASSED_PER_RANK_CP * advancement) * sign

        if self._gsc:
            weight = _lerp(_PAWN_STRUCT_GSC_MG, _PAWN_STRUCT_GSC_EG, phase)
            return raw * weight
        return raw

    @staticmethod
    def _is_passed(board: chess.Board, square: int, color: chess.Color) -> bool:
        """Check if a pawn is passed (no enemy pawns ahead on adjacent files).

        Args:
            board: The current board position.
            square: The square of the pawn to check.
            color: The color of the pawn.

        Returns:
            True if the pawn is passed, False otherwise.
        """
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        enemy = chess.Color.BLACK if color == chess.Color.WHITE else chess.Color.WHITE
        enemy_pawns = board.pieces(chess.PAWN, enemy)

        for check_file in (file - 1, file, file + 1):
            if not (0 <= check_file <= 7):
                continue
            for ep in enemy_pawns:
                if chess.square_file(ep) != check_file:
                    continue
                er = chess.square_rank(ep)
                if color == chess.WHITE and er > rank:
                    return False
                if color == chess.BLACK and er < rank:
                    return False
        return True


# --- Mobility ---

# Per-piece mobility weights (centipawn value per legal move)
_MOBILITY_WEIGHTS: dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 5.0,
    chess.BISHOP: 5.0,
    chess.ROOK: 3.0,
    chess.QUEEN: 2.0,
}

# GSC: mobility barely matters in the opening, matters a lot mid/endgame
_MOBILITY_GSC_MG = 0.3
_MOBILITY_GSC_EG = 1.3


class MobilityComponent(EvalComponent):
    """Weighted legal-move count per piece.

    Without GSC: flat contribution.
    With GSC:    mobility is heavily discounted in the opening
                 and amplified from the middlegame onward.
    """

    def __init__(self, *, gsc: bool = False) -> None:
        self._gsc = gsc

    def score(self, board: chess.Board, phase: float) -> float:
        """Return mobility evaluation in centipawns.

        Args:
            board: The current board position.
            phase: Game phase value in [0.0, 1.0] for interpolation.

        Returns:
            Mobility score (positive favors White).
        """
        raw = self._raw_mobility(board)
        if self._gsc:
            weight = _lerp(_MOBILITY_GSC_MG, _MOBILITY_GSC_EG, phase)
            return raw * weight
        return raw

    @staticmethod
    def _raw_mobility(board: chess.Board) -> float:
        """Compute raw mobility differential for the side to move.

        Args:
            board: The current board position.

        Returns:
            Weighted mobility score (positive favors White).
        """
        move_counts: dict[int, int] = {}
        for m in board.legal_moves:
            move_counts[m.from_square] = move_counts.get(m.from_square, 0) + 1

        total = 0.0
        for sq, cnt in move_counts.items():
            piece = board.piece_at(sq)
            if piece is None or piece.piece_type == chess.KING:
                continue
            w = _MOBILITY_WEIGHTS.get(piece.piece_type, 0.0)
            if piece.color == chess.WHITE:
                total += cnt * w
            else:
                total -= cnt * w

        return total


# --- King Safety ---

_PAWN_SHIELD_BONUS_CP = 15.0
_OPEN_FILE_PENALTY_CP = 30.0
_ATTACK_ZONE_WEIGHT_CP = 8.0

# GSC: king safety matters most in the middlegame, less in endgame
_KING_SAFETY_GSC_MG = 1.3
_KING_SAFETY_GSC_EG = 0.4


class KingSafetyComponent(EvalComponent):
    """King safety based on pawn shield, open files, and attack zone.

    Without GSC: flat contribution.
    With GSC:    king safety is amplified in the middlegame (when attacks are
                 dangerous) and fades in the endgame (when the king should
                 be active).
    """

    def __init__(self, *, gsc: bool = False) -> None:
        self._gsc = gsc

    def score(self, board: chess.Board, phase: float) -> float:
        """Return king safety evaluation in centipawns.

        Evaluates pawn shield, open files near king, and attack zone pressure.

        Args:
            board: The current board position.
            phase: Game phase value in [0.0, 1.0] for interpolation.

        Returns:
            King safety score (positive favors White).
        """
        raw = 0.0
        for color in (chess.WHITE, chess.BLACK):
            sign = 1.0 if color == chess.WHITE else -1.0
            ks = board.king(color)
            if ks is None:
                continue  # type: ignore[unreachable]
            kf = chess.square_file(ks)
            kr = chess.square_rank(ks)

            raw += self._pawn_shield(board, color, kf, kr) * sign
            raw += self._open_file_penalty(board, kf) * sign
            raw -= self._attack_zone_pressure(board, color, kf, kr) * sign

        if self._gsc:
            weight = _lerp(_KING_SAFETY_GSC_MG, _KING_SAFETY_GSC_EG, phase)
            return raw * weight
        return raw

    @staticmethod
    def _pawn_shield(
        board: chess.Board,
        color: chess.Color,
        king_file: int,
        king_rank: int,
    ) -> float:
        """Calculate pawn shield bonus for the king.

        Args:
            board: The current board position.
            color: The king's color.
            king_file: The king's file (0-7).
            king_rank: The king's rank (0-7).

        Returns:
            Pawn shield bonus in centipawns.
        """
        bonus = 0.0
        direction = 1 if color == chess.Color.WHITE else -1
        for df in (-1, 0, 1):
            f = king_file + df
            r = king_rank + direction
            if not (0 <= f <= 7 and 0 <= r <= 7):
                continue
            piece = board.piece_at(r * 8 + f)
            if (
                piece is not None
                and piece.piece_type == chess.PAWN
                and piece.color == color
            ):
                bonus += _PAWN_SHIELD_BONUS_CP
        return bonus

    @staticmethod
    def _open_file_penalty(board: chess.Board, king_file: int) -> float:
        """Calculate penalty for king on an open file (no pawns).

        Args:
            board: The current board position.
            king_file: The king's file (0-7).

        Returns:
            Open file penalty in centipawns (negative value).
        """
        for sq in chess.SQUARES:
            if chess.square_file(sq) != king_file:
                continue
            piece = board.piece_at(sq)
            if piece is not None and piece.piece_type == chess.PAWN:
                return 0.0
        return -_OPEN_FILE_PENALTY_CP

    @staticmethod
    def _attack_zone_pressure(
        board: chess.Board,
        color: chess.Color,
        king_file: int,
        king_rank: int,
    ) -> float:
        """Count enemy pieces that attack the 3x3 king zone.

        Args:
            board: The current board position.
            color: The king's color.
            king_file: The king's file (0-7).
            king_rank: The king's rank (0-7).

        Returns:
            Attack zone pressure score in centipawns.
        """
        enemy = chess.Color.BLACK if color == chess.Color.WHITE else chess.Color.WHITE
        pressure = 0.0
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                f, r = king_file + df, king_rank + dr
                if not (0 <= f <= 7 and 0 <= r <= 7):
                    continue
                sq = r * 8 + f
                piece = board.piece_at(sq)
                if piece is not None and piece.color == enemy:
                    pressure += _ATTACK_ZONE_WEIGHT_CP
        return pressure

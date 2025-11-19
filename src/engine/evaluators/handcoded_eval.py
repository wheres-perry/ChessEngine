"""
Handcoded chess position evaluator with modular features.

Implements traditional chess evaluation using piece-square tables, mobility,
pawn structure, and king safety. Fully modular according to Tree 2 dependencies.

Tree 2: State Evaluation Optimizations
E0 (Material) -> E1 (PST) -> E2 (Tapered), E3 (Pawn), E4 (Mobility), E6 (SEE)
E4 -> E5 (King Safety)
"""

from engine._core import chess_engine_core as chess
from src.engine.config import EvaluationConfig
from src.engine.evaluators.base_evaluator import BaseEvaluator
from src.engine.evaluators.pst_tables import (
    PIECE_SQUARE_TABLES_EG,
    PIECE_SQUARE_TABLES_MG,
)


class HandcodedEvaluator(BaseEvaluator):
    """
    Handcoded evaluator with traditional chess heuristics.

    Evaluates positions using:
    - Material count (E0)
    - Piece-square tables (E1)
    - Tapered evaluation (E2)
    - Pawn structure (E3)
    - Piece mobility (E4)
    - King safety (E5)
    - Static exchange evaluation (E6)

    Each component can be individually enabled/disabled via configuration.
    """

    def __init__(self, board: chess.Board, config: EvaluationConfig):
        """
        Initialize handcoded evaluator.

        Args:
            board: Chess board to evaluate
            config: Configuration specifying which features to use
        """
        super().__init__(board, config)

    def evaluate(self) -> float:
        """
        Evaluate position using enabled features.

        Returns:
            Evaluation score (positive favors White, negative favors Black)
        """
        score = 0.0

        # Node E0: Material (required for complex evaluator)
        if self.config.use_material:
            score += self._evaluate_material()

        # Node E1: Piece-Square Tables (requires E0)
        if self.config.use_pst:
            score += self._evaluate_pst()

        # Node E3: Pawn Structure (requires E1)
        if self.config.use_pawn_structure:
            score += self._evaluate_pawn_structure()

        # Node E4: Mobility (requires E1)
        if self.config.use_mobility:
            score += self._evaluate_mobility()

        # Node E5: King Safety (requires E4)
        if self.config.use_king_safety:
            score += self._evaluate_king_safety()

        # Node E6: Static Exchange Evaluation (requires E1)
        # SEE is typically used for move ordering, but can contribute to eval
        # if self.config.use_see:
        #     score += self._evaluate_see()

        return score

    # =========================================================================
    # Node E0: Material Evaluation (Tree 2 Root)
    # =========================================================================

    def _evaluate_material(self) -> float:
        """Evaluate material balance, return score in pawns (+ favors White)."""
        white_material, black_material = self.count_material()

        # Return difference in pawns
        return (white_material - black_material) / 100.0

    # =========================================================================
    # Node E1: Piece-Square Tables (Requires E0)
    # =========================================================================

    def _evaluate_pst(self) -> float:
        """Evaluate piece placement using piece-square tables, return score."""
        score = 0.0

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue

            # Get PST values for this piece
            mg_value = PIECE_SQUARE_TABLES_MG[piece.piece_type][square]
            eg_value = PIECE_SQUARE_TABLES_EG[piece.piece_type][square]

            # Interpolate based on game phase if tapered eval enabled
            if self.config.use_tapered_eval:
                value = self.interpolate(mg_value, eg_value)
            else:
                value = mg_value

            # Add or subtract based on color
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value

        return score / 100.0  # Convert to pawns

    # =========================================================================
    # Node E3: Pawn Structure (Requires E1)
    # =========================================================================

    def _evaluate_pawn_structure(self) -> float:
        """Evaluate pawn structure considering doubled, isolated, and passed pawns."""
        score = 0.0

        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1.0 if color == chess.WHITE else -1.0

            pawns = self.board.pieces(chess.PAWN, color)

            for pawn_square in pawns:
                file = chess.square_file(pawn_square)
                rank = chess.square_rank(pawn_square)

                # Check for doubled pawns (multiple pawns on same file)
                pawns_on_file = [p for p in pawns if chess.square_file(p) == file]
                if len(pawns_on_file) > 1:
                    score -= 0.2 * multiplier

                # Check for isolated pawns (no friendly pawns on adjacent files)
                has_neighbor = False
                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file <= 7:
                        adj_pawns = [
                            p for p in pawns if chess.square_file(p) == adj_file
                        ]
                        if adj_pawns:
                            has_neighbor = True
                            break

                if not has_neighbor:
                    score -= 0.25 * multiplier

                # Check for passed pawns (no enemy pawns blocking or controlling path)
                is_passed = self._is_passed_pawn(pawn_square, color)
                if is_passed:
                    # Passed pawns are more valuable closer to promotion
                    advancement = rank if color == chess.WHITE else (7 - rank)
                    score += (0.1 + 0.1 * advancement) * multiplier

        return score

    def _is_passed_pawn(self, square: int, color: chess.Color) -> bool:
        """Check if pawn is passed (no enemy pawns can stop it)."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        enemy_color = not color
        enemy_pawns = self.board.pieces(chess.PAWN, enemy_color)

        # Check files: same file and adjacent files
        for check_file in [file - 1, file, file + 1]:
            if not (0 <= check_file <= 7):
                continue

            for enemy_pawn in enemy_pawns:
                enemy_file = chess.square_file(enemy_pawn)
                enemy_rank = chess.square_rank(enemy_pawn)

                if enemy_file != check_file:
                    continue

                # Check if enemy pawn is ahead of us
                if color == chess.WHITE:
                    if enemy_rank > rank:
                        return False
                elif enemy_rank < rank:
                    return False

        return True

    # =========================================================================
    # Node E4: Mobility (Requires E1)
    # =========================================================================

    def _evaluate_mobility(self) -> float:
        """Evaluate piece mobility by counting weighted legal moves for each piece."""
        score = 0.0

        # Count legal moves for each side
        # Note: This is somewhat expensive, might want to cache
        white_mobility = 0.0
        black_mobility = 0.0

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                # Skip empty squares and kings
                continue

            mobility = self.get_piece_mobility(square)

            # Weight mobility by piece type (more important for minor pieces)
            weight = {
                chess.PAWN: 0.01,
                chess.KNIGHT: 0.05,
                chess.BISHOP: 0.05,
                chess.ROOK: 0.03,
                chess.QUEEN: 0.02,
            }.get(piece.piece_type, 0.0)

            if piece.color == chess.WHITE:
                white_mobility += mobility * weight
            else:
                black_mobility += mobility * weight

        return white_mobility - black_mobility

    # =========================================================================
    # Node E5: King Safety (Requires E4)
    # =========================================================================

    def _evaluate_king_safety(self) -> float:
        """Evaluate king safety via pawn shield and open files analysis."""
        score = 0.0

        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1.0 if color == chess.WHITE else -1.0

            # Find king position
            king_square = self.board.king(color)
            if king_square is None:
                continue

            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)

            # Evaluate pawn shield
            pawn_shield_score = self._evaluate_pawn_shield(color, king_file, king_rank)
            score += pawn_shield_score * multiplier

            # Penalize king on open files in middlegame
            if not self.is_endgame():
                score += self._evaluate_open_file_penalty(king_file) * multiplier

        return score

    def _evaluate_pawn_shield(
        self, color: chess.Color, king_file: int, king_rank: int
    ) -> float:
        """Evaluate pawn shield in front of the king."""
        pawn_shield_score = 0.0
        for file_offset in [-1, 0, 1]:
            check_file = king_file + file_offset
            if not (0 <= check_file <= 7):
                continue

            # Look for pawns in front of king
            pawn_rank = king_rank + (1 if color == chess.WHITE else -1)
            if 0 <= pawn_rank <= 7:
                check_square = chess.square(check_file, pawn_rank)
                piece = self.board.piece_at(check_square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawn_shield_score += 0.15
        return pawn_shield_score

    def _evaluate_open_file_penalty(self, king_file: int) -> float:
        """Calculate penalty for king on an open file (no pawns)."""
        # Check if king's file has no pawns
        pawns_on_king_file = []
        for sq in chess.SQUARES:
            if chess.square_file(sq) != king_file:
                continue
            piece = self.board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                pawns_on_king_file.append(sq)

        if not pawns_on_king_file:
            return -0.3
        return 0.0

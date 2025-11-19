"""
Base evaluator with shared functionality for all evaluation approaches.

This module provides the foundation for both handcoded and neural network
evaluators, implementing common utilities and the modular evaluation framework
according to Tree 2 (State Evaluation Optimizations).

Tree 2: State Evaluation Optimizations
E0 (Material) -> E1 (PST) -> E2 (Tapered), E3 (Pawn), E4 (Mobility), E6 (SEE)
E4 -> E5 (King Safety)
E2 -> E9 (Endgame Tables)
E2 & E3 & E4 & E5 -> E10 (Tuning), E7 (Eval Caching)
E0 -> E8 (Bitboards)
"""

from abc import ABC, abstractmethod

from engine._core import chess_engine_core as chess
from src.engine.config import EvaluationConfig


class BaseEvaluator(ABC):
    """
    Abstract base class for chess position evaluators.

    Provides shared functionality and enforces modular evaluation structure
    according to Tree 2 dependencies. Subclasses implement specific evaluation
    strategies (handcoded, neural network, hybrid).
    """

    def __init__(self, board: chess.Board, config: EvaluationConfig):
        """
        Initialize the base evaluator.

        Args:
            board: Chess board to evaluate
            config: Evaluation configuration specifying which features to use
        """
        self.board = board
        self.config = config
        self.score: float | None = None

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration consistency (double-check runtime safety)."""
        cfg = self.config

        # Skip validation for mock and NN evaluators (they have special rules)
        if cfg.evaluator_type in ("mock", "nn"):
            return

        # Basic validation - material is required for 'complex' or 'handcoded'
        if cfg.evaluator_type in ("complex", "handcoded") and not cfg.use_material:
            raise ValueError(
                "Complex evaluator requires material evaluation (Tree 2 root node E0)"
            )

    @abstractmethod
    def evaluate(self) -> float:
        """
        Evaluate the current board position.

        Returns:
            Evaluation score where:
            - Positive values favor White
            - Negative values favor Black
            - 0.0 indicates equal position
            - Typical range: [-10.0, 10.0] for material-based evaluation

        The implementation should respect the configuration flags and only
        evaluate enabled features according to Tree 2 dependencies.
        """

    # =========================================================================
    # Shared Utility Methods (Available to All Evaluators)
    # =========================================================================

    def get_game_phase(self) -> float:
        """
        Calculate the game phase (0.0 = endgame, 1.0 = opening/midgame).

        Useful for tapered evaluation (Node E2). Based on remaining material.

        Returns:
            Phase value between 0.0 and 1.0
        """
        # Count total material (excluding pawns and kings)
        total_material = 0

        # Queen = 9, Rook = 5, Bishop = 3, Knight = 3
        piece_values = {
            chess.QUEEN: 9,
            chess.ROOK: 5,
            chess.BISHOP: 3,
            chess.KNIGHT: 3,
        }

        for piece_type, value in piece_values.items():
            for color in [chess.WHITE, chess.BLACK]:
                count = len(self.board.pieces(piece_type, color))
                total_material += count * value

        # Maximum material at start: 2Q(18) + 4R(20) + 4B(12) + 4N(12) = 62
        max_material = 62

        # Phase: 1.0 at start (opening), 0.0 when all pieces gone (endgame)
        return min(1.0, total_material / max_material)

    def count_material(self) -> tuple[int, int]:
        """
        Count material for both sides (Node E0 - Tree 2 root).

        Returns:
            Tuple of (white_material, black_material) in centipawns
        """
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,  # King has no material value
        }

        white_material = 0
        black_material = 0

        for piece_type, value in piece_values.items():
            white_count = len(self.board.pieces(piece_type, chess.WHITE))
            black_count = len(self.board.pieces(piece_type, chess.BLACK))

            white_material += white_count * value
            black_material += black_count * value

        return white_material, black_material

    def get_piece_count(self, piece_type: chess.PieceType, color: chess.Color) -> int:
        """
        Count pieces of a specific type and color.

        Args:
            piece_type: Type of piece (PAWN, KNIGHT, etc.)
            color: Color (WHITE or BLACK)

        Returns:
            Number of pieces
        """
        return len(self.board.pieces(piece_type, color))

    def is_endgame(self) -> bool:
        """
        Determine if position is in endgame.

        Simple heuristic: No queens or very little material remaining.
        Useful for endgame-specific logic (Node E9).

        Returns:
            True if in endgame phase
        """
        # No queens on board = definitely endgame
        if not (
            self.board.pieces(chess.QUEEN, chess.WHITE)
            or self.board.pieces(chess.QUEEN, chess.BLACK)
        ):
            return True

        # Or very few pieces remaining
        phase = self.get_game_phase()
        return phase < 0.3

    def get_piece_mobility(self, square: int) -> int:
        """
        Calculate mobility (number of legal moves) for piece on square.

        Useful for mobility evaluation (Node E4).

        Args:
            square: Square containing the piece

        Returns:
            Number of legal moves for the piece
        """
        piece = self.board.piece_at(square)
        if piece is None:
            return 0

        # Count legal moves from this square
        legal_moves = list(self.board.legal_moves)
        moves_from_square = [m for m in legal_moves if m.from_square == square]

        return len(moves_from_square)

    def square_to_index(self, square: int, flip_for_black: bool = True) -> int:
        """
        Convert square to array index, optionally flipping for Black's perspective.

        Useful for PST lookups (Node E1).

        Args:
            square: Chess square (0-63)
            flip_for_black: If True, flip vertically for Black pieces

        Returns:
            Index for PST array access
        """
        if flip_for_black:
            # Flip rank for Black (square 0 becomes 56, 1 becomes 57, etc.)
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            return int((7 - rank) * 8 + file)
        return int(square)

    def interpolate(self, mg_value: float, eg_value: float) -> float:
        """
        Interpolate between middlegame and endgame values (Node E2 - Tapered Eval).

        Args:
            mg_value: Middlegame evaluation
            eg_value: Endgame evaluation

        Returns:
            Interpolated value based on game phase
        """
        if not self.config.use_tapered_eval:
            # If tapered eval disabled, just return middlegame value
            return mg_value

        phase = self.get_game_phase()
        return mg_value * phase + eg_value * (1.0 - phase)


class MockEvaluator(BaseEvaluator):
    """
    Mock evaluator for testing purposes.

    Always returns a fixed value regardless of position.
    """

    def __init__(self, board: chess.Board, config: EvaluationConfig | None = None):
        # Use minimal config for mock evaluator
        if config is None:
            from src.engine.config import EvaluationConfig  # noqa: PLC0415

            config = EvaluationConfig(evaluator_type="mock")
        super().__init__(board, config)

    def evaluate(self) -> float:
        """Return fixed value for testing."""
        return 0.0


class SimpleEvaluator(BaseEvaluator):
    """
    Simple material-only evaluator (Node E0 only).

    Evaluates position based purely on material balance.
    Fast but not very strong.
    """

    def __init__(self, board: chess.Board, config: EvaluationConfig | None = None):
        if config is None:
            from src.engine.config import EvaluationConfig  # noqa: PLC0415

            config = EvaluationConfig(
                evaluator_type="simple",
                use_material=True,
                use_pst=False,
                use_mobility=False,
                use_pawn_structure=False,
                use_king_safety=False,
                use_tapered_eval=False,
                use_bitboards=False,
            )
        super().__init__(board, config)

    def evaluate(self) -> float:
        """Evaluate based on material only."""
        white_material, black_material = self.count_material()

        # Return difference in centipawns, converted to pawns
        return (white_material - black_material) / 100.0

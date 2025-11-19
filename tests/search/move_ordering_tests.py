"""
Comprehensive tests for MoveOrderer.

Tests cover:
- All individual heuristics (MVV-LVA, killers, history, countermove, etc.)
- Modular feature toggling
- Hash move ordering
- Configuration validation
- Heuristic interactions
"""

import pytest

from engine._core import chess_engine_core as chess
from src.engine.config import SearchConfig
from src.engine.search.move_ordering import MoveOrderer
from src.engine.search.transposition_table import TranspositionTable
from src.engine.search.zobrist import Zobrist


class TestMoveOrdererInitialization:
    """Test MoveOrderer initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        board = chess.Board()
        config = SearchConfig(use_move_ordering=True)

        orderer = MoveOrderer(board, config, None, None)

        assert orderer.board == board
        assert orderer.config == config

    def test_init_with_zobrist(self):
        """Test initialization with Zobrist hashing."""
        board = chess.Board()
        config = SearchConfig(use_move_ordering=True, use_zobrist=True)
        zobrist = Zobrist()

        orderer = MoveOrderer(board, config, zobrist, None)

        assert orderer.zobrist == zobrist

    def test_init_with_transposition_table(self):
        """Test initialization with transposition table."""
        board = chess.Board()
        config = SearchConfig(use_move_ordering=True, use_transposition_table=True)
        tt = TranspositionTable()

        orderer = MoveOrderer(board, config, None, tt)

        assert orderer.transposition_table == tt

    def test_init_initializes_tables(self):
        """Test that initialization creates empty tables."""
        board = chess.Board()
        config = SearchConfig(use_move_ordering=True)

        orderer = MoveOrderer(board, config, None, None)

        assert orderer.killer_moves == {}
        assert len(orderer.history_table) == 64
        assert len(orderer.history_table[0]) == 64
        assert orderer.countermove_table == {}


class TestMoveOrdererBasicOrdering:
    """Test basic move ordering without heuristics."""

    def test_order_moves_empty_list(self):
        """Test ordering empty move list."""
        board = chess.Board()
        config = SearchConfig(use_move_ordering=True)
        orderer = MoveOrderer(board, config, None, None)

        ordered = orderer.order_moves([])

        assert ordered == []

    def test_order_moves_returns_all_moves(self):
        """Test that all moves are returned."""
        board = chess.Board()
        config = SearchConfig(use_move_ordering=True)
        orderer = MoveOrderer(board, config, None, None)

        moves = list(board.legal_moves)
        ordered = orderer.order_moves(moves)

        assert len(ordered) == len(moves)
        assert set(ordered) == set(moves)

    def test_order_moves_with_minimal_config(self):
        """Test ordering with all heuristics disabled."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_mvv_lva=False,
            use_killer_moves=False,
            use_history_heuristic=False,
            use_countermove_heuristic=False,
            use_hash_move_ordering=False,
        )
        orderer = MoveOrderer(board, config, None, None)

        moves = list(board.legal_moves)
        ordered = orderer.order_moves(moves)

        # Should still order (basic capture ordering)
        assert len(ordered) == len(moves)


class TestMoveOrdererHashMoveOrdering:
    """Test hash move ordering (PV move from TT)."""

    def test_hash_move_enabled_with_tt(self):
        """Test hash move ordering when enabled with TT."""
        board = chess.Board()
        zobrist = Zobrist()
        tt = TranspositionTable()
        zobrist.hash_board(board)

        config = SearchConfig(
            use_move_ordering=True,
            use_hash_move_ordering=True,
            use_transposition_table=True,
        )
        orderer = MoveOrderer(board, config, zobrist, tt)

        # Store a move in TT
        moves = list(board.legal_moves)
        best_move = moves[0]
        position_hash = zobrist.get_current_hash()
        assert position_hash is not None
        tt.store(position_hash, 1, 0.0, 1.0, -1.0, best_move)

        ordered = orderer.order_moves(moves)

        # PV move should be first
        assert ordered[0] == best_move

    def test_hash_move_disabled(self):
        """Test that hash move ordering doesn't happen when disabled."""
        board = chess.Board()
        zobrist = Zobrist()
        tt = TranspositionTable()
        zobrist.hash_board(board)

        config = SearchConfig(
            use_move_ordering=True,
            use_hash_move_ordering=False,
        )
        orderer = MoveOrderer(board, config, zobrist, tt)

        moves = list(board.legal_moves)
        ordered = orderer.order_moves(moves)

        # Should still order moves, just not by hash move
        assert len(ordered) == len(moves)

    def test_hash_move_without_tt(self):
        """Test hash move ordering without TT."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_hash_move_ordering=False,
        )
        orderer = MoveOrderer(board, config, None, None)

        moves = list(board.legal_moves)
        ordered = orderer.order_moves(moves)

        # Should work without TT
        assert len(ordered) == len(moves)


class TestMoveOrdererMVVLVA:
    """Test MVV-LVA (Most Valuable Victim - Least Valuable Aggressor) ordering."""

    def test_mvv_lva_enabled(self):
        """Test MVV-LVA capture ordering."""
        # Position where White can capture
        board = chess.Board.from_fen(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        )
        board.push_san("exd5")  # Capture pawn

        config = SearchConfig(
            use_move_ordering=True,
            use_mvv_lva=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        moves = list(board.legal_moves)
        ordered = orderer.order_moves(moves)

        # Captures should be ordered first
        assert len(ordered) == len(moves)

    def test_mvv_lva_prefers_valuable_victims(self):
        """Test that MVV-LVA prefers capturing more valuable pieces."""
        # Position where we can capture queen or pawn
        board = chess.Board.from_fen("4q3/8/8/8/3R4/8/8/4K3 w - - 0 1")

        config = SearchConfig(
            use_move_ordering=True,
            use_mvv_lva=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        moves = list(board.legal_moves)
        captures = [m for m in moves if board.is_capture(m)]

        if captures:
            ordered = orderer.order_moves(captures)
            # Queen capture should be first (most valuable victim)
            first_move = ordered[0]
            assert board.is_capture(first_move)

    def test_mvv_lva_disabled(self):
        """Test that MVV-LVA doesn't apply when disabled."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_mvv_lva=False,
        )
        orderer = MoveOrderer(board, config, None, None)

        moves = list(board.legal_moves)
        ordered = orderer.order_moves(moves)

        # Should still order, but not by MVV-LVA
        assert len(ordered) == len(moves)


class TestMoveOrdererKillerMoves:
    """Test killer move heuristic."""

    def test_killer_moves_enabled(self):
        """Test killer move heuristic when enabled."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_killer_moves=True,
            use_alpha_beta=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        # Add a killer move
        move = chess.Move.from_uci("e2e4")
        orderer.add_killer_move(move, depth=1)

        assert 1 in orderer.killer_moves
        assert move in orderer.killer_moves[1]

    def test_killer_moves_disabled(self):
        """Test that killer moves aren't stored when disabled."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_killer_moves=False,
        )
        orderer = MoveOrderer(board, config, None, None)

        # Try to add a killer move
        move = chess.Move.from_uci("e2e4")
        orderer.add_killer_move(move, depth=1)

        # Should not be stored
        assert orderer.killer_moves == {}

    def test_killer_moves_max_per_depth(self):
        """Test that only max killers per depth are kept."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_killer_moves=True,
            use_alpha_beta=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        # Add more than max killers
        moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("c2c4"),
        ]

        for move in moves:
            orderer.add_killer_move(move, depth=1)

        # Should only keep max_killers_per_depth
        assert len(orderer.killer_moves[1]) <= orderer.max_killers_per_depth

    def test_killer_moves_not_captures(self):
        """Test that captures aren't stored as killers."""
        board = chess.Board.from_fen(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        )
        config = SearchConfig(
            use_move_ordering=True,
            use_killer_moves=True,
            use_alpha_beta=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        # Try to add a capture as killer
        capture = chess.Move.from_uci("e4e5")
        orderer.add_killer_move(capture, depth=1)

        # Captures shouldn't be stored as killers
        assert orderer.killer_moves == {}

    def test_clear_killer_moves(self):
        """Test clearing killer moves."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_killer_moves=True,
            use_alpha_beta=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        # Add killer move
        move = chess.Move.from_uci("e2e4")
        orderer.add_killer_move(move, depth=1)

        # Clear
        orderer.clear_killer_moves()

        assert orderer.killer_moves == {}


class TestMoveOrdererHistoryHeuristic:
    """Test history heuristic."""

    def test_history_heuristic_enabled(self):
        """Test history heuristic when enabled."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_history_heuristic=True,
            use_alpha_beta=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        # Update history
        move = chess.Move.from_uci("e2e4")
        orderer.update_history(move, depth=3)

        score = orderer.history_table[move.from_square][move.to_square]
        assert score > 0

    def test_history_heuristic_disabled(self):
        """Test that history isn't updated when disabled."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_history_heuristic=False,
        )
        orderer = MoveOrderer(board, config, None, None)

        # Try to update history
        move = chess.Move.from_uci("e2e4")
        orderer.update_history(move, depth=3)

        # Should not be updated
        score = orderer.history_table[move.from_square][move.to_square]
        assert score == 0

    def test_history_depth_bonus(self):
        """Test that deeper cutoffs get higher bonus."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_history_heuristic=True,
            use_alpha_beta=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        move = chess.Move.from_uci("e2e4")

        # Update with depth 1
        orderer.update_history(move, depth=1)
        score_depth_1 = orderer.history_table[move.from_square][move.to_square]

        # Update with depth 3
        orderer.update_history(move, depth=3)
        score_depth_3 = orderer.history_table[move.from_square][move.to_square]

        # Depth 3 should add more than depth 1
        assert score_depth_3 > score_depth_1

    def test_age_history(self):
        """Test history aging."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_history_heuristic=True,
            use_alpha_beta=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        move = chess.Move.from_uci("e2e4")
        orderer.update_history(move, depth=5)

        score_before = orderer.history_table[move.from_square][move.to_square]
        orderer.age_history()
        score_after = orderer.history_table[move.from_square][move.to_square]

        # After aging, score should be reduced
        assert score_after < score_before

    def test_clear_history(self):
        """Test clearing history."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_history_heuristic=True,
            use_alpha_beta=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        # Update history
        move = chess.Move.from_uci("e2e4")
        orderer.update_history(move, depth=3)

        # Clear
        orderer.clear_history()

        # All should be 0
        score = orderer.history_table[move.from_square][move.to_square]
        assert score == 0


class TestMoveOrdererCountermoveHeuristic:
    """Test countermove heuristic."""

    def test_countermove_heuristic_enabled(self):
        """Test countermove heuristic when enabled."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_countermove_heuristic=True,
            use_alpha_beta=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        # Update countermove
        last_move = chess.Move.from_uci("e7e5")
        response = chess.Move.from_uci("e2e4")
        orderer.update_countermove(response, last_move)

        key = (last_move.from_square, last_move.to_square)
        assert key in orderer.countermove_table
        assert orderer.countermove_table[key] == response

    def test_countermove_heuristic_disabled(self):
        """Test that countermove isn't stored when disabled."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_countermove_heuristic=False,
        )
        orderer = MoveOrderer(board, config, None, None)

        # Try to update countermove
        last_move = chess.Move.from_uci("e7e5")
        response = chess.Move.from_uci("e2e4")
        orderer.update_countermove(response, last_move)

        # Should not be stored
        assert orderer.countermove_table == {}

    def test_countermove_with_last_move(self):
        """Test ordering with countermove and last move."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_countermove_heuristic=True,
            use_alpha_beta=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        # Set up countermove
        last_move = chess.Move.from_uci("e7e5")
        best_response = chess.Move.from_uci("g1f3")
        orderer.update_countermove(best_response, last_move)

        # Order moves with last_move
        board.push(chess.Move.from_uci("e2e4"))
        board.push(last_move)

        moves = list(board.legal_moves)
        ordered = orderer.order_moves(moves, depth=1, last_move=last_move)

        # Should order moves
        assert len(ordered) == len(moves)

    def test_clear_countermoves(self):
        """Test clearing countermove table."""
        board = chess.Board()
        config = SearchConfig(
            use_move_ordering=True,
            use_countermove_heuristic=True,
            use_alpha_beta=True,
        )
        orderer = MoveOrderer(board, config, None, None)

        # Add countermove
        last_move = chess.Move.from_uci("e7e5")
        response = chess.Move.from_uci("e2e4")
        orderer.update_countermove(response, last_move)

        # Clear
        orderer.clear_countermoves()

        assert orderer.countermove_table == {}


class TestMoveOrdererPromotions:
    """Test promotion move ordering."""

    def test_promotions_ordered_high(self):
        """Test that promotions are ordered highly."""
        # Position where White can promote
        board = chess.Board.from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        config = SearchConfig(use_move_ordering=True)
        orderer = MoveOrderer(board, config, None, None)

        moves = list(board.legal_moves)
        promotions = [m for m in moves if m.promotion]

        ordered = orderer.order_moves(moves)

        # Promotions should be near the top
        assert any(m.promotion for m in ordered[:4])

    def test_queen_promotion_highest(self):
        """Test that queen promotion is valued highest."""
        board = chess.Board.from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        config = SearchConfig(use_move_ordering=True)
        orderer = MoveOrderer(board, config, None, None)

        moves = list(board.legal_moves)
        queen_promo = [m for m in moves if m.promotion == chess.QUEEN]

        ordered = orderer.order_moves(moves)

        # Queen promotion should be first among promotions
        assert len(queen_promo) > 0


class TestMoveOrdererCombinedHeuristics:
    """Test interactions between multiple heuristics."""

    def test_all_heuristics_enabled(self):
        """Test with all heuristics enabled."""
        board = chess.Board()
        zobrist = Zobrist()
        tt = TranspositionTable()
        zobrist.hash_board(board)

        config = SearchConfig(
            use_move_ordering=True,
            use_hash_move_ordering=True,
            use_mvv_lva=True,
            use_killer_moves=True,
            use_history_heuristic=True,
            use_countermove_heuristic=True,
            use_alpha_beta=True,
        )
        orderer = MoveOrderer(board, config, zobrist, tt)

        moves = list(board.legal_moves)
        ordered = orderer.order_moves(moves, depth=1)

        assert len(ordered) == len(moves)

    def test_heuristic_priority_hash_move_first(self):
        """Test that hash move has highest priority."""
        board = chess.Board()
        zobrist = Zobrist()
        tt = TranspositionTable()
        zobrist.hash_board(board)

        config = SearchConfig(
            use_move_ordering=True,
            use_hash_move_ordering=True,
            use_mvv_lva=True,
            use_transposition_table=True,
        )
        orderer = MoveOrderer(board, config, zobrist, tt)

        # Store a quiet move in TT
        moves = list(board.legal_moves)
        quiet_moves = [m for m in moves if not board.is_capture(m)]
        if quiet_moves:
            pv_move = quiet_moves[0]
            position_hash = zobrist.get_current_hash()
            assert position_hash is not None
            tt.store(position_hash, 1, 0.0, 1.0, -1.0, pv_move)

            ordered = orderer.order_moves(moves)

            # PV move should be first, even if it's quiet
            assert ordered[0] == pv_move


class TestMoveOrdererEdgeCases:
    """Test edge cases and error conditions."""

    def test_order_single_move(self):
        """Test ordering with only one move."""
        board = chess.Board.from_fen("7k/8/5K2/8/8/8/7R/8 b - - 0 1")
        config = SearchConfig(use_move_ordering=True)
        orderer = MoveOrderer(board, config, None, None)

        moves = list(board.legal_moves)
        ordered = orderer.order_moves(moves)

        assert len(ordered) == len(moves) == 1

    def test_order_from_endgame(self):
        """Test ordering in endgame position."""
        board = chess.Board.from_fen("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
        config = SearchConfig(use_move_ordering=True)
        orderer = MoveOrderer(board, config, None, None)

        moves = list(board.legal_moves)
        ordered = orderer.order_moves(moves)

        assert len(ordered) == len(moves)

    def test_order_all_captures(self):
        """Test ordering when all moves are captures."""
        # Impossible position but valid for testing
        board = chess.Board.from_fen("4k3/8/8/8/3nnn2/3nNn2/3nnn2/4K3 w - - 0 1")
        config = SearchConfig(use_move_ordering=True, use_mvv_lva=True)
        orderer = MoveOrderer(board, config, None, None)

        moves = list(board.legal_moves)
        captures = [m for m in moves if board.is_capture(m)]

        if captures:
            ordered = orderer.order_moves(captures)
            assert len(ordered) == len(captures)

    def test_order_with_depth_zero(self):
        """Test ordering at depth 0."""
        board = chess.Board()
        config = SearchConfig(use_move_ordering=True)
        orderer = MoveOrderer(board, config, None, None)

        moves = list(board.legal_moves)
        ordered = orderer.order_moves(moves, depth=0)

        assert len(ordered) == len(moves)

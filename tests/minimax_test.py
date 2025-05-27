from util import add_project_root_to_path

import chess
from typing import Optional
from src.engine.evaluators.eval import Eval
from src.engine.minimax import Minimax
from src.engine.constants import PIECE_VALUES

class MockEval(Eval):
    """
    A mock evaluator that returns a predefined score.
    Useful for testing purposes.
    """
    def __init__(self, board: chess.Board, fixed_score: float):
        super().__init__(board)
        self.fixed_score = fixed_score
        self.score = fixed_score 

    def evaluate(self) -> float:
        """
        Returns the predefined fixed score.
        """
        self.score = self.fixed_score
        return self.score

    def set_score(self, score: float):
        """
        Allows updating the fixed score after instantiation.
        """
        self.fixed_score = score
        self.score = score




class TestMoveOrdering:
    def test_order_moves_returns_list(self):
        """Test that order_moves returns a list of moves."""
        board = chess.Board()
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)
        
        moves = list(board.legal_moves)
        ordered_moves = minimax.order_moves(moves)
        
        assert isinstance(ordered_moves, list)
        assert len(ordered_moves) == len(moves)
    
    def test_order_moves_with_empty_list(self):
        """Test that order_moves handles empty move list."""
        board = chess.Board()
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)
        
        empty_moves = []
        ordered_moves = minimax.order_moves(empty_moves)
        
        assert ordered_moves == []
    
    def test_order_moves_with_single_move(self):
        """Test that order_moves handles a single move."""
        board = chess.Board("8/8/8/r7/6PK/8/6k1/8 w - - 0 1")  # Only pawn can move
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)
        
        moves = list(board.legal_moves)
        ordered_moves = minimax.order_moves(moves)
        
        assert len(ordered_moves) == len(moves)
        assert ordered_moves[0] in moves
    
    def test_order_moves_with_two_moves(self):
        """Test a pawn move vs capture move."""
        board = chess.Board("8/8/8/r4q2/6PK/8/6k1/8 w - - 0 1") 
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)
        
        expected_ordered_moves = [
            chess.Move.from_uci("g4f5"),  # Capture move (should be prioritized)
            chess.Move.from_uci("g4g5")   # Quiet move
        ]
        
        moves = list(board.legal_moves)
        ordered_moves = minimax.order_moves(moves)
        
        # Assert that the ordered moves match our expectations
        assert ordered_moves == expected_ordered_moves
        assert len(ordered_moves) == 2

    def test_order_moves_with_five_moves(self):
        board = chess.Board("8/8/8/r4q2/6PK/1r6/2P5/7k w - - 0 1")
        
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)
        
        # Get all legal moves and verify we have the expected number
        all_legal_moves = list(board.legal_moves)
        assert len(all_legal_moves) == 5, f"Expected 5 legal moves, got {len(all_legal_moves)}: {[move.uci() for move in all_legal_moves]}"
        
        # Order the moves
        ordered_moves = minimax.order_moves(all_legal_moves)
        assert len(ordered_moves) == 5, f"Expected 5 ordered moves, got {len(ordered_moves)}"
        
        # Define the expected high-priority captures
        queen_capture = chess.Move.from_uci("g4f5")  # Pawn captures queen
        rook_capture = chess.Move.from_uci("c2b3")   # Pawn captures rook
        
        # Verify these moves are actually legal in the position
        assert queen_capture in all_legal_moves, f"Queen capture {queen_capture.uci()} not in legal moves: {[m.uci() for m in all_legal_moves]}"
        assert rook_capture in all_legal_moves, f"Rook capture {rook_capture.uci()} not in legal moves: {[m.uci() for m in all_legal_moves]}"
        
        # The first two moves should be the captures, prioritized by piece value (queen > rook)
        assert ordered_moves[0] == queen_capture, f"First move should be queen capture, got {ordered_moves[0].uci()}"
        assert ordered_moves[1] == rook_capture, f"Second move should be rook capture, got {ordered_moves[1].uci()}"
        
        # The remaining moves should be the quiet moves (order doesn't matter)
        capture_moves = {queen_capture, rook_capture}
        expected_quiet_moves = set(all_legal_moves) - capture_moves
        actual_quiet_moves = set(ordered_moves[2:])
        
        assert actual_quiet_moves == expected_quiet_moves, f"Quiet moves don't match. Expected: {[m.uci() for m in expected_quiet_moves]}, Got: {[m.uci() for m in actual_quiet_moves]}"
            
        
    def test_order_moves_with_check_moves(self):
        """Test that order_moves prioritizes check moves."""
        board = chess.Board("2r5/8/8/5k2/7b/3PP3/7r/3K4 w - - 0 1")
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)
        
        expected_ordered_moves = [
            chess.Move.from_uci("e3e4"),  # Check move
            chess.Move.from_uci("d3d4"),  #  quiet move
        ]
        
        moves = list(board.legal_moves)
        ordered_moves = minimax.order_moves(moves)
        
        # Assert that the ordered moves match our expectations
        assert ordered_moves == expected_ordered_moves
        
        
    def test_order_moves_with_stalemate_moves(self):
        """Test that order_moves handles stalemate moves."""
        board = chess.Board("8/8/8/2R5/2p2kN1/2N4P/PP3PP1/6K1 b - - 0 33")  # Black to move, stalemate
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator) 
        
        moves = list(board.legal_moves)
        ordered_moves = minimax.order_moves(moves)

        assert len(ordered_moves) == 0, "In stalemate, there should be no legal moves."
        
        # In stalemate, all moves are quiet moves, so order doesn't matter
        
    def test_order_moves_with_player_alr_checkmated(self):
        """Test that order_moves handles stalemate moves."""
        board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")  # Black to move, stalemate
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator) 
        
        moves = list(board.legal_moves)
        ordered_moves = minimax.order_moves(moves)

        assert len(ordered_moves) == 0, "In checkmate, there should be no legal moves."
        
        
    def test_complex_move_ordering(self):
        """Test complex move ordering with captures, checks, and quiet moves."""
        board = chess.Board("1k6/8/8/rpp1bp2/1P3P2/3K4/5q2/N6N w - - 0 1")
        evaluator = MockEval(board, 0.0)
        minimax = Minimax(board, evaluator)

        # Get all legal moves and verify we have the expected number
        all_legal_moves = list(board.legal_moves)
        expected_total_moves = 7
        assert len(all_legal_moves) == expected_total_moves, f"Expected {expected_total_moves} legal moves, got {len(all_legal_moves)}: {[move.uci() for move in all_legal_moves]}"
        
        # Order the moves
        ordered_moves = minimax.order_moves(all_legal_moves)
        assert len(ordered_moves) == expected_total_moves, f"Expected {expected_total_moves} ordered moves, got {len(ordered_moves)}"
        
        # Define the expected high-priority captures (in order of piece value)
        expected_priority_captures = [
            chess.Move.from_uci("h1f2"),  # Knight captures queen (highest value)
            chess.Move.from_uci("b4a5"),  # Pawn captures rook
            chess.Move.from_uci("f4e5"),  # Pawn captures bishop
            chess.Move.from_uci("b4c5"),  # Pawn captures pawn (lowest value capture)
        ]
        
        # Verify these moves are actually legal in the position
        for i, move in enumerate(expected_priority_captures):
            assert move in all_legal_moves, f"Priority capture {i+1} ({move.uci()}) not in legal moves: {[m.uci() for m in all_legal_moves]}"
        
        # The first 4 moves should be the captures in exact order (by piece value)
        for i, expected_move in enumerate(expected_priority_captures):
            assert ordered_moves[i] == expected_move, f"Move {i+1} should be {expected_move.uci()}, got {ordered_moves[i].uci()}"
        
        # Define the expected quiet moves (order doesn't matter for these)
        expected_quiet_moves = {
            chess.Move.from_uci("a1b3"),  # Knight move
            chess.Move.from_uci("a1c2"),  # Knight move  
            chess.Move.from_uci("h1g3"),  # Knight move
        }
        
        # Verify quiet moves are legal
        for move in expected_quiet_moves:
            assert move in all_legal_moves, f"Quiet move {move.uci()} not in legal moves: {[m.uci() for m in all_legal_moves]}"
        
        # The remaining moves should be the quiet moves (order doesn't matter)
        priority_capture_set = set(expected_priority_captures)
        calculated_quiet_moves = set(all_legal_moves) - priority_capture_set
        actual_quiet_moves = set(ordered_moves[4:])
        
        # Verify we have the expected quiet moves
        assert actual_quiet_moves == expected_quiet_moves, f"Expected quiet moves: {[m.uci() for m in expected_quiet_moves]}, Got: {[m.uci() for m in actual_quiet_moves]}"
        assert calculated_quiet_moves == expected_quiet_moves, f"Calculated quiet moves don't match expected. Calculated: {[m.uci() for m in calculated_quiet_moves]}, Expected: {[m.uci() for m in expected_quiet_moves]}"
            
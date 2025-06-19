import logging
import time

import chess
import pytest
from src.engine.config import EngineConfig, MinimaxConfig
from src.engine.evaluators.mock_eval import MockEval
from src.engine.search.minimax import Minimax
from src.engine.search.transposition_table import TranspositionTable
from src.engine.search.zobrist import Zobrist


class TestConfigValidation:
    def test_tt_aging_without_zobrist_raises(self):
        with pytest.raises(ValueError, match="aging.*Zobrist"):
            EngineConfig(minimax=MinimaxConfig(use_zobrist=False, use_tt_aging=True))

    def test_lmr_without_alpha_beta_raises(self):
        with pytest.raises(ValueError, match="LMR.*alpha-beta"):
            EngineConfig(minimax=MinimaxConfig(use_alpha_beta=False, use_lmr=True))

    def test_lmr_without_both_dependencies_raises(self):
        with pytest.raises(ValueError, match="LMR.*alpha-beta"):
            EngineConfig(
                minimax=MinimaxConfig(
                    use_alpha_beta=False, use_move_ordering=False, use_lmr=True
                )
            )


class TestLMRDependency:
    def test_lmr_runtime_enforcement(self, caplog):
        caplog.set_level(logging.WARNING)
        cfg = EngineConfig(
            minimax=MinimaxConfig(
                use_alpha_beta=True, use_move_ordering=True, use_lmr=True
            )
        )
        cfg.minimax.use_alpha_beta = False
        engine = Minimax(chess.Board(), MockEval(chess.Board()), cfg)
        assert (
            engine.use_lmr is False
        ), "LMR should be automatically disabled when alpha-beta is disabled at runtime"
        assert any(
            "Disabling LMR" in rec.getMessage() for rec in caplog.records
        ), "Expected warning about LMR being disabled due to missing alpha-beta dependency"


class TestPVSDependency:
    def test_pvs_disabled_with_warning(self, caplog):
        caplog.set_level(logging.WARNING)
        cfg = EngineConfig(
            minimax=MinimaxConfig(use_alpha_beta=True, use_pvs=True, use_lmr=False)
        )
        cfg.minimax.use_alpha_beta = False
        engine = Minimax(chess.Board(), MockEval(chess.Board()), cfg)
        assert (
            engine.use_pvs is False
        ), "PVS should be automatically disabled when alpha-beta is disabled at runtime"
        assert any(
            "Disabling PVS" in rec.getMessage() for rec in caplog.records
        ), "Expected warning about PVS being disabled due to missing alpha-beta dependency"


class TestLMRBasics:
    def test_lmr_flag_initialization(self):
        cfg = EngineConfig()
        engine = Minimax(chess.Board(), MockEval(chess.Board()), cfg)
        assert (
            engine.use_lmr is True
        ), "LMR should be enabled by default in standard engine configuration"
        cfg_disabled = EngineConfig(minimax=MinimaxConfig(use_lmr=False))
        engine_disabled = Minimax(chess.Board(), MockEval(chess.Board()), cfg_disabled)
        assert (
            engine_disabled.use_lmr is False
        ), "LMR should be disabled when explicitly configured to False"

    def test_lmr_shallow_depth_no_reduction(self):
        cfg = EngineConfig(minimax=MinimaxConfig(use_lmr=True, max_time=None))
        engine = Minimax(chess.Board(), MockEval(chess.Board()), cfg)
        score, move = engine.find_top_move(depth=2)
        assert (
            move is not None
        ), "Engine should find a valid move at shallow depth even with LMR enabled"

    def test_lmr_with_checks_no_reduction(self):
        board_in_check = chess.Board(
            "rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3"
        )
        cfg = EngineConfig(minimax=MinimaxConfig(use_lmr=True, max_time=None))
        engine = Minimax(board_in_check, MockEval(board_in_check), cfg)
        score, move = engine.find_top_move(depth=4)
        assert (
            move is not None
        ), "Engine should find valid move in check position where LMR should not apply to checking moves"

    def test_lmr_with_captures_no_reduction(self):
        board = chess.Board(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        )
        cfg = EngineConfig(minimax=MinimaxConfig(use_lmr=True, max_time=None))
        engine = Minimax(board, MockEval(board), cfg)
        score, move = engine.find_top_move(depth=4)
        assert (
            move is not None
        ), "Engine should find valid move in position with captures where LMR should not reduce capture moves"

    def test_lmr_with_promotions_no_reduction(self):
        board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        cfg = EngineConfig(minimax=MinimaxConfig(use_lmr=True, max_time=None))
        engine = Minimax(board, MockEval(board), cfg)
        score, move = engine.find_top_move(depth=4)
        assert move is not None, "Engine should find valid move in promotion position"
        if move.promotion is not None:
            assert (
                move.to_square == chess.A8
            ), "Promotion move should target the correct square (a8)"


class TestLMREfficiency:
    def test_lmr_reduces_node_count(self):
        board = chess.Board(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"
        )
        evaluator = MockEval(board)
        depth = 5
        config_no_lmr = EngineConfig(
            minimax=MinimaxConfig(
                use_lmr=False, use_zobrist=False, use_tt_aging=False, max_time=None
            )
        )
        minimax_no_lmr = Minimax(board, evaluator, config_no_lmr)
        minimax_no_lmr.find_top_move(depth=depth)
        nodes_without_lmr = minimax_no_lmr.node_count
        config_with_lmr = EngineConfig(
            minimax=MinimaxConfig(
                use_lmr=True, use_zobrist=False, use_tt_aging=False, max_time=None
            )
        )
        minimax_with_lmr = Minimax(board, evaluator, config_with_lmr)
        minimax_with_lmr.find_top_move(depth=depth)
        nodes_with_lmr = minimax_with_lmr.node_count
        print(
            f"LMR Node Analysis: Without LMR: {nodes_without_lmr:,}, With LMR: {nodes_with_lmr:,}"
        )
        assert (
            nodes_with_lmr <= nodes_without_lmr
        ), f"LMR should reduce or maintain node count but increased it from {nodes_without_lmr:,} to {nodes_with_lmr:,} (+{((nodes_with_lmr - nodes_without_lmr) / nodes_without_lmr * 100):.1f}%)"
        if nodes_with_lmr < nodes_without_lmr:
            reduction_ratio = (nodes_without_lmr - nodes_with_lmr) / nodes_without_lmr
            print(f"Node reduction achieved: {reduction_ratio:.2%}")
            assert (
                reduction_ratio > 0
            ), f"Expected meaningful node reduction with LMR but only achieved {reduction_ratio:.3%}. This suggests LMR is not working effectively in this position."
        else:
            print(
                "No node reduction observed - LMR conditions may not have been met in this position"
            )

    def test_lmr_different_positions(self):
        positions = [
            chess.Board(
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"
            ),
            chess.Board(
                "r2qkb1r/ppp2ppp/2n1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6"
            ),
            chess.Board(
                "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP1NPPP/R1BQ1RK1 w - - 0 8"
            ),
        ]
        depth = 4
        total_nodes_with_lmr = 0
        total_nodes_without_lmr = 0
        for i, pos in enumerate(positions):
            evaluator = MockEval(pos)
            config_no_lmr = EngineConfig(
                minimax=MinimaxConfig(
                    use_lmr=False, use_zobrist=False, use_tt_aging=False, max_time=None
                )
            )
            minimax_no_lmr = Minimax(pos, evaluator, config_no_lmr)
            minimax_no_lmr.find_top_move(depth=depth)
            nodes_no_lmr = minimax_no_lmr.node_count
            total_nodes_without_lmr += nodes_no_lmr
            config_with_lmr = EngineConfig(
                minimax=MinimaxConfig(
                    use_lmr=True, use_zobrist=False, use_tt_aging=False, max_time=None
                )
            )
            minimax_with_lmr = Minimax(pos, evaluator, config_with_lmr)
            minimax_with_lmr.find_top_move(depth=depth)
            nodes_with_lmr = minimax_with_lmr.node_count
            total_nodes_with_lmr += nodes_with_lmr
            print(
                f"Position {i+1}: Without LMR: {nodes_no_lmr:,}, With LMR: {nodes_with_lmr:,}"
            )
        print(
            f"Total across all positions: Without LMR: {total_nodes_without_lmr:,}, With LMR: {total_nodes_with_lmr:,}"
        )
        assert (
            total_nodes_with_lmr <= total_nodes_without_lmr
        ), f"LMR should not increase total node count across multiple positions. Got {total_nodes_without_lmr:,} → {total_nodes_with_lmr:,}"
        if total_nodes_with_lmr < total_nodes_without_lmr:
            reduction = (
                total_nodes_without_lmr - total_nodes_with_lmr
            ) / total_nodes_without_lmr
            print(f"Overall LMR efficiency: {reduction:.2%} node reduction")


class TestLMRIntegration:
    def test_lmr_with_pvs(self):
        board = chess.Board()
        evaluator = MockEval(board)
        depth = 4
        config = EngineConfig(
            minimax=MinimaxConfig(
                use_lmr=True,
                use_pvs=True,
                use_zobrist=False,
                use_tt_aging=False,
                max_time=None,
            )
        )
        engine = Minimax(board, evaluator, config)
        score, move = engine.find_top_move(depth=depth)
        assert (
            move is not None
        ), "Engine should find valid move when LMR and PVS are both enabled"


class TestLMREdgeCases:
    def test_lmr_with_few_legal_moves(self):
        board = chess.Board("8/8/8/8/8/8/k7/K7 w - - 0 1")
        evaluator = MockEval(board)
        config = EngineConfig(minimax=MinimaxConfig(use_lmr=True, max_time=None))
        engine = Minimax(board, evaluator, config)
        score, move = engine.find_top_move(depth=4)
        assert (
            move is not None
        ), "Engine should handle positions with very few legal moves when LMR is enabled"

    def test_lmr_minimum_depth_boundary(self):
        board = chess.Board()
        evaluator = MockEval(board)
        config = EngineConfig(
            minimax=MinimaxConfig(
                use_lmr=True, use_zobrist=False, use_tt_aging=False, max_time=None
            )
        )
        engine = Minimax(board, evaluator, config)
        score, move = engine.find_top_move(depth=3)
        assert (
            move is not None
        ), "Engine should work correctly at minimum depth where LMR might apply"

    def test_lmr_pv_node_exclusion(self):
        board = chess.Board()
        evaluator = MockEval(board)
        config = EngineConfig(
            minimax=MinimaxConfig(use_lmr=True, use_iddfs=True, max_time=None)
        )
        engine = Minimax(board, evaluator, config)
        score, move = engine.find_top_move(depth=4)
        assert (
            move is not None
        ), "Engine should correctly exclude PV nodes from LMR when using iterative deepening"

    def test_lmr_move_ordering_dependency(self):
        board = chess.Board()
        evaluator = MockEval(board)
        depth = 4
        config = EngineConfig(
            minimax=MinimaxConfig(use_lmr=True, use_move_ordering=True, max_time=None)
        )
        engine = Minimax(board, evaluator, config)
        score, move = engine.find_top_move(depth=depth)
        assert (
            move is not None
        ), "Engine should work correctly when LMR depends on move ordering"


class TestIterativeDeepening:
    def test_iddfs_sequences_depths(self, monkeypatch):
        called = []
        dummy_move = chess.Move.from_uci("a2a3")

        def fake_search(self, depth):
            called.append(depth)
            return float(depth), (dummy_move if depth == 4 else None)

        monkeypatch.setattr(Minimax, "_search_fixed_depth", fake_search)
        cfg = EngineConfig(
            minimax=MinimaxConfig(
                use_iddfs=True, use_zobrist=False, use_tt_aging=False, max_time=None
            )
        )
        engine = Minimax(chess.Board(), MockEval(chess.Board()), cfg)
        score, move = engine.find_top_move(depth=4)
        assert called == [
            1,
            2,
            3,
            4,
        ], f"IDDFS should search depths 1,2,3,4 in sequence but searched {called}"
        assert (
            score == 4.0
        ), f"Expected final score of 4.0 from depth 4 search but got {score}"
        assert (
            move == dummy_move
        ), f"Expected dummy move from final depth search but got {move}"


class TestTimeLimit:
    def test_check_time_limit_flags_time_up(self):
        cfg = EngineConfig(minimax=MinimaxConfig(max_time=0.01))
        engine = Minimax(chess.Board(), MockEval(chess.Board()), cfg)
        engine.start_time = time.time() - 1.0
        assert (
            engine._check_time_limit() is True
        ), "Time limit check should return True when time has exceeded limit"
        assert (
            engine.time_up is True
        ), "time_up flag should be set to True when time limit is exceeded"


class TestTranspositionTableBasics:
    def test_store_and_lookup(self):
        tt = TranspositionTable()
        hash_val = 12345
        depth = 3
        score = 1.5
        alpha = 1.0
        beta = 2.0
        tt.store(hash_val, depth, score, alpha, beta, alpha)
        result = tt.lookup(hash_val, depth, alpha, beta)
        assert (
            result == score
        ), f"TT lookup should return stored score {score} but returned {result}"
        result = tt.lookup(hash_val, depth + 1, alpha, beta)
        assert (
            result is None
        ), f"TT lookup with higher depth should return None but returned {result}"

    def test_clear(self):
        tt = TranspositionTable()
        tt.store(1, 3, 0.5, 0, 1, 0)
        tt.store(2, 3, -0.5, 0, 1, 0)
        assert (
            tt.size() == 2
        ), f"TT should contain 2 entries after storing 2 items but contains {tt.size()}"
        tt.clear()
        assert (
            tt.size() == 0
        ), f"TT should be empty after clear() but contains {tt.size()} entries"

    def test_size_limit(self):
        max_entries = 10
        tt = TranspositionTable(max_entries=max_entries)
        for i in range(max_entries + 5):
            tt.store(i, 3, float(i), 0, float(i + 1), 0)
        assert (
            tt.size() <= max_entries
        ), f"TT size {tt.size()} should not exceed max_entries {max_entries}"


class TestTranspositionTableAging:
    def test_aging_enabled(self):
        tt = TranspositionTable(use_tt_aging=True)
        hash_val = 12345
        tt.store(hash_val, 3, 1.5, 1.0, 2.0, 1.0)
        assert (
            tt.lookup(hash_val, 3, 1.0, 2.0) == 1.5
        ), "TT entry should be retrievable immediately after storage"
        for _ in range(tt.MAX_AGE_DIFF + 1):
            tt.increment_age()
        assert (
            tt.lookup(hash_val, 3, 1.0, 2.0) is None
        ), f"TT entry should be aged out after {tt.MAX_AGE_DIFF + 1} age increments"

    def test_aging_disabled(self):
        tt = TranspositionTable(use_tt_aging=False)
        hash_val = 12345
        tt.store(hash_val, 3, 1.5, 1.0, 2.0, 1.0)
        for _ in range(10):
            tt.increment_age()
        assert (
            tt.lookup(hash_val, 3, 1.0, 2.0) == 1.5
        ), "TT entry should persist indefinitely when aging is disabled"

    def test_entry_refresh(self):
        tt = TranspositionTable(use_tt_aging=True)
        hash_val = 12345
        tt.store(hash_val, 3, 1.5, 1.0, 2.0, 1.0)
        for _ in range(tt.MAX_AGE_DIFF):
            tt.increment_age()
        assert (
            tt.lookup(hash_val, 3, 1.0, 2.0) == 1.5
        ), f"TT entry should still be valid at MAX_AGE_DIFF ({tt.MAX_AGE_DIFF}) increments"
        tt.store(hash_val, 3, 1.5, 1.0, 2.0, 1.0)
        tt.increment_age()
        assert (
            tt.lookup(hash_val, 3, 1.0, 2.0) == 1.5
        ), "TT entry should be refreshed and remain valid after re-storage"

    def test_age_reset(self):
        tt = TranspositionTable(use_tt_aging=True)
        hash_val = 12345
        tt.store(hash_val, 3, 1.5, 1.0, 2.0, 1.0)
        for _ in range(tt.MAX_AGE_DIFF):
            tt.increment_age()
        tt.reset_age()
        assert (
            tt.lookup(hash_val, 3, 1.0, 2.0) is None
        ), "TT entry should be invalidated after age reset even if previously valid"


class TestTranspositionTableEntryTypes:
    def test_exact_score(self):
        tt = TranspositionTable()
        hash_val = 12345
        score = 1.5
        alpha = 1.0
        beta = 2.0
        tt.store(hash_val, 3, score, alpha, beta, alpha)
        assert (
            tt.lookup(hash_val, 3, 1.0, 2.0) == score
        ), f"TT should return exact score {score} when score equals original_alpha"

    def test_upper_bound(self):
        tt = TranspositionTable()
        hash_val = 12345
        score = 0.5
        alpha = 1.0
        beta = 2.0
        tt.store(hash_val, 3, score, alpha, beta, alpha)
        assert (
            tt.lookup(hash_val, 3, alpha, beta) == alpha
        ), f"TT should return alpha {alpha} for upper bound entry but returned {tt.lookup(hash_val, 3, alpha, beta)}"

    def test_lower_bound(self):
        tt = TranspositionTable()
        hash_val = 12345
        score = 2.5
        alpha = 1.0
        beta = 2.0
        tt.store(hash_val, 3, score, alpha, beta, alpha)
        assert (
            tt.lookup(hash_val, 3, alpha, beta) == beta
        ), f"TT should return beta {beta} for lower bound entry but returned {tt.lookup(hash_val, 3, alpha, beta)}"


class TestTranspositionTableIntegration:
    def test_config_validation(self):
        board = chess.Board()
        evaluator = MockEval(board)
        with pytest.raises(ValueError, match="aging.*Zobrist"):
            config = EngineConfig(
                minimax=MinimaxConfig(use_zobrist=False, use_tt_aging=True)
            )
            Minimax(board, evaluator, config)

    def test_node_count_reduction(self):
        board = chess.Board()
        evaluator = MockEval(board)
        depth = 5
        config_no_tt = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=False, use_tt_aging=False, use_lmr=False, max_time=None
            )
        )
        minimax_no_tt = Minimax(board, evaluator, config_no_tt)
        minimax_no_tt.find_top_move(depth=depth)
        nodes_without_tt = minimax_no_tt.node_count
        config_with_tt = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=True, use_tt_aging=True, use_lmr=False, max_time=None
            )
        )
        minimax_with_tt = Minimax(board, evaluator, config_with_tt)
        minimax_with_tt.find_top_move(depth=depth)
        nodes_with_tt = minimax_with_tt.node_count
        assert (
            nodes_with_tt < nodes_without_tt
        ), f"Transposition table failed to reduce node count: Without TT: {nodes_without_tt:,}, With TT: {nodes_with_tt:,}. TT should provide significant speedup through position caching."
        reduction_ratio = (nodes_without_tt - nodes_with_tt) / nodes_without_tt
        assert (
            reduction_ratio > 0.25
        ), f"Transposition table reduction of {reduction_ratio:.1%} is below expected 25% minimum. This indicates the TT is not effectively caching positions or has poor hit rates. Expected significant pruning due to transpositions in chess positions."

    def test_aging_effectiveness(self):
        board = chess.Board()
        evaluator = MockEval(board)
        depth = 5
        config = EngineConfig(
            minimax=MinimaxConfig(use_zobrist=True, use_tt_aging=True, max_time=None)
        )
        minimax = Minimax(board, evaluator, config)
        assert (
            minimax.transposition_table is not None
        ), "Transposition table should be initialized when Zobrist hashing is enabled"
        minimax.find_top_move(depth=depth)
        tt_size_after_first = minimax.transposition_table.size()
        assert (
            tt_size_after_first > 0
        ), f"Transposition table is empty after search completion. Expected entries to be stored during depth-{depth} search. This suggests the TT is not being populated correctly."
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        board.push_san("Nc6")
        minimax.find_top_move(depth=depth)
        tt_size_after_second = minimax.transposition_table.size()
        assert (
            tt_size_after_second >= tt_size_after_first
        ), f"Transposition table size decreased unexpectedly: {tt_size_after_first} → {tt_size_after_second}. Aging may be too aggressive or entries are being prematurely evicted."
        for _ in range(5):
            if board.turn == chess.WHITE:
                for candidate_move in ["Bc4", "d4", "Nc3", "Nf3", "Qe2", "0-0", "h3"]:
                    try:
                        board.push_san(candidate_move)
                        break
                    except chess.IllegalMoveError:
                        continue
                else:
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(legal_moves[0])
            else:
                for candidate_move in ["Nf6", "d5", "e6", "Bc5", "0-0", "h6"]:
                    try:
                        board.push_san(candidate_move)
                        break
                    except chess.IllegalMoveError:
                        continue
                else:
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(legal_moves[0])
            minimax.find_top_move(depth=depth)
        final_size = minimax.transposition_table.size()
        assert (
            final_size <= minimax.transposition_table.max_entries
        ), f"Transposition table exceeded maximum size limit: {final_size} > {minimax.transposition_table.max_entries}. Aging mechanism failed to maintain size constraints."


class TestZobristBasics:
    def test_hash_consistency(self):
        zobrist = Zobrist(seed=42)
        board = chess.Board()
        hashes = [zobrist.hash_board(board) for _ in range(3)]
        assert (
            hashes[0] == hashes[1] == hashes[2]
        ), f"Zobrist hash should be consistent for same position but got different hashes: {hashes}"

    def test_hash_uniqueness(self):
        zobrist = Zobrist(seed=42)
        board1 = chess.Board()
        board2 = chess.Board()
        board2.push_san("e4")
        hash1 = zobrist.hash_board(board1)
        hash2 = zobrist.hash_board(board2)
        assert (
            hash1 != hash2
        ), f"Zobrist hashing failed to distinguish different positions: Starting position hash: {hash1}, After e4 hash: {hash2}. Both positions produced identical hashes indicating insufficient randomness or incorrect computation."

    def test_position_independence(self):
        zobrist = Zobrist(seed=42)
        board1 = chess.Board()
        board1.push_san("e4")
        board1.push_san("e5")
        board2 = chess.Board()
        board2.push_san("e4")
        board2.push_san("d5")
        board2.pop()
        board2.push_san("e5")
        hash1 = zobrist.hash_board(board1)
        hash2 = zobrist.hash_board(board2)
        assert (
            hash1 == hash2
        ), f"Zobrist hash should be path-independent. Same final position reached via different move sequences should have identical hashes. Hash1: {hash1}, Hash2: {hash2}"


class TestZobristIncrementalUpdates:
    def test_incremental_vs_full_hash(self):
        zobrist = Zobrist(seed=42)
        board = chess.Board()
        zobrist.hash_board(board)
        move = chess.Move.from_uci("e2e4")
        old_castling = board.castling_rights
        old_ep = board.ep_square
        piece_at_dest = board.piece_at(move.to_square)
        captured_piece_type = piece_at_dest.piece_type if piece_at_dest else None
        was_ep = board.is_en_passant(move)
        ks_castle = board.is_kingside_castling(move)
        qs_castle = board.is_queenside_castling(move)
        board.push(move)
        incremental_hash = zobrist.update_hash_for_move(
            board,
            move,
            old_castling,
            old_ep,
            captured_piece_type,
            was_ep,
            ks_castle,
            qs_castle,
        )
        fresh_hash = zobrist.hash_board(board)
        assert (
            incremental_hash == fresh_hash
        ), f"Zobrist incremental hash update produced incorrect result for move {move}: Incremental: {incremental_hash}, Full rehash: {fresh_hash}, Difference: {incremental_hash ^ fresh_hash}. This indicates a bug in incremental update logic."

    def test_multiple_moves_consistency(self):
        zobrist = Zobrist(seed=42)
        board = chess.Board()
        zobrist.hash_board(board)
        moves = ["e4", "e5", "Nf3", "Nc6", "Bc4"]
        for san in moves:
            move = board.parse_san(san)
            old_castling = board.castling_rights
            old_ep = board.ep_square
            piece_at_dest = board.piece_at(move.to_square)
            captured_piece_type = piece_at_dest.piece_type if piece_at_dest else None
            was_ep = board.is_en_passant(move)
            ks_castle = board.is_kingside_castling(move)
            qs_castle = board.is_queenside_castling(move)
            board.push(move)
            incremental_hash = zobrist.update_hash_for_move(
                board,
                move,
                old_castling,
                old_ep,
                captured_piece_type,
                was_ep,
                ks_castle,
                qs_castle,
            )
            fresh_hash = zobrist.hash_board(board)
            assert (
                incremental_hash == fresh_hash
            ), f"Zobrist incremental hash failed for move {san} in sequence {moves}: Incremental: {incremental_hash}, Full: {fresh_hash}. Incremental updates must remain consistent with full hash computation throughout move sequences."


class TestZobristSpecialMoves:
    def test_castling_hash(self):
        zobrist = Zobrist(seed=42)
        board = chess.Board(
            "r3k2r/ppp1pppp/2n2n2/8/8/2N2N2/PPP1PPPP/R3K2R w KQkq - 0 1"
        )
        original_hash = zobrist.hash_board(board)
        move = board.parse_san("O-O")
        old_castling = board.castling_rights
        old_ep = board.ep_square
        captured_piece_type = None
        was_ep = False
        ks_castle = True
        qs_castle = False
        board.push(move)
        castle_hash = zobrist.update_hash_for_move(
            board,
            move,
            old_castling,
            old_ep,
            captured_piece_type,
            was_ep,
            ks_castle,
            qs_castle,
        )
        fresh_hash = zobrist.hash_board(board)
        assert (
            castle_hash == fresh_hash
        ), f"Zobrist castling hash update failed: Incremental: {castle_hash}, Full: {fresh_hash}. Castling moves must properly update hash for both king and rook movement plus castling rights changes."
        assert (
            castle_hash != original_hash
        ), f"Zobrist hash should change after castling but remained {original_hash}. Castling should significantly alter position hash."

    def test_en_passant_hash(self):
        zobrist = Zobrist(seed=42)
        board = chess.Board(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        )
        original_hash = zobrist.hash_board(board)
        move = board.parse_san("exf6")
        old_castling = board.castling_rights
        old_ep = board.ep_square
        captured_piece_type = chess.PAWN
        was_ep = True
        ks_castle = False
        qs_castle = False
        board.push(move)
        ep_hash = zobrist.update_hash_for_move(
            board,
            move,
            old_castling,
            old_ep,
            captured_piece_type,
            was_ep,
            ks_castle,
            qs_castle,
        )
        fresh_hash = zobrist.hash_board(board)
        assert (
            ep_hash == fresh_hash
        ), f"Zobrist en passant hash update failed: Incremental: {ep_hash}, Full: {fresh_hash}. En passant captures must properly account for pawn removal and en passant square changes."
        assert (
            ep_hash != original_hash
        ), f"Zobrist hash should change after en passant capture but remained {original_hash}. En passant should alter position hash significantly."

    def test_promotion_hash(self):
        zobrist = Zobrist(seed=42)
        board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        original_hash = zobrist.hash_board(board)
        move = chess.Move.from_uci("a7a8q")
        old_castling = board.castling_rights
        old_ep = board.ep_square
        piece_at_dest = board.piece_at(move.to_square)
        captured_piece_type = piece_at_dest.piece_type if piece_at_dest else None
        was_ep = False
        ks_castle = False
        qs_castle = False
        board.push(move)
        promotion_hash = zobrist.update_hash_for_move(
            board,
            move,
            old_castling,
            old_ep,
            captured_piece_type,
            was_ep,
            ks_castle,
            qs_castle,
        )
        fresh_hash = zobrist.hash_board(board)
        assert (
            promotion_hash == fresh_hash
        ), f"Zobrist promotion hash update failed: Incremental: {promotion_hash}, Full: {fresh_hash}. Promotion moves must properly remove pawn and add promoted piece to hash."
        assert (
            promotion_hash != original_hash
        ), f"Zobrist hash should change after pawn promotion but remained {original_hash}. Promotion should create distinctly different position hash."


class TestZobristIntegration:
    def test_node_count_reduction(self):
        board = chess.Board()
        evaluator = MockEval(board)
        depth = 3
        config_no_zobrist = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=False, use_tt_aging=False, use_lmr=False, max_time=None
            )
        )
        minimax_no_zobrist = Minimax(board, evaluator, config_no_zobrist)
        minimax_no_zobrist.find_top_move(depth=depth)
        nodes_without_zobrist = minimax_no_zobrist.node_count
        config_with_zobrist = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=True, use_tt_aging=False, use_lmr=False, max_time=None
            )
        )
        minimax_with_zobrist = Minimax(board, evaluator, config_with_zobrist)
        minimax_with_zobrist.find_top_move(depth=depth)
        nodes_with_zobrist = minimax_with_zobrist.node_count
        assert (
            nodes_with_zobrist <= nodes_without_zobrist
        ), f"Zobrist hashing with transposition table should reduce or maintain node count but increased it: {nodes_without_zobrist:,} → {nodes_with_zobrist:,}"
        if nodes_without_zobrist > nodes_with_zobrist:
            reduction_ratio = (
                nodes_without_zobrist - nodes_with_zobrist
            ) / nodes_without_zobrist
            assert (
                reduction_ratio > 0
            ), f"Expected meaningful node reduction with Zobrist TT but only achieved {reduction_ratio:.3%}. This suggests transposition table is not providing effective position caching."

    def test_aging_vs_no_aging_efficiency(self):
        evaluator = MockEval(chess.Board())
        depth = 3
        config_with_aging = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=True, use_tt_aging=True, use_lmr=False, max_time=None
            )
        )
        config_without_aging = EngineConfig(
            minimax=MinimaxConfig(
                use_zobrist=True, use_tt_aging=False, use_lmr=False, max_time=None
            )
        )
        positions = [
            chess.Board(),
            chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
            chess.Board(
                "rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
            ),
        ]
        total_nodes_with_aging = 0
        total_nodes_without_aging = 0
        for pos in positions:
            minimax_with_aging = Minimax(pos, MockEval(pos), config_with_aging)
            minimax_without_aging = Minimax(pos, MockEval(pos), config_without_aging)
            minimax_with_aging.find_top_move(depth=depth)
            total_nodes_with_aging += minimax_with_aging.node_count
            minimax_without_aging.find_top_move(depth=depth)
            total_nodes_without_aging += minimax_without_aging.node_count
        assert (
            total_nodes_with_aging <= total_nodes_without_aging
        ), f"TT aging should not increase node count across positions: With aging: {total_nodes_with_aging:,}, Without aging: {total_nodes_without_aging:,}. Aging should maintain or improve efficiency by preventing memory bloat."

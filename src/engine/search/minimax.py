"""Negamax-based search implementation for the Python engine."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from engine._core import chess_engine_core as chess
from engine.search.move_ordering import MoveSorter
from engine.search.stats import SearchStats
from engine.search.transposition_table import BoundType, TranspositionTable
from engine.search.zobrist import Zobrist

if TYPE_CHECKING:
    from engine.config import EngineConfig
    from engine.evaluators import Evaluator


class Minimax:
    """Config-driven negamax searcher with optional engine optimizations."""

    NEG_INF = float("-inf")
    POS_INF = float("inf")
    MATE_SCORE = 100_000
    TIME_CHECK_INTERVAL = 2048

    def __init__(
        self,
        board: chess.Board,
        evaluator: Evaluator,
        config: EngineConfig,
    ):
        """Set up search state, TT, Zobrist hasher, and move sorter from config."""
        self.board = board
        self.evaluator = evaluator
        self.config = config
        self.search_cfg = config.search

        self.stats = SearchStats()
        self.node_count = 0
        self.time_up = False
        self.start_time: float | None = None

        self.zobrist: Zobrist | None = None
        self.tt: TranspositionTable | None = None
        if self.search_cfg.use_transposition_table:
            self.zobrist = Zobrist()
            self.tt = TranspositionTable(self.search_cfg)
            self.zobrist.hash_board(self.board)

        self.move_sorter: MoveSorter | None = None
        if self.search_cfg.use_move_ordering:
            self.move_sorter = MoveSorter(self.search_cfg)

        self.root_best_move: chess.Move | None = None

    def reset_state(
        self,
        clear_tt: bool = True,
        clear_history: bool = True,
        clear_killers: bool = True,
    ) -> None:
        """Reset search state; optionally preserve TT, history, and killer tables."""
        if clear_tt and self.tt is not None:
            self.tt.clear()

        if self.move_sorter is not None:
            self.move_sorter.reset(
                clear_history=clear_history,
                clear_killers=clear_killers,
            )

        self.stats.reset()
        self.node_count = 0
        self.root_best_move = None

    def find_best_move(
        self,
        depth: int | None = None,
    ) -> tuple[float | None, chess.Move | None]:
        """Run IDDFS up to *depth*.

        Returns the best (score, move) pair from White's perspective.
        """
        target_depth = max(1, depth if depth is not None else self.config.search_depth)

        self.stats.reset()
        self.node_count = 0
        self.time_up = False
        self.start_time = time.time()
        self.root_best_move = None

        if self.tt is not None:
            self.tt.increment_age()

        if self.zobrist is not None:
            self.zobrist.hash_board(self.board)

        root_turn_is_white = bool(self.board.turn)
        previous_score: float | None = None
        final_relative_score: float | None = None

        for current_depth in range(1, target_depth + 1):
            if self._check_time_limit():
                break

            alpha = self.NEG_INF
            beta = self.POS_INF
            if (
                self.search_cfg.use_alpha_beta
                and self.search_cfg.use_aspiration_windows
                and previous_score is not None
            ):
                margin = max(10, self.search_cfg.aspiration_window_margin)
                alpha = previous_score - margin
                beta = previous_score + margin

            relative_score = self._search_with_window(
                depth=current_depth,
                alpha=alpha,
                beta=beta,
            )
            if self.time_up:
                break

            previous_score = relative_score
            final_relative_score = relative_score
            self.stats.depth = current_depth

        if final_relative_score is None:
            return None, None

        if self.tt is not None:
            self.stats.hashfull = int((self.tt.size() * 1000) / self.tt.max_entries)

        if self.move_sorter is not None:
            self.stats.history_saturation = self.move_sorter.history_saturation()

        white_score = (
            final_relative_score if root_turn_is_white else -final_relative_score
        )
        self.stats.score = int(white_score)
        self.node_count = self.stats.nodes
        return white_score, self.root_best_move

    def find_top_move(self, depth: int = 1) -> tuple[float | None, chess.Move | None]:
        """Backward-compatible alias for previous API."""
        return self.find_best_move(depth)

    def _search_with_window(self, depth: int, alpha: float, beta: float) -> float:
        """Run negamax with optional aspiration window.

        Widens the window up to 6 times on fail-high or fail-low.
        """
        if (
            not (
                self.search_cfg.use_alpha_beta
                and self.search_cfg.use_aspiration_windows
            )
            or alpha == self.NEG_INF
            or beta == self.POS_INF
        ):
            return self._negamax(
                depth=depth,
                alpha=self.NEG_INF if not self.search_cfg.use_alpha_beta else alpha,
                beta=self.POS_INF if not self.search_cfg.use_alpha_beta else beta,
                ply=0,
                previous_move=None,
                extensions_left=self.search_cfg.max_check_extensions,
            )

        current_alpha = alpha
        current_beta = beta

        for _ in range(6):
            score = self._negamax(
                depth=depth,
                alpha=current_alpha,
                beta=current_beta,
                ply=0,
                previous_move=None,
                extensions_left=self.search_cfg.max_check_extensions,
            )
            if self.time_up:
                return score
            if score <= current_alpha:
                current_alpha -= max(50, self.search_cfg.aspiration_window_margin)
                continue
            if score >= current_beta:
                current_beta += max(50, self.search_cfg.aspiration_window_margin)
                continue
            return score

        return self._negamax(
            depth=depth,
            alpha=self.NEG_INF,
            beta=self.POS_INF,
            ply=0,
            previous_move=None,
            extensions_left=self.search_cfg.max_check_extensions,
        )

    def _negamax(  # noqa: C901, PLR0911, PLR0912
        self,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        previous_move: chess.Move | None,
        extensions_left: int,
    ) -> float:
        """Core negamax with TT, RFP, NMP, IID, futility, LMR, PVS, check extensions."""
        self.stats.nodes += 1
        self.stats.seldepth = max(self.stats.seldepth, ply)

        if (
            self.stats.nodes % self.TIME_CHECK_INTERVAL == 0
            and self._check_time_limit()
        ):
            return self._relative_eval()

        game_state = self.board.is_game_over()
        if game_state != chess.GameState.ONGOING:
            return self._terminal_score(game_state, ply)

        if depth <= 0:
            if self.search_cfg.use_quiescence_search:
                return self._quiescence(alpha, beta, ply, 0)
            return self._relative_eval()

        in_check = bool(self.board.is_check())
        if (
            self.search_cfg.use_check_extensions
            and in_check
            and extensions_left > 0
            and self.search_cfg.use_alpha_beta
        ):
            depth += 1
            extensions_left -= 1
            self.stats.check_extensions += 1

        key = self._current_hash()
        hash_move: chess.Move | None = None
        if self.tt is not None and key is not None:
            entry = self.tt.probe(key)
            if entry is not None:
                hash_move = entry.best_move
                if self.search_cfg.use_alpha_beta:
                    hit_score = self.tt.try_get_score(entry, depth, alpha, beta)
                    if hit_score is not None:
                        self.stats.tt_hits += 1
                        return float(hit_score)
                elif entry.bound == "exact" and entry.depth >= depth:
                    self.stats.tt_hits += 1
                    return float(entry.score)

        static_eval = self._relative_eval()

        if (
            self.search_cfg.use_alpha_beta
            and self.search_cfg.use_reverse_futility_pruning
            and not in_check
            and depth <= self.search_cfg.rfp_max_depth
            and beta < self.POS_INF
        ):
            margin = self.search_cfg.rfp_margin_multiplier * depth
            if static_eval - margin >= beta:
                return beta

        if (
            self.search_cfg.use_alpha_beta
            and self.search_cfg.use_null_move_pruning
            and not in_check
            and depth >= self.search_cfg.nmp_min_depth
            and self._has_non_pawn_material()
            and beta < self.POS_INF
        ):
            null_score = self._null_move_search(depth, beta, ply, extensions_left)
            if null_score >= beta:
                self.stats.null_move_cuts += 1
                return beta

        if (
            self.search_cfg.use_iid
            and self.search_cfg.use_alpha_beta
            and depth >= self.search_cfg.iid_min_depth
            and hash_move is None
            and self.tt is not None
            and key is not None
        ):
            self.stats.iid_searches += 1
            shallow_depth = max(1, depth - self.search_cfg.iid_depth_reduction)
            self._negamax(
                depth=shallow_depth,
                alpha=alpha,
                beta=beta,
                ply=ply,
                previous_move=previous_move,
                extensions_left=extensions_left,
            )
            iid_entry = self.tt.probe(key)
            if iid_entry is not None:
                hash_move = iid_entry.best_move

        legal_moves = list(self.board.generate_legal_moves())
        if not legal_moves:
            return -self.MATE_SCORE + ply if in_check else 0.0

        if self.move_sorter is not None:
            legal_moves = self.move_sorter.sort_moves(
                board=self.board,
                moves=legal_moves,
                ply=ply,
                hash_move=hash_move,
                previous_move=previous_move,
            )

        original_alpha = alpha
        best_score = self.NEG_INF
        best_move: chess.Move | None = None

        for index, move in enumerate(legal_moves):
            if self.time_up:
                break

            is_tactical = self._is_tactical_move(move)
            if self._can_apply_futility(
                depth,
                static_eval,
                alpha,
                in_check,
                is_tactical,
            ):
                continue

            saved_hash = self._push_move_with_hash(move)
            gives_check = bool(self.board.is_check())

            child_extensions = extensions_left
            next_depth = depth - 1
            if (
                self.search_cfg.use_check_extensions
                and gives_check
                and child_extensions > 0
                and self.search_cfg.use_alpha_beta
            ):
                next_depth += 1
                child_extensions -= 1
                self.stats.check_extensions += 1

            score = self._search_child(
                index=index,
                next_depth=next_depth,
                alpha=alpha,
                beta=beta,
                ply=ply,
                move=move,
                in_check=in_check,
                gives_check=gives_check,
                is_tactical=is_tactical,
                extensions_left=child_extensions,
            )

            self._pop_move_with_hash(saved_hash)

            if score > best_score:
                best_score = score
                best_move = move
                if ply == 0:
                    if self.root_best_move is None or self.root_best_move != move:
                        self.stats.root_move_changes += 1
                    self.root_best_move = move

            if self.search_cfg.use_alpha_beta:
                alpha = max(alpha, score)
                if alpha >= beta:
                    self.stats.beta_cutoffs += 1
                    if index == 0:
                        self.stats.first_move_cuts += 1
                    if self.move_sorter is not None:
                        self.move_sorter.on_beta_cutoff(
                            move=move,
                            ply=ply,
                            depth=depth,
                            previous_move=previous_move,
                            is_tactical=is_tactical,
                        )
                    break

        if best_move is None:
            return static_eval

        if self.tt is not None and key is not None:
            bound = self._determine_bound(
                best_score=best_score,
                original_alpha=original_alpha,
                beta=beta,
            )
            self.tt.store(
                key=key,
                depth=depth,
                score=best_score,
                best_move=best_move,
                bound=bound,
            )

        return best_score

    def _search_child(
        self,
        index: int,
        next_depth: int,
        alpha: float,
        beta: float,
        ply: int,
        move: chess.Move,
        in_check: bool,
        gives_check: bool,
        is_tactical: bool,
        extensions_left: int,
    ) -> float:
        """Dispatch to plain negamax, PVS null-window, or LMR reduced search."""
        if not self.search_cfg.use_alpha_beta:
            return -self._negamax(
                depth=next_depth,
                alpha=self.NEG_INF,
                beta=self.POS_INF,
                ply=ply + 1,
                previous_move=move,
                extensions_left=extensions_left,
            )

        if self.search_cfg.use_pvs and index > 0:
            score = alpha + 1
            if self._can_apply_lmr(
                index,
                next_depth,
                in_check,
                gives_check,
                is_tactical,
            ):
                reduction = self._lmr_reduction(next_depth, index)
                reduced_depth = max(0, next_depth - reduction)
                score = -self._negamax(
                    depth=reduced_depth,
                    alpha=-alpha - 1,
                    beta=-alpha,
                    ply=ply + 1,
                    previous_move=move,
                    extensions_left=extensions_left,
                )
                if score > alpha:
                    self.stats.lmr_researches += 1

            if score > alpha:
                score = -self._negamax(
                    depth=next_depth,
                    alpha=-alpha - 1,
                    beta=-alpha,
                    ply=ply + 1,
                    previous_move=move,
                    extensions_left=extensions_left,
                )
                if alpha < score < beta:
                    self.stats.pvs_researches += 1
                    score = -self._negamax(
                        depth=next_depth,
                        alpha=-beta,
                        beta=-alpha,
                        ply=ply + 1,
                        previous_move=move,
                        extensions_left=extensions_left,
                    )
            return score

        return -self._negamax(
            depth=next_depth,
            alpha=-beta,
            beta=-alpha,
            ply=ply + 1,
            previous_move=move,
            extensions_left=extensions_left,
        )

    def _quiescence(  # noqa: C901
        self,
        alpha: float,
        beta: float,
        ply: int,
        qs_depth: int,
    ) -> float:
        """Search captures and promotions until a quiet position is reached."""
        self.stats.qsearch_nodes += 1
        self.stats.seldepth = max(self.stats.seldepth, ply)

        if qs_depth >= self.search_cfg.qs_max_depth:
            return self._relative_eval()

        game_state = self.board.is_game_over()
        if game_state != chess.GameState.ONGOING:
            return self._terminal_score(game_state, ply)

        stand_pat = self._relative_eval()
        if self.search_cfg.use_alpha_beta:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            alpha = max(alpha, stand_pat)

        tactical_moves = [
            move
            for move in self.board.generate_legal_moves()
            if self._is_tactical_move(move)
        ]
        if not tactical_moves:
            return alpha

        if self.move_sorter is not None:
            tactical_moves = self.move_sorter.sort_tactical(self.board, tactical_moves)

        for move in tactical_moves:
            if (
                self.search_cfg.use_delta_pruning
                and self.search_cfg.use_alpha_beta
                and (
                    stand_pat + self._capture_gain(move) + self.search_cfg.delta_margin
                    < alpha
                )
            ):
                self.stats.qs_delta_pruning += 1
                continue

            if (
                self.search_cfg.use_see_pruning_in_qs
                and self.move_sorter is not None
                and self.board.is_capture(move)
                and self.move_sorter.see(self.board, move) < 0
            ):
                self.stats.qs_see_pruning += 1
                continue

            saved_hash = self._push_move_with_hash(move)
            score = -self._quiescence(-beta, -alpha, ply + 1, qs_depth + 1)
            self._pop_move_with_hash(saved_hash)

            if self.search_cfg.use_alpha_beta and score >= beta:
                return beta
            alpha = max(alpha, score)

        return alpha

    def _null_move_search(
        self,
        depth: int,
        beta: float,
        ply: int,
        extensions_left: int,
    ) -> float:
        """Perform a null-move search with O(1) incremental Zobrist hash update."""
        saved_hash = (
            self.zobrist.get_current_hash() if self.zobrist is not None else None
        )

        # O(1) incremental null-move hash (toggle side + remove EP)
        null_hash: int | None = None
        if self.zobrist is not None and saved_hash is not None:
            null_hash = self.zobrist.make_null_move_hash(self.board)

        self.board.push_null()

        if self.zobrist is not None:
            if null_hash is not None:
                self.zobrist.set_current_hash(null_hash)
            else:
                self.zobrist.hash_board(self.board)

        reduction = max(1, self.search_cfg.nmp_reduction_r)
        score = -self._negamax(
            depth=max(0, depth - 1 - reduction),
            alpha=-beta,
            beta=-beta + 1,
            ply=ply + 1,
            previous_move=None,
            extensions_left=extensions_left,
        )

        self.board.pop()
        if self.zobrist is not None and saved_hash is not None:
            self.zobrist.set_current_hash(saved_hash)

        return score

    def _check_time_limit(self) -> bool:
        max_time = self.search_cfg.max_time
        if max_time is None or self.start_time is None:
            return False
        if time.time() - self.start_time >= max_time:
            self.time_up = True
            return True
        return False

    def _relative_eval(self) -> float:
        white_perspective = float(self.evaluator.go(self.board))
        return white_perspective if bool(self.board.turn) else -white_perspective

    def _terminal_score(self, game_state: chess.GameState, ply: int) -> float:
        if game_state == chess.GameState.CHECKMATE:
            return -self.MATE_SCORE + ply
        return 0.0

    def _current_hash(self) -> int | None:
        if self.zobrist is None:
            return None
        return self.zobrist.get_current_hash()

    def _push_move_with_hash(self, move: chess.Move) -> int | None:
        """Push *move* onto the board and update the Zobrist hash.

        Returns the saved hash before the move (used by _pop_move_with_hash).
        """
        saved_hash = self._current_hash()
        next_hash: int | None = None
        if self.zobrist is not None and saved_hash is not None:
            next_hash = self.zobrist.make_move_hash(self.board, move)

        self.board.push(move)

        if self.zobrist is not None:
            if next_hash is not None:
                self.zobrist.set_current_hash(next_hash)
            else:
                self.zobrist.hash_board(self.board)

        return saved_hash

    def _pop_move_with_hash(self, saved_hash: int | None) -> None:
        self.board.pop()
        if self.zobrist is not None and saved_hash is not None:
            self.zobrist.set_current_hash(saved_hash)

    def _capture_gain(self, move: chess.Move) -> int:
        """Return the centipawn value of the piece captured by *move* (0 if none)."""
        piece = self.board.piece_at(move.to_square)
        if piece is None and self.board.is_en_passant(move):
            return int(MoveSorter.PIECE_VALUES_CP[chess.PAWN])
        if piece is None:
            return 0
        return int(MoveSorter.PIECE_VALUES_CP.get(piece.piece_type, 0))

    def _has_non_pawn_material(self) -> bool:
        color = chess.WHITE if bool(self.board.turn) else chess.BLACK
        return any(
            len(self.board.pieces(piece_type, color)) > 0
            for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        )

    def _is_tactical_move(self, move: chess.Move) -> bool:
        return bool(self.board.is_capture(move) or int(move.promotion) != 0)

    def _can_apply_futility(
        self,
        depth: int,
        static_eval: float,
        alpha: float,
        in_check: bool,
        is_tactical: bool,
    ) -> bool:
        """Return True if futility or extended-futility pruning skips this node."""
        if not self.search_cfg.use_alpha_beta:
            return False
        if in_check or is_tactical:
            return False

        if (
            self.search_cfg.use_futility_pruning
            and depth == 1
            and static_eval + self.search_cfg.futility_margin_standard <= alpha
        ):
            return True

        return (
            self.search_cfg.use_extended_futility_pruning
            and depth == 2
            and static_eval + self.search_cfg.futility_margin_extended <= alpha
        )

    def _can_apply_lmr(
        self,
        move_index: int,
        depth: int,
        in_check: bool,
        gives_check: bool,
        is_tactical: bool,
    ) -> bool:
        """Return True if Late Move Reduction is safe to apply to this move."""
        if not self.search_cfg.use_lmr:
            return False
        if in_check or gives_check or is_tactical:
            return False
        if depth < self.search_cfg.lmr_min_depth:
            return False
        return bool(move_index >= self.search_cfg.lmr_min_move_number)

    @staticmethod
    def _lmr_reduction(depth: int, move_index: int) -> int:
        """Compute LMR depth reduction.

        Formula: 0.75 * ln(depth) * ln(move_idx+1), capped to [1, 3].
        """
        base = 0.75 * math.log(max(2, depth)) * math.log(max(2, move_index + 1))
        return max(1, min(3, int(base)))

    def _determine_bound(
        self,
        best_score: float,
        original_alpha: float,
        beta: float,
    ) -> BoundType:
        """Classify the TT entry bound type based on score vs alpha/beta."""
        if not self.search_cfg.use_alpha_beta:
            return "exact"
        if best_score <= original_alpha:
            return "upper"
        if best_score >= beta:
            return "lower"
        return "exact"

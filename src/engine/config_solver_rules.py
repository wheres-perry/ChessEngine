"""Z3 symbolic rules for chess engine configuration validation."""

from typing import Any

from z3 import And, Bool, Implies, Int  # type: ignore


class ConfigSolverRules:
    """Symbolic constraint rules for engine configuration dependencies.

    All variables are z3 symbolic references.  Rules are ``(description,
    constraint)`` pairs -- every constraint must evaluate to ``True`` for a
    configuration to be valid.
    """

    def __init__(self) -> None:
        """Initialize the symbolic constraint rules for engine configuration.

        Creates z3 symbolic variables for evaluation settings, search boolean flags,
        and search integer parameters, then builds the corresponding validation rules.
        """
        # ==================================================================
        # Evaluation variables
        # ==================================================================
        self.eval_vars: dict[str, Any] = {
            "use_pst": Bool("use_pst"),
            "use_pawn_structure": Bool("use_pawn_structure"),
            "use_mobility": Bool("use_mobility"),
            "use_king_safety": Bool("use_king_safety"),
            "game_stage_conscious": Bool("game_stage_conscious"),
        }

        e = self.eval_vars

        self.eval_rules: list[tuple[str, object]] = [
            (
                "Pawn structure evaluation requires Piece-Square Tables (PST).",
                Implies(e["use_pawn_structure"], e["use_pst"]),
            ),
        ]

        # ==================================================================
        # Search boolean variables
        # ==================================================================
        self.search_bool_vars: dict[str, Any] = {
            "use_move_ordering": Bool("use_move_ordering"),
            "use_mvv_lva": Bool("use_mvv_lva"),
            "use_history_heuristic": Bool("use_history_heuristic"),
            "use_countermove_heuristic": Bool("use_countermove_heuristic"),
            "use_see_ordering": Bool("use_see_ordering"),
            "use_killer_moves": Bool("use_killer_moves"),
            "use_hash_move_ordering": Bool("use_hash_move_ordering"),
            "use_alpha_beta": Bool("use_alpha_beta"),
            "use_pvs": Bool("use_pvs"),
            "use_quiescence_search": Bool("use_quiescence_search"),
            "use_iid": Bool("use_iid"),
            "use_null_move_pruning": Bool("use_null_move_pruning"),
            "use_lmr": Bool("use_lmr"),
            "use_futility_pruning": Bool("use_futility_pruning"),
            "use_extended_futility_pruning": Bool("use_extended_futility_pruning"),
            "use_reverse_futility_pruning": Bool("use_reverse_futility_pruning"),
            "use_delta_pruning": Bool("use_delta_pruning"),
            "use_see_pruning_in_qs": Bool("use_see_pruning_in_qs"),
            "use_aspiration_windows": Bool("use_aspiration_windows"),
            "use_check_extensions": Bool("use_check_extensions"),
            "use_transposition_table": Bool("use_transposition_table"),
            "use_tt_aging": Bool("use_tt_aging"),
        }

        # ==================================================================
        # Search integer variables
        # ==================================================================
        self.search_int_vars: dict[str, Any] = {
            "history_max_score": Int("history_max_score"),
            "killer_slots_per_ply": Int("killer_slots_per_ply"),
            "qs_max_depth": Int("qs_max_depth"),
            "iid_min_depth": Int("iid_min_depth"),
            "iid_depth_reduction": Int("iid_depth_reduction"),
            "nmp_reduction_r": Int("nmp_reduction_r"),
            "nmp_min_depth": Int("nmp_min_depth"),
            "lmr_min_depth": Int("lmr_min_depth"),
            "lmr_min_move_number": Int("lmr_min_move_number"),
            "futility_margin_standard": Int("futility_margin_standard"),
            "futility_margin_extended": Int("futility_margin_extended"),
            "rfp_margin_multiplier": Int("rfp_margin_multiplier"),
            "rfp_max_depth": Int("rfp_max_depth"),
            "delta_margin": Int("delta_margin"),
            "aspiration_window_margin": Int("aspiration_window_margin"),
            "max_check_extensions": Int("max_check_extensions"),
            "tt_size_mb": Int("tt_size_mb"),
        }

        b = self.search_bool_vars
        i = self.search_int_vars

        # ==================================================================
        # Search rules
        # ==================================================================
        self.search_rules: list[tuple[str, object]] = [
            # ---------------------------------------------------------------
            # Binary flag dependencies
            # ---------------------------------------------------------------
            # Move ordering sub-features → parent flag
            (
                "MVV-LVA ordering requires move ordering.",
                Implies(b["use_mvv_lva"], b["use_move_ordering"]),
            ),
            (
                "SEE ordering requires move ordering.",
                Implies(b["use_see_ordering"], b["use_move_ordering"]),
            ),
            (
                "Killer moves require move ordering.",
                Implies(b["use_killer_moves"], b["use_move_ordering"]),
            ),
            (
                "History heuristic requires move ordering.",
                Implies(b["use_history_heuristic"], b["use_move_ordering"]),
            ),
            (
                "Countermove heuristic requires move ordering.",
                Implies(b["use_countermove_heuristic"], b["use_move_ordering"]),
            ),
            (
                "Countermove heuristic requires history heuristic.",
                Implies(b["use_countermove_heuristic"], b["use_history_heuristic"]),
            ),
            # Alpha-beta sub-features → parent flag
            (
                "Aspiration windows require alpha-beta pruning.",
                Implies(b["use_aspiration_windows"], b["use_alpha_beta"]),
            ),
            (
                "Null move pruning requires alpha-beta pruning.",
                Implies(b["use_null_move_pruning"], b["use_alpha_beta"]),
            ),
            (
                "Check extensions require alpha-beta pruning.",
                Implies(b["use_check_extensions"], b["use_alpha_beta"]),
            ),
            # Dual-parent features
            (
                "LMR requires both alpha-beta pruning and move ordering.",
                Implies(
                    b["use_lmr"],
                    And(b["use_alpha_beta"], b["use_move_ordering"]),
                ),
            ),
            (
                "PVS requires both alpha-beta pruning and move ordering.",
                Implies(
                    b["use_pvs"],
                    And(b["use_alpha_beta"], b["use_move_ordering"]),
                ),
            ),
            # Hash move ordering
            (
                "Hash move ordering requires move ordering and transposition table.",
                Implies(
                    b["use_hash_move_ordering"],
                    And(b["use_move_ordering"], b["use_transposition_table"]),
                ),
            ),
            # IID
            (
                "IID requires hash move ordering.",
                Implies(b["use_iid"], b["use_hash_move_ordering"]),
            ),
            # TT aging
            (
                "TT aging requires the transposition table.",
                Implies(b["use_tt_aging"], b["use_transposition_table"]),
            ),
            # Futility pruning (all three variants, not mutually exclusive)
            (
                "Futility pruning requires alpha-beta pruning.",
                Implies(b["use_futility_pruning"], b["use_alpha_beta"]),
            ),
            (
                "Extended futility pruning requires alpha-beta pruning.",
                Implies(b["use_extended_futility_pruning"], b["use_alpha_beta"]),
            ),
            (
                "Reverse futility pruning requires alpha-beta pruning.",
                Implies(b["use_reverse_futility_pruning"], b["use_alpha_beta"]),
            ),
            # Quiescence search
            (
                "Quiescence search requires alpha-beta pruning.",
                Implies(b["use_quiescence_search"], b["use_alpha_beta"]),
            ),
            (
                "Delta pruning requires quiescence search.",
                Implies(b["use_delta_pruning"], b["use_quiescence_search"]),
            ),
            (
                "SEE pruning in QS requires quiescence search.",
                Implies(b["use_see_pruning_in_qs"], b["use_quiescence_search"]),
            ),
            # ---------------------------------------------------------------
            # Scalar constraints
            # ---------------------------------------------------------------
            (
                "LMR minimum depth must be >= 1.",
                Implies(b["use_lmr"], i["lmr_min_depth"] >= 1),
            ),
            (
                "LMR minimum move number must be >= 1.",
                Implies(b["use_lmr"], i["lmr_min_move_number"] >= 1),
            ),
            (
                "IID minimum depth must exceed its depth reduction.",
                Implies(b["use_iid"], i["iid_min_depth"] > i["iid_depth_reduction"]),
            ),
            (
                "NMP reduction constant (R) must be >= 1.",
                Implies(b["use_null_move_pruning"], i["nmp_reduction_r"] >= 1),
            ),
            (
                "NMP minimum depth must be >= 1.",
                Implies(b["use_null_move_pruning"], i["nmp_min_depth"] >= 1),
            ),
            (
                "History max score must be positive.",
                Implies(b["use_history_heuristic"], i["history_max_score"] > 0),
            ),
            (
                "Must allocate at least 1 killer slot per ply.",
                Implies(b["use_killer_moves"], i["killer_slots_per_ply"] >= 1),
            ),
            (
                "Quiescence search max depth must be >= 1.",
                Implies(b["use_quiescence_search"], i["qs_max_depth"] >= 1),
            ),
            (
                "Futility margin (standard) must be positive.",
                Implies(b["use_futility_pruning"], i["futility_margin_standard"] > 0),
            ),
            (
                "Futility margin (extended) must be positive.",
                Implies(
                    b["use_extended_futility_pruning"],
                    i["futility_margin_extended"] > 0,
                ),
            ),
            (
                "Reverse futility pruning margin multiplier must be positive.",
                Implies(
                    b["use_reverse_futility_pruning"],
                    i["rfp_margin_multiplier"] > 0,
                ),
            ),
            (
                "Reverse futility pruning max depth must be >= 1.",
                Implies(b["use_reverse_futility_pruning"], i["rfp_max_depth"] >= 1),
            ),
            (
                "Delta margin must be positive.",
                Implies(b["use_delta_pruning"], i["delta_margin"] > 0),
            ),
            (
                "Aspiration window margin must be positive.",
                Implies(
                    b["use_aspiration_windows"],
                    i["aspiration_window_margin"] > 0,
                ),
            ),
            (
                "Max check extensions must be >= 1.",
                Implies(b["use_check_extensions"], i["max_check_extensions"] >= 1),
            ),
            (
                "Transposition table size must be positive.",
                Implies(b["use_transposition_table"], i["tt_size_mb"] > 0),
            ),
        ]

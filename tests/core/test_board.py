"""Test general board functionality like copying and state preservation."""

from engine._core import moray_core as core  # type: ignore


def test_copy_independence():
    """Verify that board copies are independent of the original."""
    board = core.Board.from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )

    board_copy = board.copy()
    e2e4_move = core.Move(12, 28, 0)  # e2 to e4
    board_copy.make_move(e2e4_move)

    assert (
        board.to_fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ), "Original board changed when copy was modified"

    assert (
        board_copy.to_fen()
        == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    ), "Copy not updated correctly"


def test_copy_complete_state():
    """Verify that board copies have complete state information preserved."""
    original = core.Board.from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    )

    copy = original.copy()

    assert original.to_fen() == copy.to_fen(), "FEN representation should match"
    assert original.get_side_to_move() == copy.get_side_to_move(), (
        "Side to move should match"
    )
    assert original.get_castling_rights() == copy.get_castling_rights(), (
        "Castling rights should match"
    )
    assert original.get_en_passant_square() == copy.get_en_passant_square(), (
        "En passant square should match"
    )
    assert original.get_halfmove_clock() == copy.get_halfmove_clock(), (
        "Halfmove clock should match"
    )
    assert original.get_fullmove_number() == copy.get_fullmove_number(), (
        "Fullmove number should match"
    )

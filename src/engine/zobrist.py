import random
import chess

class Zobrist:
    def __init__(self):
        self.keys = {}
        # Piece keys: 12 pieces (6 types x 2 colors) x 64 squares
        for piece in range(1, 7):  # PAWN to KING
            for color in [chess.WHITE, chess.BLACK]:
                for square in range(64):
                    self.keys[(piece, color, square)] = random.randint(0, 2**64 - 1)
        # Castling rights
        self.keys['castling_wk'] = random.randint(0, 2**64 - 1)
        self.keys['castling_wq'] = random.randint(0, 2**64 - 1)
        self.keys['castling_bk'] = random.randint(0, 2**64 - 1)
        self.keys['castling_bq'] = random.randint(0, 2**64 - 1)
        # En passant file
        for file in range(8):
            self.keys[('ep', file)] = random.randint(0, 2**64 - 1)
        # Turn
        self.keys['turn'] = random.randint(0, 2**64 - 1)

    def hash_board(self, board):
        hash_val = 0
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                hash_val ^= self.keys[(piece.piece_type, piece.color, square)]
        # Castling
        if board.has_kingside_castling_rights(chess.WHITE):
            hash_val ^= self.keys['castling_wk']
        if board.has_queenside_castling_rights(chess.WHITE):
            hash_val ^= self.keys['castling_wq']
        if board.has_kingside_castling_rights(chess.BLACK):
            hash_val ^= self.keys['castling_bk']
        if board.has_queenside_castling_rights(chess.BLACK):
            hash_val ^= self.keys['castling_bq']
        # En passant
        if board.ep_square:
            file = chess.square_file(board.ep_square)
            hash_val ^= self.keys[('ep', file)]
        # Turn
        if board.turn == chess.BLACK:
            hash_val ^= self.keys['turn']
        return hash_val

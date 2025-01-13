from __future__ import annotations

import numpy as np
import torch
import chess

from src.games.Game import Game
from src.games.chess.ChessBoard import ChessBoard, ChessMove
from src.util.ZobristHasher import ZobristHasher

ENCODING_CHANNELS = 13  # 12 for pieces + 1 for castling rights
BOARD_LENGTH = 8
BOARD_SIZE = BOARD_LENGTH * BOARD_LENGTH
ACTION_SIZE = 468  # 64 squares * 73 possible moves per square (approximate)
# TODO action size


class ChessGame(Game[ChessMove]):
    Hasher = ZobristHasher(ENCODING_CHANNELS, BOARD_LENGTH, BOARD_LENGTH)

    @property
    def null_move(self) -> ChessMove:
        return chess.Move.null()

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        return (ENCODING_CHANNELS, BOARD_LENGTH, BOARD_LENGTH)

    @property
    def average_num_moves_per_game(self) -> int:
        return 40

    def get_canonical_board(self, board: ChessBoard) -> np.ndarray:
        # TODO canonical board
        tensor = np.zeros(self.representation_shape, dtype=np.float32)
        piece_map = board.board.piece_map()
        for square, piece in piece_map.items():
            piece_index = self._piece_to_index(piece)
            row, col = divmod(square, BOARD_LENGTH)
            tensor[piece_index, BOARD_LENGTH - 1 - row, col] = 1
        tensor[12, :, :] = 1 if board.current_player == 1 else 0
        return tensor

    def _piece_to_index(self, piece: chess.Piece) -> int:
        mapping = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
        index = mapping[piece.piece_type]
        return index if piece.color == chess.WHITE else index + 6

    def hash_boards(self, boards: torch.Tensor) -> list[int]:
        # Implement an efficient hashing mechanism
        return self.Hasher.zobrist_hash_boards(boards)

    def encode_move(self, move: ChessMove) -> int:
        # TODO encode move
        # Simple encoding: from_square * 64 + to_square
        return move.from_square * BOARD_SIZE + move.to_square

    def decode_move(self, move: int) -> ChessMove:
        # TODO decode move
        from_square = move // BOARD_SIZE
        to_square = move % BOARD_SIZE
        return chess.Move(from_square, to_square)

    def symmetric_variations(
        self, board: np.ndarray, action_probabilities: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        # TODO only vertical flips
        variations = []
        for k in range(4):
            rotated_board = np.rot90(board, k=k, axes=(1, 2))
            rotated_probs = np.rot90(action_probabilities.reshape(8, 8, ACTION_SIZE // 64), k=k).flatten()
            variations.append((rotated_board, rotated_probs))
        return variations

    def get_initial_board(self) -> ChessBoard:
        return ChessBoard()

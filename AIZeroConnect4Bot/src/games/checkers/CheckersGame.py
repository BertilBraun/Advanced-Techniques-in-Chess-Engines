from __future__ import annotations

import torch
import numpy as np
from typing import List

from src.games.Game import Game
from src.games.checkers.CheckersBoard import BOARD_SIZE, BOARD_SQUARES, CheckersBoard, CheckersMove
from src.util.ZobristHasher import ZobristHasher

ROW_COUNT = BOARD_SIZE
COLUMN_COUNT = BOARD_SIZE
ENCODING_CHANNELS = 4

_black_squares = [
    r * BOARD_SIZE + c
    for r in range(BOARD_SIZE)
    for c in range(BOARD_SIZE)
    # Assume black square if (r + c) % 2 == 1 (typical checkers pattern)
    if (r + c) % 2 == 1
]
_black_squares_np = np.array(_black_squares, dtype=np.uint64)

BLACK_SQUARE_COUNT = len(_black_squares)
assert BLACK_SQUARE_COUNT == 32

# Create inverse mapping
_square_to_black = [-1] * BOARD_SQUARES
for i, sq in enumerate(_black_squares):
    _square_to_black[sq] = i


class CheckersGame(Game[CheckersMove]):
    Hasher = ZobristHasher(ENCODING_CHANNELS, ROW_COUNT, COLUMN_COUNT)

    @property
    def null_move(self) -> CheckersMove:
        return -1, -1

    @property
    def action_size(self) -> int:
        # Only from black squares to black squares possible -> (BOARD_SQUARES // 2) * (BOARD_SQUARES // 2) = 32 * 32 = 1024 instead of 64 * 64 = 4096
        return BLACK_SQUARE_COUNT * BLACK_SQUARE_COUNT

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        # Since only the black squares will ever be occupied, the representation also colapses that input down to only half of the columns of the actual board
        return ENCODING_CHANNELS, ROW_COUNT, COLUMN_COUNT // 2

    @property
    def average_num_moves_per_game(self) -> int:
        return 20

    def get_canonical_board(self, board: CheckersBoard) -> np.ndarray:
        # turn the 4 bitboards into a single 4x4x8 tensor

        def encode_bitboards_to_tensor(bitboards: list[int]) -> np.ndarray:
            """
            Encode multiple 64-bit bitboards into a binary tensor focusing only on black squares.

            Parameters:
            - bitboards: np.ndarray of shape (C,), dtype=np.uint64
                        C is the number of channels (e.g., 4 for black kings, black pieces, etc.)

            Returns:
            - tensor: np.ndarray of shape (C, 32), dtype=np.int8
                    Each row corresponds to a channel, and each column corresponds to a black square.
            """
            # Ensure bitboards is a NumPy array with dtype uint64
            bitboards_np = np.array(bitboards, dtype=np.uint64)  # Shape: (C,)

            # Expand dimensions to broadcast during bit shifting
            # Shape after expansion: (C, 1)
            bitboards_expanded = bitboards_np[:, np.newaxis]  # Shape: (C, 1)

            # Perform right bit-shift and bitwise AND to extract bits at _black_squares positions
            # _black_squares has shape (32,), so the result will have shape (C, 32)
            bits = ((bitboards_expanded >> _black_squares_np) & 1).astype(np.int8)  # Shape: (C, 32)

            return bits.reshape(self.representation_shape)  # Each row is a channel, each column is a black square

        if board.current_player == 1:
            return encode_bitboards_to_tensor(
                [board.black_kings, board.black_pieces, board.white_kings, board.white_pieces]
            )
        else:
            return np.flip(
                encode_bitboards_to_tensor(
                    [board.white_kings, board.white_pieces, board.black_kings, board.black_pieces]
                ),
                axis=1,
            )

    def hash_boards(self, boards: torch.Tensor) -> List[int]:
        assert boards.shape[1:] == self.representation_shape, f'Invalid shape: {boards.shape}'
        return self.Hasher.zobrist_hash_boards(boards)

    def encode_move(self, move: CheckersMove) -> int:
        move_from, move_to = move

        assert 0 <= move_from < BOARD_SQUARES, f'Invalid move: {move}'
        assert 0 <= move_to < BOARD_SQUARES, f'Invalid move: {move}'

        black_from = _square_to_black[move_from]
        black_to = _square_to_black[move_to]

        assert black_from != -1, f'Invalid move: {move}'
        assert black_to != -1, f'Invalid move: {move}'

        return black_from * BLACK_SQUARE_COUNT + black_to

    def decode_move(self, move: int) -> CheckersMove:
        move_black_from = move // BLACK_SQUARE_COUNT
        move_black_to = move % BLACK_SQUARE_COUNT

        assert 0 <= move_black_from < BLACK_SQUARE_COUNT, f'Invalid move: {move}'
        assert 0 <= move_black_to < BLACK_SQUARE_COUNT, f'Invalid move: {move}'

        move_from = _black_squares[move_black_from]
        move_to = _black_squares[move_black_to]

        assert move_from != -1, f'Invalid move: {move}'
        assert move_to != -1, f'Invalid move: {move}'

        return move_from, move_to

    def symmetric_variations(
        self, board: np.ndarray, action_probabilities: np.ndarray
    ) -> List[tuple[np.ndarray, np.ndarray]]:
        return [
            # Original board
            (board, action_probabilities),
            # Vertical flip
            # 1234 -> becomes -> 4321
            # 5678               8765
            # NOTE only valid since the board is reduced to 4x4x8 which removes the white cells, whereby the vertical flip will become a valid symmetry
            # NOTE Problem: The moves are not flippable, as the move encoding is not symmetric and would result with moves from white to white squares
        ]

    def _flip_black_index_4wide(self, black_idx: int) -> int:
        row = black_idx // 4
        subcol = black_idx % 4
        flipped_subcol = 3 - subcol
        flipped_idx = row * 4 + flipped_subcol
        return flipped_idx

    def _flip_move_4wide(self, move_idx: int) -> int:
        black_from = move_idx // 32
        black_to = move_idx % 32

        flipped_from = self._flip_black_index_4wide(black_from)
        flipped_to = self._flip_black_index_4wide(black_to)

        return flipped_from * 32 + flipped_to

    def _flip_action_probs(self, action_probs: np.ndarray) -> np.ndarray:
        flipped = np.zeros_like(action_probs)
        for move_idx, prob in enumerate(action_probs):
            if prob == 0.0:
                continue
            flipped_idx = self._flip_move_4wide(move_idx)
            flipped[flipped_idx] = prob
        return flipped

    def get_initial_board(self) -> CheckersBoard:
        return CheckersBoard()

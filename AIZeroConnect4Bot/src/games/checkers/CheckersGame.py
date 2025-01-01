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

_black_squares = []
for r in range(BOARD_SIZE):
    for c in range(BOARD_SIZE):
        # Assume black square if (r + c) % 2 == 1 (typical checkers pattern)
        if (r + c) % 2 == 1:
            _black_squares.append(r * BOARD_SIZE + c)

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
        return ENCODING_CHANNELS, ROW_COUNT, COLUMN_COUNT

    @property
    def average_num_moves_per_game(self) -> int:
        return 30

    def get_canonical_board(self, board: CheckersBoard) -> np.ndarray:
        # turn the 4 bitboards into a single 4x8x8 tensor
        def bitfield_to_tensor(bitfield: np.uint64) -> np.ndarray:
            # turn 64 bit integer into a list of 8x 8bit integers, then use np.unpackbits to get a 8x8 tensor
            return np.unpackbits(
                np.frombuffer(bitfield.tobytes(), dtype=np.uint8),
            ).reshape(ROW_COUNT, COLUMN_COUNT)

        if board.current_player == 1:
            return np.stack(
                [
                    bitfield_to_tensor(board.black_kings),
                    bitfield_to_tensor(board.black_pieces),
                    bitfield_to_tensor(board.white_kings),
                    bitfield_to_tensor(board.white_pieces),
                ]
            )
        else:
            return np.stack(
                [
                    np.flip(bitfield_to_tensor(board.white_kings), axis=0),
                    np.flip(bitfield_to_tensor(board.white_pieces), axis=0),
                    np.flip(bitfield_to_tensor(board.black_kings), axis=0),
                    np.flip(bitfield_to_tensor(board.black_pieces), axis=0),
                ]
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
            (np.flip(board, axis=2), np.flip(action_probabilities)),
            # NOTE: The following implementations DO NOT WORK. They are incorrect. This would give wrong symmetries to train on.
            # Player flip
            # yield -board, action_probabilities, -result
            # Player flip and vertical flip
            # yield -board[:, ::-1], action_probabilities[::-1], -result
        ]

    def get_initial_board(self) -> CheckersBoard:
        return CheckersBoard()

from __future__ import annotations

import torch
import numpy as np
from typing import List

from src.games.Game import Game
from src.games.tictactoe.TicTacToeBoard import (
    ROW_COUNT,
    COLUMN_COUNT,
    BOARD_SIZE,
    TicTacToeBoard,
    TicTacToeMove,
)
from src.util.ZobristHasher import ZobristHasher

ENCODING_CHANNELS = 3
ACTION_SIZE = BOARD_SIZE


class TicTacToeGame(Game[TicTacToeMove]):
    Hasher = ZobristHasher(ENCODING_CHANNELS, ROW_COUNT, COLUMN_COUNT)

    @property
    def null_move(self) -> TicTacToeMove:
        return -1

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        return ENCODING_CHANNELS, ROW_COUNT, COLUMN_COUNT

    @property
    def average_num_moves_per_game(self) -> int:
        AVERAGE_NUM_MOVES_PER_GAME = 5
        return AVERAGE_NUM_MOVES_PER_GAME

    def get_canonical_board(self, board: TicTacToeBoard) -> np.ndarray:
        canonical_board = board.board * board.current_player
        return (
            np.stack(
                (
                    (canonical_board == 1),
                    (canonical_board == -1),
                    (canonical_board == 0),
                )
            )
            .astype(np.float32)
            .reshape(self.representation_shape)
        )

    def hash_boards(self, boards: torch.Tensor) -> List[int]:
        assert boards.shape[1:] == self.representation_shape, f'Invalid shape: {boards.shape}'
        return self.Hasher.zobrist_hash_boards(boards)

    def encode_move(self, move: TicTacToeMove) -> int:
        assert 0 <= move < BOARD_SIZE, f'Invalid move: {move}'
        return move

    def decode_move(self, move: int) -> TicTacToeMove:
        assert 0 <= move < BOARD_SIZE, f'Invalid move: {move}'
        return move

    def symmetric_variations(
        self, board: np.ndarray, action_probabilities: np.ndarray
    ) -> List[tuple[np.ndarray, np.ndarray]]:
        return [
            (  # 90*k degree rotation
                np.rot90(board, k=k, axes=(1, 2)),
                np.rot90(action_probabilities.reshape(ROW_COUNT, COLUMN_COUNT), k=k).flatten(),
            )
            for k in range(4)
        ]

    def get_initial_board(self) -> TicTacToeBoard:
        return TicTacToeBoard()

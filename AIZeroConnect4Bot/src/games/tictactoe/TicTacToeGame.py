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

ENCODING_CHANNELS = 3  # TODO also try out for 3 channels
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
    def network_properties(self) -> tuple[int, int]:
        NUM_RES_BLOCKS = 4
        NUM_HIDDEN = 64
        return NUM_RES_BLOCKS, NUM_HIDDEN

    @property
    def average_num_moves_per_game(self) -> int:
        AVERAGE_NUM_MOVES_PER_GAME = 5
        return AVERAGE_NUM_MOVES_PER_GAME

    def get_canonical_board(self, board: TicTacToeBoard) -> np.ndarray:
        # TODO 3 channels for player 1, player 2 and empty cells?
        canonical_board = board.board * board.current_player
        return np.stack(
            [
                (canonical_board == 1).astype(np.float32),
                (canonical_board == -1).astype(np.float32),
                (canonical_board == 0).astype(np.float32),
            ]
        ).reshape(self.representation_shape)
        # return (board.board * board.current_player).reshape(self.representation_shape)

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
            # Original board
            (board, action_probabilities),
            # Vertical flip
            # 1234 -> becomes -> 4321
            # 5678               8765
            # TODO check - should work (np.flip(board, axis=2), np.flip(action_probabilities)),
            # NOTE: The following implementations DO NOT WORK. They are incorrect. This would give wrong symmetries to train on.
            # Player flip
            # yield -board, action_probabilities, -result
            # Player flip and vertical flip
            # yield -board[:, ::-1], action_probabilities[::-1], -result
        ]

    def get_initial_board(self) -> TicTacToeBoard:
        return TicTacToeBoard()

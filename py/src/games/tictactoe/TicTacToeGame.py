from __future__ import annotations

import numpy as np

from src.games.Game import Game
from src.games.tictactoe.TicTacToeBoard import (
    ROW_COUNT,
    COLUMN_COUNT,
    BOARD_SIZE,
    TicTacToeBoard,
    TicTacToeMove,
)

ENCODING_CHANNELS = 3
ACTION_SIZE = BOARD_SIZE


class TicTacToeGame(Game[TicTacToeMove]):
    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        return ENCODING_CHANNELS, ROW_COUNT, COLUMN_COUNT

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

    def encode_move(self, move: TicTacToeMove) -> int:
        assert 0 <= move < BOARD_SIZE, f'Invalid move: {move}'
        return move

    def decode_move(self, move: int) -> TicTacToeMove:
        assert 0 <= move < BOARD_SIZE, f'Invalid move: {move}'
        return move

    def symmetric_variations(
        self, board: np.ndarray, action_probabilities: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        # Return all 90 degree rotations of the board and action probabilities
        # In addition a flip + all 90 degree rotations of the board and action probabilities
        return [
            (  # 90*k degree rotation
                np.rot90(board, k=k, axes=(1, 2)),
                np.rot90(action_probabilities.reshape(ROW_COUNT, COLUMN_COUNT), k=k).flatten(),
            )
            for k in range(4)
        ] + [
            (  # 90*k degree rotation + flip
                np.rot90(np.flip(board, axis=2), k=k, axes=(1, 2)),
                np.rot90(np.flip(action_probabilities.reshape(ROW_COUNT, COLUMN_COUNT), axis=1), k=k).flatten(),
            )
            for k in range(4)
        ]

    def get_initial_board(self) -> TicTacToeBoard:
        return TicTacToeBoard()

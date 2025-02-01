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
        self, board: np.ndarray, visit_counts: list[tuple[int, int]]
    ) -> list[tuple[np.ndarray, list[tuple[int, int]]]]:
        # Return all 90 degree rotations of the board and action probabilities
        # In addition a flip + all 90 degree rotations of the board and action probabilities

        return [
            (  # 90*k degree rotation
                np.rot90(board, k=k, axes=(1, 2)),
                self._flip_and_rotate_visit_counts(visit_counts, k, flip=False),
            )
            for k in range(4)
        ] + [
            (  # 90*k degree rotation + flip
                np.rot90(np.flip(board, axis=2), k=k, axes=(1, 2)),
                self._flip_and_rotate_visit_counts(visit_counts, k, flip=True),
            )
            for k in range(4)
        ]

    def _flip_and_rotate_visit_counts(
        self, visit_counts: list[tuple[int, int]], k: int, flip: bool
    ) -> list[tuple[int, int]]:
        updated_visit_counts: list[tuple[int, int]] = []
        for move, count in visit_counts:
            row, col = move // ROW_COUNT, move % COLUMN_COUNT
            if flip:
                col = COLUMN_COUNT - col - 1
            for _ in range(k):
                row, col = ROW_COUNT - col - 1, row
            updated_move = row * ROW_COUNT + col
            updated_visit_counts.append((updated_move, count))
        return updated_visit_counts

    def get_initial_board(self) -> TicTacToeBoard:
        return TicTacToeBoard()

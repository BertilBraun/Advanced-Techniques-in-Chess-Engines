from __future__ import annotations

import numpy as np

from src.games.Game import Game
from src.games.connect4.Connect4Board import Connect4Board, Connect4Move, ROW_COUNT, COLUMN_COUNT

ENCODING_CHANNELS = 3
ACTION_SIZE = COLUMN_COUNT


class Connect4Game(Game[Connect4Move]):
    @property
    def null_move(self) -> Connect4Move:
        return -1

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        return ENCODING_CHANNELS, ROW_COUNT, COLUMN_COUNT

    def get_canonical_board(self, board: Connect4Board) -> np.ndarray:
        canonical_board = board.board * board.current_player
        return (
            np.stack(
                (
                    canonical_board == 1,
                    canonical_board == -1,
                    canonical_board == 0,
                )
            )
            .astype(np.float32)
            .reshape(self.representation_shape)
        )

    def encode_move(self, move: Connect4Move) -> int:
        assert 0 <= move < COLUMN_COUNT, f'Invalid move: {move}'
        return move

    def decode_move(self, move: int) -> Connect4Move:
        assert 0 <= move < COLUMN_COUNT, f'Invalid move: {move}'
        return move

    def symmetric_variations(
        self, board: np.ndarray, visit_counts: list[tuple[int, int]]
    ) -> list[tuple[np.ndarray, list[tuple[int, int]]]]:
        return [
            # Original board
            (board, visit_counts),
            # Vertical flip
            # 1234 -> becomes -> 4321
            # 5678               8765
            (np.flip(board, axis=2), self._flip_action_probs(visit_counts)),
            # NOTE: The following implementations DO NOT WORK. They are incorrect. This would give wrong symmetries to train on.
            # Player flip
            # yield -board, action_probabilities, -result
            # Player flip and vertical flip
            # yield -board[:, ::-1], action_probabilities[::-1], -result
        ]

    def _flip_action_probs(self, visit_counts: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(COLUMN_COUNT - 1 - action, count) for action, count in visit_counts]

    def get_initial_board(self) -> Connect4Board:
        return Connect4Board()

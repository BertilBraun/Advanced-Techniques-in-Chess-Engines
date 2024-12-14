from __future__ import annotations

import numpy as np
from typing import List, Optional

import torch

from AIZeroConnect4Bot.src.games.Game import Board, Game, Player
from AIZeroConnect4Bot.src.games.connect4.hashing import zobrist_hash_boards
from AIZeroConnect4Bot.src.games.connect4.Connect4Defines import (
    ROW_COUNT,
    COLUMN_COUNT,
    ACTION_SIZE,
    ENCODING_CHANNELS,
    NUM_RES_BLOCKS,
    NUM_HIDDEN,
    AVERAGE_NUM_MOVES_PER_GAME,
)

Connect4Move = int


class Connect4Board(Board[Connect4Move]):
    def __init__(self) -> None:
        super().__init__()
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
        self._winner: Optional[Player] = None

    def make_move(self, column: Connect4Move) -> bool:
        if column < 0 or column >= COLUMN_COUNT or self.board[0][column] != 0:
            return False
        for row in range(ROW_COUNT - 1, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                if self.__check_winner(row, column):
                    self._winner = self.current_player
                self._switch_player()
                return True
        assert False, 'Unreachable code'

    def __check_winner(self, row: int, col: int) -> bool:
        piece = self.board[row][col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for delta_row, delta_col in directions:
            count = 1
            for dr, dc in [(delta_row, delta_col), (-delta_row, -delta_col)]:
                r, c = row + dr, col + dc
                while 0 <= r < ROW_COUNT and 0 <= c < COLUMN_COUNT and self.board[r][c] == piece:
                    count += 1
                    if count >= 4:
                        return True
                    r += dr
                    c += dc
        return False

    def check_winner(self) -> Optional[Player]:
        return self._winner

    def is_full(self) -> bool:
        return np.all(self.board[0] != 0)  # type: ignore

    def get_valid_moves(self) -> list[Connect4Move]:
        return [col for col in range(COLUMN_COUNT) if self.board[0][col] == 0]

    def get_board_state(self) -> np.ndarray:
        return self.board.copy()

    def copy(self) -> Connect4Board:
        game = Connect4Board()
        game.board = self.board.copy()
        game.current_player = self.current_player
        game._winner = self._winner
        return game

    def is_game_over(self) -> bool:
        return self._winner is not None or self.is_full()


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

    @property
    def network_properties(self) -> tuple[int, int]:
        return NUM_RES_BLOCKS, NUM_HIDDEN

    @property
    def average_num_moves_per_game(self) -> int:
        return AVERAGE_NUM_MOVES_PER_GAME

    def get_canonical_board(self, board: Connect4Board) -> np.ndarray:
        return (board.board * board.current_player).reshape(self.representation_shape)

    def hash_boards(self, boards: torch.Tensor) -> List[int]:
        assert boards.shape[1:] == self.representation_shape, f'Invalid shape: {boards.shape}'
        return zobrist_hash_boards(boards)

    def encode_move(self, move: Connect4Move) -> int:
        assert 0 <= move < COLUMN_COUNT, f'Invalid move: {move}'
        return move

    def decode_move(self, move: int) -> Connect4Move:
        assert 0 <= move < COLUMN_COUNT, f'Invalid move: {move}'
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
            (np.flip(board, axis=2), np.flip(action_probabilities)),
            # NOTE: The following implementations DO NOT WORK. They are incorrect. This would give wrong symmetries to train on.
            # Player flip
            # yield -board, action_probabilities, -result
            # Player flip and vertical flip
            # yield -board[:, ::-1], action_probabilities[::-1], -result
        ]

    def get_initial_board(self) -> Board[int]:
        return Connect4Board()


# TODO remove
board = Connect4Board()
game = Connect4Game()

if __name__ == '__main__':

    def play(game: Connect4Board) -> None:
        while True:
            column = int(input(f'Player {game.current_player}, choose a column (0-{COLUMN_COUNT-1}): '))
            if game.make_move(column):
                if game.check_winner():
                    print(f'Player {game.current_player} wins!')
                    break
                elif game.is_full():
                    print('The game is a draw!')
                    break
            else:
                print('Invalid move. Try again.')

            print(game.board)

    game = Connect4Board()
    play(game)

from __future__ import annotations

import numpy as np
from typing import Optional


from src.games.Game import Board
from src.games.Board import Player

ROW_COUNT = 7
COLUMN_COUNT = 8

Connect4Move = int


class Connect4Board(Board[Connect4Move]):
    def __init__(self) -> None:
        super().__init__()
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
        self._winner: Optional[Player] = None

    @property
    def board_dimensions(self) -> tuple[int, int]:
        return ROW_COUNT, COLUMN_COUNT

    def make_move(self, column: Connect4Move) -> None:
        assert self._winner is None, 'Game is already over'
        assert 0 <= column < COLUMN_COUNT and self.board[0][column] == 0, 'Invalid move'
        row = np.argmax(self.board[:, column] != 0).item() - 1
        if row == -1:  # If the column is empty
            row = ROW_COUNT - 1
        self.board[row][column] = self.current_player
        if self.__check_winner(row, column):
            self._winner = self.current_player
        self._switch_player()

    def __check_winner(self, row: int, col: int) -> bool:
        piece = self.board[row][col]
        for delta_row, delta_col in ((1, 0), (0, 1), (1, 1), (1, -1)):
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
        return np.all(self.board[0] != 0).item()

    def get_valid_moves(self) -> list[Connect4Move]:
        return np.where(self.board[0] == 0)[0].tolist()

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

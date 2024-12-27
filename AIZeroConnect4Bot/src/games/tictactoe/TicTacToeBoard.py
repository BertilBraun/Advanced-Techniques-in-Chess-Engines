from __future__ import annotations

import numpy as np
from typing import Optional


from src.games.Game import Board
from src.games.Board import Player

TicTacToeMove = int

ROW_COUNT = 3
COLUMN_COUNT = 3
BOARD_SIZE = ROW_COUNT * COLUMN_COUNT


class TicTacToeBoard(Board[TicTacToeMove]):
    def __init__(self) -> None:
        super().__init__()
        self.board = np.zeros((BOARD_SIZE), dtype=int)
        self._winner: Optional[Player] = None

    @property
    def board_dimensions(self) -> tuple[int, int]:
        return ROW_COUNT, COLUMN_COUNT

    def make_move(self, move: TicTacToeMove) -> bool:
        if move < 0 or move >= BOARD_SIZE or self.board[move] != 0:
            return False

        self.board[move] = self.current_player
        if self._check_winner():
            self._winner = self.current_player
        self._switch_player()
        return True

    def _check_winner(self) -> bool:
        # Only need to check the current player
        for i in range(ROW_COUNT):
            if np.all(self.board[i * ROW_COUNT : (i + 1) * ROW_COUNT] == self.current_player):
                return True
        for i in range(COLUMN_COUNT):
            if np.all(self.board[i::ROW_COUNT] == self.current_player):
                return True
        if np.all(self.board[:: ROW_COUNT + 1] == self.current_player):
            return True
        if np.all(self.board[ROW_COUNT - 1 : BOARD_SIZE - 1 : ROW_COUNT - 1] == self.current_player):
            return True
        return False

    def check_winner(self) -> Optional[Player]:
        return self._winner

    def is_full(self) -> bool:
        return np.all(self.board != 0)  # type: ignore

    def get_valid_moves(self) -> list[TicTacToeMove]:
        return [move for move in range(BOARD_SIZE) if self.board[move] == 0]

    def get_board_state(self) -> np.ndarray:
        return self.board.copy()

    def copy(self) -> TicTacToeBoard:
        game = TicTacToeBoard()
        game.board = self.board.copy()
        game.current_player = self.current_player
        game._winner = self._winner
        return game

    def is_game_over(self) -> bool:
        return self._winner is not None or self.is_full()

    # TODO remove
    def display(self):
        print(self.to_string(True))

    def to_string(self, large) -> str:
        s = ''
        mp = {0: ' ' if large else '.', 1: 'X', -1: 'O'}
        for i in range(ROW_COUNT):
            concat = '|' if large else ''
            s += concat.join([mp[self.board[i * ROW_COUNT + j]] for j in range(COLUMN_COUNT)]) + '\n'
            if i < ROW_COUNT - 1 and large:
                s += '-' * 5 + '\n'
        return s

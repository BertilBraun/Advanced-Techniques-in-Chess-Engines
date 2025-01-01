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
        if self._check_winner(move):
            self._winner = self.current_player
        self._switch_player()
        return True

    def _check_winner(self, move: TicTacToeMove) -> bool:
        r, c = divmod(move, ROW_COUNT)

        return (
            np.sum(self.board[r * ROW_COUNT : (r + 1) * ROW_COUNT]) == ROW_COUNT * self.current_player
            or np.sum(self.board[c::ROW_COUNT]) == COLUMN_COUNT * self.current_player
            or np.sum(self.board[:: ROW_COUNT + 1]) == ROW_COUNT * self.current_player
            or np.sum(self.board[ROW_COUNT - 1 : BOARD_SIZE - 1 : ROW_COUNT - 1]) == ROW_COUNT * self.current_player
        )

    def check_winner(self) -> Optional[Player]:
        return self._winner

    def is_full(self) -> bool:
        return np.all(self.board != 0).item()

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

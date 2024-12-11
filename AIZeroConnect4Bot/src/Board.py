from __future__ import annotations

import numpy as np
from typing import List, Literal, Optional

from AIZeroConnect4Bot.settings import COLUMN_COUNT, ROW_COUNT

_WINNER_NOT_YET_CHECKED = object()


class Connect4:
    def __init__(self) -> None:
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
        self.current_player: Color = 1
        self._winner: Color | None | object = _WINNER_NOT_YET_CHECKED

    def make_move(self, column: int) -> bool:
        if column < 0 or column >= COLUMN_COUNT or self.board[0][column] != 0:
            return False
        for row in range(ROW_COUNT - 1, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                self._winner = _WINNER_NOT_YET_CHECKED
                return True
        return False

    def check_winner(self) -> Optional[int]:
        if self._winner is not _WINNER_NOT_YET_CHECKED:
            return self._winner  # type: ignore
        for row in range(ROW_COUNT):
            for col in range(COLUMN_COUNT):
                if self.board[row][col] == 0:
                    continue
                if (
                    self.__check_direction(row, col, 1, 0)
                    or self.__check_direction(row, col, 0, 1)
                    or self.__check_direction(row, col, 1, 1)
                    or self.__check_direction(row, col, 1, -1)
                ):
                    self._winner = self.board[row][col]
                    return self.board[row][col]
        self._winner = None
        return None

    def __check_direction(self, row: int, col: int, delta_row: int, delta_col: int) -> bool:
        piece = self.board[row][col]
        for i in range(1, 4):
            r, c = row + delta_row * i, col + delta_col * i
            if r < 0 or r >= ROW_COUNT or c < 0 or c >= COLUMN_COUNT or self.board[r][c] != piece:
                return False
        return True

    def switch_player(self) -> None:
        self.current_player = -1 if self.current_player == 1 else 1

    def is_full(self) -> bool:
        return np.all(self.board[0] != 0)  # type: ignore

    def get_valid_moves(self) -> List[int]:
        return [col for col in range(COLUMN_COUNT) if self.board[0][col] == 0]

    def get_board_state(self) -> np.ndarray:
        return self.board.copy()

    def get_canonical_board(self) -> np.ndarray:
        return self.board * self.current_player

    def copy(self) -> Connect4:
        game = Connect4()
        game.board = self.board.copy()
        game.current_player = self.current_player
        return game

    def is_game_over(self) -> bool:
        return self.check_winner() is not None or self.is_full()


Board = Connect4
Move = int
null_move = -1
Color = Literal[-1, 1]

if __name__ == '__main__':

    def play(game: Connect4) -> None:
        while True:
            column = int(input(f'Player {game.current_player}, choose a column (0-{COLUMN_COUNT-1}): '))
            if game.make_move(column):
                winner = game.check_winner()
                if winner:
                    print(f'Player {winner} wins!')
                    break
                elif game.is_full():
                    print('The game is a draw!')
                    break
                game.switch_player()
            else:
                print('Invalid move. Try again.')

            print(game.board)

    game = Connect4()
    play(game)

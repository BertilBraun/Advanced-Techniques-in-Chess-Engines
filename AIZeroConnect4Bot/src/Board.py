from __future__ import annotations

import numpy as np
from typing import List, Literal, Optional

from AIZeroConnect4Bot.src.settings import COLUMN_COUNT, ROW_COUNT


class Connect4:
    def __init__(self) -> None:
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
        self.current_player: Color = 1
        self._winner: Optional[Color] = None

    def make_move(self, column: int) -> bool:
        if column < 0 or column >= COLUMN_COUNT or self.board[0][column] != 0:
            return False
        for row in range(ROW_COUNT - 1, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                if self.__check_winner(row, column):
                    self._winner = self.current_player
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

    def check_winner(self) -> Optional[int]:
        return self._winner

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
        game._winner = self._winner
        return game

    def is_game_over(self) -> bool:
        return self._winner is not None or self.is_full()


Board = Connect4
Move = int
null_move = -1
Color = Literal[-1, 1]

if __name__ == '__main__':

    def play(game: Connect4) -> None:
        while True:
            column = int(input(f'Player {game.current_player}, choose a column (0-{COLUMN_COUNT-1}): '))
            if game.make_move(column):
                if game.check_winner():
                    print(f'Player {game.current_player} wins!')
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

from __future__ import annotations

import numpy as np
from typing import Optional, List

from src.games.Game import Board
from src.games.Board import Player
from src.util.ZobristHasherNumpy import ZobristHasherNumpy

# Global board size
SIZE: int = 7

# Global Zobrist hasher using single-plane (signed values)
hasher = ZobristHasherNumpy(planes=1, rows=SIZE, cols=SIZE)

_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]


class HexBoard(Board[int]):
    """This class represents the Hex board and implements the game logic.

    The game is played on a hexagonal grid, and players take turns placing their pieces
    on the board. The goal is to connect opposite sides of the board with a continuous
    path of their pieces. Player 1 connects top to bottom, while Player -1 connects left to right.
    No two players can occupy the same cell, and the game ends when one player connects their sides.
    Therefore the game cannot end in a draw.
    """

    def __init__(self) -> None:
        super().__init__()
        self.board = np.zeros((SIZE, SIZE), dtype=np.int8)
        self._winner: Optional[Player] = None

    def make_move(self, move: int) -> None:
        assert self._winner is None, 'Game is already over'
        assert 0 <= move < SIZE * SIZE, 'Invalid move index'
        r, c = divmod(move, SIZE)
        assert self.board[r, c] == 0, 'Position already occupied'
        self.board[r, c] = self.current_player
        if self.__check_winner(self.current_player, r, c):
            self._winner = self.current_player
        self._switch_player()

    def __check_winner(self, player: Player, last_r: int, last_c: int) -> bool:
        """
        Return True iff placing `player`'s stone at (last_r, last_c) wins the game.
        Assumes self.board[last_r, last_c] is already set to `player`.
        """
        stack = [(last_r, last_c)]
        visited = {(last_r, last_c)}

        # Track whether this connected component touches the two goal edges
        touches_top = last_r == 0
        touches_bottom = last_r == SIZE - 1
        touches_left = last_c == 0
        touches_right = last_c == SIZE - 1

        while stack:
            r, c = stack.pop()
            for dr, dc in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < SIZE and 0 <= nc < SIZE and (nr, nc) not in visited and self.board[nr, nc] == player:
                    visited.add((nr, nc))
                    stack.append((nr, nc))

                    # update edge flags incrementally
                    if nr == 0:
                        touches_top = True
                    if nr == SIZE - 1:
                        touches_bottom = True
                    if nc == 0:
                        touches_left = True
                    if nc == SIZE - 1:
                        touches_right = True

                    # early exit once both goal edges are reached
                    if player == 1 and touches_top and touches_bottom:
                        return True
                    if player == -1 and touches_left and touches_right:
                        return True

        return False

    def check_winner(self) -> Optional[Player]:
        return self._winner

    def is_game_over(self) -> bool:
        return self._winner is not None or not self.get_valid_moves()

    def get_valid_moves(self) -> List[int]:
        empties = np.where(self.board.flatten() == 0)[0]
        return empties.tolist()

    def copy(self) -> HexBoard:
        new = HexBoard()
        new.board = self.board.copy()
        new.current_player = self.current_player
        new._winner = self._winner
        return new

    def quick_hash(self) -> int:
        plane = self.board[np.newaxis, :, :].astype(np.int64)
        return hasher.zobrist_hash_board(plane)

    def __repr__(self) -> str:
        symbols = {1: 'X', -1: 'O', 0: '.'}
        lines = []
        for i, row in enumerate(self.board):
            indent = ' ' * i
            line = ' '.join(symbols.get(cell, 'invalid') for cell in row)
            lines.append(indent + line)
        return '\n'.join(lines)


if __name__ == '__main__':
    import random

    for _ in range(50_000):
        b = HexBoard()
        moves = b.get_valid_moves()
        while not b.is_game_over():
            mv = random.choice(moves)
            b.make_move(mv)
            moves.remove(mv)
        # assert that exactly one winner is recorded
        assert b.check_winner() in (1, -1)

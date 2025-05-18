from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple

from src.games.Game import Board
from src.games.Board import Player
from src.util.ZobristHasherNumpy import ZobristHasherNumpy

# Global board size
SIZE: int = 11

# Global Zobrist hasher using single-plane (signed values)
hasher = ZobristHasherNumpy(planes=1, rows=SIZE, cols=SIZE)


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
        self.board = np.zeros((SIZE, SIZE), dtype=int)
        self._winner: Optional[Player] = None

    def make_move(self, move: int) -> None:
        assert self._winner is None, 'Game is already over'
        assert 0 <= move < SIZE * SIZE, 'Invalid move index'
        r, c = divmod(move, SIZE)
        assert self.board[r, c] == 0, 'Position already occupied'
        self.board[r, c] = self.current_player
        if self.__check_winner(self.current_player):
            self._winner = self.current_player
        self._switch_player()

    def __check_winner(self, player: Player) -> bool:
        # Check connectivity via DFS between opposite sides
        visited = set()
        stack: List[Tuple[int, int]] = []
        if player == 1:
            # Player 1 connects top (row 0) to bottom (row SIZE-1)
            for col in range(SIZE):
                if self.board[0, col] == player:
                    stack.append((0, col))
                    visited.add((0, col))
            target = ('row', SIZE - 1)
        else:
            # Player -1 connects left (col 0) to right (col SIZE-1)
            for row in range(SIZE):
                if self.board[row, 0] == player:
                    stack.append((row, 0))
                    visited.add((row, 0))
            target = ('col', SIZE - 1)
        # Neighbor offsets on hex grid
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
        while stack:
            r, c = stack.pop()
            if (target[0] == 'row' and r == target[1]) or (target[0] == 'col' and c == target[1]):
                return True
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < SIZE and 0 <= nc < SIZE and (nr, nc) not in visited and self.board[nr, nc] == player:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
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
            line = ' '.join(symbols[cell] for cell in row)
            lines.append(indent + line)
        return '\n'.join(lines)

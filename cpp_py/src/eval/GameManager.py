from __future__ import annotations

import chess
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.eval.Bot import Bot


class GameManager:
    def __init__(self, white: Bot, black: Bot) -> None:
        """Initializes the game manager with two players."""
        self.players = {1: white, -1: black}

    def play_game(self, verify_moves=True) -> Optional[Player]:
        """Manages the gameplay loop until the game is over or a player quits."""
        self.board = chess.Board()
        while not self.board.is_game_over():
            current_player = self.players[self.board.current_player]

            current_player.restart_clock()
            move = current_player.think(self.board)

            if verify_moves and move not in self.board.get_valid_moves():
                raise ValueError(f'Invalid move {move} for player {current_player.name}')

            self.board.make_move(move)
            print(repr(self.board))

        return self.board.check_winner()

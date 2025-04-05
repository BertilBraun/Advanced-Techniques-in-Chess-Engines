from __future__ import annotations

import chess
from typing import Optional, TYPE_CHECKING

from src.eval.Bot import check_winner

if TYPE_CHECKING:
    from src.eval.Bot import Bot


class GameManager:
    def __init__(self, white: Bot, black: Bot) -> None:
        """Initializes the game manager with two players."""
        self.players = {chess.WHITE: white, chess.BLACK: black}

    def play_game(self, verify_moves=True) -> Optional[chess.Color]:
        """Manages the gameplay loop until the game is over or a player quits."""
        self.board = chess.Board()
        while not self.board.is_game_over():
            current_player = self.players[self.board.turn]

            current_player.restart_clock()
            move = current_player.think(self.board)

            if verify_moves and move not in self.board.legal_moves:
                raise ValueError(f'Invalid move {move} for player {current_player.name}')

            self.board.push(move)
            print(repr(self.board))

        return check_winner(self.board)

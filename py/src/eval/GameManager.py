from __future__ import annotations

import random
import chess.pgn
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.eval.Bot import Bot
from src.games.Game import Player
from src.settings import CurrentGame


class GameManager:
    def __init__(self, white: Bot, black: Bot) -> None:
        """Initializes the game manager with two players."""
        self.players = {1: white, -1: black}

    def play_game(self, verify_moves: bool = True, start_random: bool = False) -> tuple[Optional[Player], str]:
        """Manages the gameplay loop until the game is over or a player quits."""
        self.board = CurrentGame.get_initial_board()

        move_list: list = []

        if start_random:
            for _ in range(2):
                move = random.choice(self.board.get_valid_moves())
                self.board.make_move(move)
                move_list.append(move)

        while not self.board.is_game_over():
            current_player = self.players[self.board.current_player]

            current_player.restart_clock()
            move = current_player.think(self.board)

            if verify_moves and move not in self.board.get_valid_moves():
                raise ValueError(f'Invalid move {move} for player {current_player.name} on board:\n{self.board}')

            self.board.make_move(move)
            print(repr(self.board))
            move_list.append(move)

        print(f'Game over! Winner: {self.board.check_winner()} Moves: {move_list}')

        pgn = chess.pgn.Game()
        node = pgn.add_variation(move_list[0])
        for move in move_list[1:]:
            node = node.add_variation(move)

        return self.board.check_winner(), str(pgn)

from __future__ import annotations

import os
import time

import chess
import chess.pgn
import chess.svg

from human import human_player
from mcts.mcts import mcts_player_with_stats
from mcts.nn import evaluate_position_with_NN, print_and_reset_stats


def play_game(player1, player2):
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            player1(board)
        else:
            player2(board)

        with open("game.svg", "w") as f:
            f.write(chess.svg.board(board, size=650))
        os.startfile("game.svg")

        time.sleep(0.1)

    print(chess.pgn.Game.from_board(board))


# play_game(mcts_player_with_stats(evaluate_position_static, itermax=2000), human_player)
play_game(
    mcts_player_with_stats(
        evaluate_position_with_NN,
        print_and_reset_stats,
        50,
        5
    ),
    human_player
)

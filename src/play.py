
import time
from IPython.display import clear_output, display
import chess
import chess.pgn

from ai_model import UCT

board = chess.Board()


def mcts_player(board: chess.Board):
    for move_choice in board.legal_moves:
        copy = board.copy()
        copy.push(move_choice)
        if copy.is_game_over():
            board.push(move_choice)
            return

    board.push(UCT(board, 100, 30))


def human_player(board: chess.Board):
    while True:
        move = input("Input Your Move:")
        if move == "q":
            raise KeyboardInterrupt
        try:
            board.push_san(move)
            break
        except Exception as e:
            print(e)


def play_game(player1, player2):
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            player1(board)
        else:
            player2(board)

        clear_output(wait=True)
        display(board)
        time.sleep(0.1)

    game = chess.pgn.Game.from_board(board)
    print(game)


if __name__ == "__main__":
    play_game(mcts_player, human_player)

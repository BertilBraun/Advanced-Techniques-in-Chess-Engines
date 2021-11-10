from typing import TextIO
import chess.pgn

from util import board_to_bitfields


def create_dataset(game: chess.pgn.Game, out: TextIO) -> None:
    state = game.end()
    while state:
        nums = board_to_bitfields(state.board())
        evaluation = state.eval()

        if evaluation is not None:
            evaluation = evaluation.relative.score(mate_score=10) / 100
            out.write(','.join(map(str, nums)) + ',' + str(evaluation) + '\n')

        state = state.parent


def preprocess(in_path: str, out_path: str) -> None:
    pgn = open(in_path, "r")
    out = open(out_path, "w")

    for i in range(100000000):
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        if int(game.headers["WhiteElo"]) > 2200 and int(game.headers["BlackElo"]) > 2200:
            print("creating Dataset for game", i)
            create_dataset(game, out)


if __name__ == "__main__":
    preprocess("../dataset/lichess_db_standard_rated_2021-10 small.pgn",
               "../dataset/nm_games small.csv")

from io import StringIO
from os import cpu_count
from typing import TextIO
import chess.pgn

from util import board_to_bitfields, pager

from multiprocessing import Lock, Pool
from bz2 import BZ2File

mutex = Lock()


def create_dataset(game: chess.pgn.Game, out: TextIO) -> None:
    state = game.end()
    while state:
        nums = board_to_bitfields(state.board())
        evaluation = state.eval()

        if evaluation is not None:
            evaluation = evaluation.relative.score(mate_score=10) / 100
            mutex.acquire()
            out.write(','.join(map(str, nums)) + ',' + str(evaluation) + '\n')
            mutex.release()

        state = state.parent


def process_game(lines: str, out: TextIO) -> None:
    game = chess.pgn.read_game(StringIO(lines))
    if game is None:
        return

    if int(game.headers["WhiteElo"]) > 2200 and int(game.headers["BlackElo"]) > 2200:
        create_dataset(game, out)


def preprocess(in_path: str, out_path: str) -> None:
    out = open(out_path, "w")

    with BZ2File(in_path, "rb") as in_file:
        with Pool(cpu_count()) as pool:
            for lines in pager(in_file):
                pool.apply_async(process_game, args=(lines, out))


if __name__ == "__main__":
    preprocess("../dataset/lichess_db_standard_rated_2021-10.pgn.bz2",
               "../dataset/nm_games.csv")

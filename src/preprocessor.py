from io import StringIO
import multiprocessing
from os import cpu_count
from typing import TextIO
import chess.pgn

from util import board_to_bitfields, pager

from bz2 import BZ2File

out_files = {}


def create_dataset(game: chess.pgn.Game) -> None:
    state = game.end()
    while state:
        nums = board_to_bitfields(state.board())
        evaluation = state.eval()

        if evaluation is not None:
            evaluation = evaluation.relative.score(mate_score=10) / 100
            out_files[multiprocessing.current_process()].write(
                ','.join(map(str, nums)) + ',' + str(evaluation) + '\n')

        state = state.parent


def process_game(lines: str, out_path: str) -> None:
    game = chess.pgn.read_game(StringIO(lines))
    if game is None:
        return

    cp = multiprocessing.current_process()
    if not out_files[cp]:
        out_files[cp] = open(out_path[:-4] + "." + cp.pid + out_path[-4:], 'w')

    if int(game.headers["WhiteElo"]) > 2200 and int(game.headers["BlackElo"]) > 2200:
        create_dataset(game)


def preprocess(in_path: str, out_path: str) -> None:

    with BZ2File(in_path, "rb") as in_file:
        with multiprocessing.Pool(cpu_count()) as pool:
            for lines in pager(in_file):
                pool.apply_async(process_game, args=(lines, out_path))


if __name__ == "__main__":
    preprocess("../dataset/lichess_db_standard_rated_2021-10.pgn.bz2",
               "../dataset/nm_games.csv")

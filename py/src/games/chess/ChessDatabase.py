import io
import random
import sys
import chess.pgn
import numpy as np
from pathlib import Path
from zipfile import ZipFile
from multiprocessing import Process

from tqdm import trange

from src.util.download import download


def load_content(year: int, month: int, folder: str) -> str:
    name = f'lichess_elite_{year}-{month:02}.zip'
    file = f'{folder}/{name}'

    download(f'https://database.nikonoel.fr/{name}', file)

    return file


def games_iterator(year: int, month: int, num_games_per_month: int):
    # unzip file in memory
    # then process the pgn file
    # for each game in the pgn file
    #     parse the game
    #     yield the game

    file = load_content(year, month, 'reference/data/chess_database')
    with ZipFile(file, 'r') as zip_ref:
        content = zip_ref.read(zip_ref.namelist()[0]).decode('utf-8')

    content = io.StringIO(content)
    for _ in trange(num_games_per_month):
        if not (game := chess.pgn.read_game(content)):
            break
        yield game


def process_month(year: int, month: int, num_games_per_month: int) -> list[Path]:
    from src.games.chess.ChessBoard import ChessBoard
    from src.games.chess.ChessGame import ChessGame
    from src.self_play.SelfPlayDataset import SelfPlayDataset
    from src.settings import TRAINING_ARGS

    chess_game = ChessGame()
    dataset = SelfPlayDataset()

    output_paths: list[Path] = []

    for game in games_iterator(year, month, num_games_per_month):
        try:
            winner = eval(game.headers['Result'])

            board = ChessBoard()
            for move in game.mainline_moves():
                if random.random() < TRAINING_ARGS.self_play.portion_of_samples_to_keep:
                    visit_counts = [(chess_game.encode_move(move, board), 1)]

                    for board_variation, visits in chess_game.symmetric_variations(board, visit_counts):
                        dataset.add_sample(
                            board_variation.copy().astype(np.int8),
                            visits,
                            winner * board.current_player,
                        )

                board.make_move(move)

            dataset.add_generation_stats(
                game_length=len(list(game.mainline_moves())),
                generation_time=0.0,
                resignation=False,
            )
        except Exception as e:
            from src.util.log import log, LogLevel

            log(f'Error in game: {game}', level=LogLevel.WARNING)
            log(e)
            import traceback

            traceback.print_exc()

        if dataset.stats.num_samples >= 20000:
            output_paths.append(dataset.save('reference/chess_database', year * 100 + month))
            dataset = SelfPlayDataset()

    output_paths.append(dataset.save('reference/chess_database', year * 100 + month))
    return output_paths


def retrieve_month_year_pairs(num_months: int) -> list[tuple[int, int]]:
    params: list[tuple[int, int]] = []
    year = 2024
    month = 10
    for _ in range(num_months):
        month -= 1
        if month == 0:
            month = 12
            year -= 1
        params.append((year, month))
    return params


if __name__ == '__main__':
    # load year and months from sys.argv
    if len(sys.argv) < 3 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python -m src.games.chess.ChessDatabase <number_of_months> <number_of_games_per_month>')
        print('Games are downloaded from https://database.nikonoel.fr/.')
        print('Make sure to check that the games are available for the year and months you are interested in.')
        sys.exit(1)

    num_months = int(sys.argv[1])
    num_games_per_month = int(sys.argv[2])

    # starting from 2024-10 go back in time by num_months
    params = retrieve_month_year_pairs(num_months)

    processes = [Process(target=process_month, args=(year, month, num_games_per_month)) for year, month in params]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

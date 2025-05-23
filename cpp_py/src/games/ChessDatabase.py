import io
import sys
import chess.pgn
import numpy as np
from pathlib import Path
from zipfile import ZipFile
from multiprocessing import Process

from tqdm import trange

from src.dataset.SelfPlayDatasetStats import SelfPlayDatasetStats
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

    file = load_content(year, month, 'reference/chess_database/data')
    with ZipFile(file, 'r') as zip_ref:
        content = zip_ref.read(zip_ref.namelist()[0]).decode('utf-8')

    content = io.StringIO(content)
    for _ in trange(num_games_per_month):
        if not (game := chess.pgn.read_game(content)):
            break
        yield game


def process_month(year: int, month: int, num_games_per_month: int) -> list[Path]:
    import AlphaZeroCpp
    from src.dataset.SelfPlayDataset import SelfPlayDataset

    dataset = SelfPlayDataset()

    output_paths: list[Path] = []

    num_datasets_written = 0

    for game in games_iterator(year, month, num_games_per_month):
        try:
            winner = eval(game.headers['Result'])

            board = chess.Board()
            for move in game.mainline_moves():
                encoded_board = np.array(AlphaZeroCpp.encode_board(board.fen()), dtype=np.uint64)
                visit_counts = np.array([(AlphaZeroCpp.encode_move(move), 1)], dtype=np.uint32)

                dataset.add_sample(
                    encoded_board,
                    visit_counts,
                    winner * (1 if board.turn == chess.WHITE else -1),
                )

                board.push(move)

            dataset.stats += SelfPlayDatasetStats(num_games=1)

        except Exception as e:
            from src.util.log import log, LogLevel

            log(f'Error in game: {game}', level=LogLevel.WARNING)
            log(e)

        if dataset.stats.num_samples >= 20000:
            output_paths.append(dataset.save('reference/chess_database', year * 100 + month, str(num_datasets_written)))
            num_datasets_written += 1
            dataset = SelfPlayDataset()

    output_paths.append(dataset.save('reference/chess_database', year * 100 + month, str(num_datasets_written)))
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
        print('Usage: python -m src.games.ChessDatabase <number_of_months> <number_of_games_per_month>')
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

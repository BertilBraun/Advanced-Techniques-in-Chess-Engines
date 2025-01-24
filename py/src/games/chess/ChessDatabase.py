import os
import io
import sys
import chess.pgn
import requests
from zipfile import ZipFile
from multiprocessing import Process

from tqdm import trange

from src.settings import *

from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.ChessGame import ChessGame
from src.self_play.SelfPlayDataset import SelfPlayDataset


def download(year: int, month: int, folder: str) -> str:
    name = f'lichess_elite_{year}-{month:02}.zip'
    file = f'{folder}/{name}'
    if os.path.exists(file):
        return file

    os.makedirs(folder, exist_ok=True)
    url = f'https://database.nikonoel.fr/{name}'
    print(f'Downloading {url}')
    r = requests.get(url)
    with open(file, 'wb') as f:
        f.write(r.content)
    return file


def games_iterator(year: int, month: int):
    # unzip file in memory
    # then process the pgn file
    # for each game in the pgn file
    #     parse the game
    #     yield the game

    file = download(year, month, 'reference/data/chess_database')
    with ZipFile(file, 'r') as zip_ref:
        content = zip_ref.read(zip_ref.namelist()[0]).decode('utf-8')

    content = io.StringIO(content)
    for _ in trange(200000):
        if not (game := chess.pgn.read_game(content)):
            break
        yield game


def process_month(year: int, month: int):
    chess_game = ChessGame()
    dataset = SelfPlayDataset()

    for game in games_iterator(year, month):
        winner = eval(game.headers['Result'])

        board = ChessBoard()
        for move in game.mainline_moves():
            encoded_board = chess_game.get_canonical_board(board)
            probabilities = chess_game.encode_moves([move])

            for board_variation, probability_variation in chess_game.symmetric_variations(encoded_board, probabilities):
                dataset.add_sample(
                    board_variation.copy().astype(np.int8),
                    probability_variation.copy().astype(np.float32),
                    winner * board.current_player,
                )

            board.make_move(move)

        dataset.add_generation_stats(
            num_games=1,
            generation_time=0.0,
            resignation=False,
        )

        if dataset.stats.num_samples >= 20000:
            dataset.save('reference/chess_database', year * 100 + month)
            dataset = SelfPlayDataset()

    dataset.save('reference/chess_database', year * 100 + month)


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
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python -m src.games.chess.ChessDatabase <number_of_months>')
        print('Games are downloaded from https://database.nikonoel.fr/.')
        print('Make sure to check that the games are available for the year and months you are interested in.')
        sys.exit(1)

    num_months = int(sys.argv[1])

    # starting from 2024-10 go back in time by num_months
    params = retrieve_month_year_pairs(num_months)

    processes = [Process(target=process_month, args=(year, month)) for year, month in params]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

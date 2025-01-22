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


def process_month(year: int, month: int):
    chess_game = ChessGame()
    dataset = SelfPlayDataset()

    file = download(year, month, 'reference/data')
    # unzip file in memory
    # then process the pgn file
    # for each game in the pgn file
    #     parse the game
    #     add the game to the dataset
    # write the dataset

    with ZipFile(file, 'r') as zip_ref:
        content = zip_ref.read(zip_ref.namelist()[0]).decode('utf-8')

    content = io.StringIO(content)
    for _ in trange(500):
        if not (game := chess.pgn.read_game(content)):
            break
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

    dataset.save('reference', month, 'chess_database')


if __name__ == '__main__':
    # load year and months from sys.argv
    if len(sys.argv) < 4 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python -m src.games.chess.ChessDatabase <year> <month_start> <month_end>')
        print('Games are downloaded from https://database.nikonoel.fr/.')
        print('Make sure to check that the games are available for the year and months you are interested in.')
        sys.exit(1)

    year = int(sys.argv[1])
    month_start = int(sys.argv[2])
    month_end = int(sys.argv[3])

    processes: list[Process] = []
    for month in range(month_start, month_end + 1):
        p = Process(target=process_month, args=(year, month))
        p.start()

        processes.append(p)

    for p in processes:
        p.join()

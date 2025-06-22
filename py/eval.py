import torch

from src.eval.TournamentManager import TournamentManager
from src.games.chess.comparison_bots.StockfishBot import ChessStockfishBot
from src.util.log import log
from src.settings import TRAINING_ARGS
from src.eval.GameManager import GameManager
from src.eval.HumanPlayer import HumanPlayer
from src.eval.AlphaZeroBotCpp import AlphaZeroBot
from src.util.save_paths import get_latest_model_iteration, model_save_path

HUMAN_PLAY = True
MAX_TIME_TO_THINK = 1.0
NETWORK_ONLY = False

STOCKFISH_SKILL_LEVEL = 4  # Stockfish skill level, can be adjusted for difficulty

MODEL_PATH = str(model_save_path(get_latest_model_iteration(TRAINING_ARGS.save_path), TRAINING_ARGS.save_path))
MODEL_PATH = R'/mnt/c/Users/berti/OneDrive/Desktop/zip9/training_data/chess/best_model.pt'


def alpha_zero_bot_factory(game_index: int) -> AlphaZeroBot:
    return AlphaZeroBot(
        current_model_path=MODEL_PATH,
        device_id=game_index % max(1, torch.cuda.device_count()),
        max_time_to_think=MAX_TIME_TO_THINK,
        network_eval_only=NETWORK_ONLY,
    )


def stockfish_bot_factory(_: int) -> ChessStockfishBot:
    return ChessStockfishBot(skill_level=STOCKFISH_SKILL_LEVEL, max_time_to_think=MAX_TIME_TO_THINK)


if __name__ == '__main__':
    if HUMAN_PLAY:
        game_manager = GameManager(
            HumanPlayer(),
            AlphaZeroBot(MODEL_PATH, device_id=0, max_time_to_think=MAX_TIME_TO_THINK, network_eval_only=NETWORK_ONLY),
            # ChessStockfishBot(skill_level=6, max_time_to_think=MAX_TIME_TO_THINK)
        )

        result = game_manager.play_game()
        log('Game over. Result:', result)
        log(game_manager.board)
    else:
        tournament_manager = TournamentManager(
            alpha_zero_bot_factory,
            stockfish_bot_factory,
            num_games=50,
        )
        results, games = tournament_manager.play_games()

        log('Tournament completed.')
        log('Games played:', len(games))
        log('Games:')
        for i, game in enumerate(games):
            log(f'Game {i + 1}:\n{game}\n')

        log('Results:', results)
        stockfish_bot_factory(0)  # To print the stockfish level and other settings

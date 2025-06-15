from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from src.games.Board import Player

if TYPE_CHECKING:
    from src.eval.Bot import Bot
from src.eval.GameManager import GameManager
from src.eval.ModelEvaluationPy import Results
import multiprocessing as mp


# define a type for a function which creates a bot
if TYPE_CHECKING:
    BotFactory = Callable[[int], Bot]


class TournamentManager:
    def __init__(self, main: BotFactory, opponent: BotFactory, num_games: int) -> None:
        """Initializes the tournament manager with main and opponent bot factories."""
        self.main = main
        self.opponent = opponent
        self.num_games = num_games

    def play_games(self, verify_moves=True) -> Results:
        """Manages the gameplay loop until the game is over, using multiprocessing for parallelization."""

        mp.set_start_method('spawn', force=True)  # Ensure the spawn method is used for multiprocessing

        # Split the games across processes
        # Use all available CPU cores
        with mp.Pool(processes=mp.cpu_count()) as pool:
            game_results = pool.starmap(self._play_single_game, [(i, verify_moves) for i in range(self.num_games)])

        # Combine results
        results = Results(0, 0, 0)

        for result, player_index in game_results:
            results.update(result, player_index)

        return results

    def _play_single_game(self, game_index: int, verify_moves: bool) -> tuple[Player | None, Player]:
        if game_index % 2 == 0:
            white = self.main(game_index)
            black = self.opponent(game_index)
            player_index = 1
        else:
            white = self.opponent(game_index)
            black = self.main(game_index)
            player_index = -1

        game_manager = GameManager(white, black)
        result = game_manager.play_game(verify_moves)
        return result, player_index

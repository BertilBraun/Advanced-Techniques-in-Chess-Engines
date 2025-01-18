from src.eval.Bot import Bot
from src.eval.GameManager import GameManager
from src.eval.ModelEvaluation import Results


class TournamentManager:
    def __init__(self, main: Bot, opponent: Bot, num_games: int) -> None:
        """Initializes the tournament manager with a list of bots."""
        self.players = [main, opponent]
        self.num_games = num_games

    async def play_games(self, verify_moves=True) -> Results:
        """Manages the gameplay loop until the game is over or a player quits."""
        results = Results(0, 0, 0)

        for i in range(self.num_games):
            if i % 2 == 0:
                white = self.players[0]
                black = self.players[1]
            else:
                white = self.players[1]
                black = self.players[0]

            game_manager = GameManager(white, black)
            result = await game_manager.play_game(verify_moves)
            results.update(result, 1 if white == self.players[0] else -1)

        return results

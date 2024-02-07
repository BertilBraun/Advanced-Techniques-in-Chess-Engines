from typing import List
from dataclasses import dataclass

from Framework import ChessBot, WHITE
from Framework.GameManager import GameManager, ThinkingTime
from Framework.GameResult import GameResult


@dataclass
class Statistics:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    thinking_time: float = 0.0
    
    def repr(self, total_rounds: int) -> str:
        return f"Wins: {self.wins}, Losses: {self.losses}, Draws: {self.draws}, Avg. Thinking Time: {self.thinking_time / total_rounds:.2f}s"

class Tournament:
    def __init__(self, participants: List[ChessBot], rounds: int) -> None:
        self.participants = participants
        self.rounds = rounds
        self.results = {bot: Statistics() for bot in participants}

    def run(self) -> dict[ChessBot, Statistics]:
        for i, first_bot in enumerate(self.participants):
            for second_bot in self.participants[i + 1:]:
                competitors = [first_bot, second_bot]
                
                for _ in range(self.rounds):
                    competitors = competitors[::-1] # Switch sides each round

                    white, black = competitors
                    
                    game_manager = GameManager(white, black)
                    result, thinking_time = game_manager.play_game()
                    self.update_statistics(white, black, result, thinking_time)
                    
        self.print_statistics()
        return self.results

    def update_statistics(self, white: ChessBot, black: ChessBot, result: GameResult, thinking_time: ThinkingTime) -> None:
        if result.winner is None: # draw
            self.results[white].draws += 1
            self.results[black].draws += 1
        elif result.winner == WHITE:
            self.results[white].wins += 1
            self.results[black].losses += 1
        else:
            self.results[white].losses += 1
            self.results[black].wins += 1

        # Accumulate thinking time
        self.results[white].thinking_time += thinking_time.white
        self.results[black].thinking_time += thinking_time.black

    def print_statistics(self) -> None:
        total_rounds = self.rounds * len(self.participants) - 1
        for name, stats in self.results.items():
            print(f"{name}: {stats.repr(total_rounds)}")
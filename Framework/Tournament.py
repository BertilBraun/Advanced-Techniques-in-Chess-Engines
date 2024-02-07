from tqdm import tqdm
from typing import Sequence
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
        length_num_rounds = len(str(total_rounds))
        return f"Wins: {self.wins:>{length_num_rounds}}, Losses: {self.losses:>{length_num_rounds}}, Draws: {self.draws:>{length_num_rounds}}, Avg. Thinking Time: {self.thinking_time / total_rounds:.2f}s"


class CompetitionStatistics(dict[ChessBot, Statistics]):
    
    def __init__(self, total_rounds: int) -> None:
        self.total_rounds = total_rounds
    
    def _update_statistics(self, white: ChessBot, black: ChessBot, result: GameResult, thinking_time: ThinkingTime) -> None:
        if result.winner is None:
            self[white].draws += 1
            self[black].draws += 1
        elif result.winner == WHITE:
            self[white].wins += 1
            self[black].losses += 1
        else:
            self[white].losses += 1
            self[black].wins += 1
            
        self[white].thinking_time += thinking_time.white
        self[black].thinking_time += thinking_time.black
         
    def _merge_into_this(self, other: "CompetitionStatistics") -> None:
        for bot, stats in other.items():
            self[bot].wins += stats.wins
            self[bot].losses += stats.losses
            self[bot].draws += stats.draws
            self[bot].thinking_time += stats.thinking_time
            
    def __repr__(self) -> str:
        out = f"Total Rounds: {self.total_rounds}\n"
        longest_bot_name = max(len(bot.name) for bot in self.keys())
        for bot, stats in self.items():
            out += f"{bot.name:<{longest_bot_name}} : {stats.repr(self.total_rounds)}\n"
        return out
           
    def __getitem__(self, key: ChessBot) -> Statistics:
        if key not in self:
            self[key] = Statistics()
        return super().get(key, Statistics())

@dataclass
class TournamentResults:
    total_results: CompetitionStatistics # Aggregated results of all the games played
    competition_results: dict[tuple[int, int], CompetitionStatistics] # (i, j) -> CompetitionStatistics where i, j are the indices of the participants that competed
    
    def __repr__(self) -> str:
        out = "Total Results:\n"
        out += self.total_results.__repr__()
        out += "\n\nCompetition Results:\n"
        for stats in self.competition_results.values():
            white, black = stats.keys()
            out += f"{white.name} vs {black.name}:\n{stats}\n"
        return out

class Tournament:
    def __init__(self, participants: Sequence[ChessBot], rounds: int) -> None:
        self.participants = participants
        self.rounds = rounds

    def run(self) -> TournamentResults:
        competition_results: dict[tuple[int, int], CompetitionStatistics] = {}
        
        for i, first_bot in tqdm(enumerate(self.participants), desc="Running tournament"):
            for j, second_bot in enumerate(self.participants[i + 1:]):
                competition_results[(i, j)] = self._run_competition(first_bot, second_bot)

        total_results = self._aggregate_results(competition_results)
        
        return TournamentResults(total_results, competition_results)

    def _run_competition(self, bot1: ChessBot, bot2: ChessBot) -> CompetitionStatistics:
        current_competitor_results = CompetitionStatistics(self.rounds)
        
        for round in range(self.rounds):
            white, black = (bot1, bot2) if round % 2 == 0 else (bot2, bot1)

            game_manager = GameManager(white, black)
            result, thinking_time = game_manager.play_game()
            current_competitor_results._update_statistics(white, black, result, thinking_time)
            
        return current_competitor_results
                
    def _aggregate_results(self, competition_results: dict[tuple[int, int], CompetitionStatistics]) -> CompetitionStatistics:
        total_rounds = self.rounds * (len(self.participants) - 1)
        total_results = CompetitionStatistics(total_rounds)
        
        for result in competition_results.values():
            total_results._merge_into_this(result)
            
        return total_results
    
    def __repr__(self) -> str:
        out = f"Tournament with {self.rounds} rounds\n"
        out += f"Participants: {[bot.name for bot in self.participants]}\n"
        return out
        
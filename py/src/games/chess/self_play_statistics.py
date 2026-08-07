from dataclasses import dataclass, field

from src.games.chess.dataset_statistics import SelfPlayDatasetStats


@dataclass
class SelfPlayStatistics:
    stats: SelfPlayDatasetStats = field(default_factory=SelfPlayDatasetStats)

    def add(self, statistics: SelfPlayDatasetStats) -> None:
        self.stats += statistics

    def record_completed_game(self, game_length: int, generation_time: float) -> None:
        self.add(
            SelfPlayDatasetStats(
                num_games=1,
                game_lengths=[game_length],
                total_generation_time=generation_time,
            )
        )

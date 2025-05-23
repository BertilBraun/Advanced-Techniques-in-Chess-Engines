from __future__ import annotations

from typing import NamedTuple


class SelfPlayDatasetStats(NamedTuple):
    num_samples: int = 0
    num_games: int = 0
    game_lengths: int = 0
    total_generation_time: float = 0.0
    resignations: int = 0
    num_too_long_games: int = 0

    def __repr__(self) -> str:
        return f"""Num samples: {self.num_samples}
Num games: {self.num_games}
Total generation time: {self.total_generation_time:.2f}s
Average game length: {self.game_lengths / self.num_games:.2f}
Average generation time: {self.total_generation_time / self.num_games:.2f}s/game
Resignations: {self.resignations/ self.num_games * 100:.2f}% ({self.resignations})
Too long games: {self.num_too_long_games}"""

    def __add__(self, other: SelfPlayDatasetStats) -> SelfPlayDatasetStats:
        return SelfPlayDatasetStats(
            num_samples=self.num_samples + other.num_samples,
            num_games=self.num_games + other.num_games,
            game_lengths=self.game_lengths + other.game_lengths,
            total_generation_time=self.total_generation_time + other.total_generation_time,
            resignations=self.resignations + other.resignations,
            num_too_long_games=self.num_too_long_games + other.num_too_long_games,
        )

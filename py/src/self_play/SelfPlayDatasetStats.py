from __future__ import annotations

from typing import NamedTuple

import numpy as np


class SelfPlayDatasetStats(NamedTuple):
    num_samples: int = 0
    num_games: int = 0
    game_lengths: list[int] = []
    total_generation_time: float = 0.0
    num_too_long_games: int = 0

    resignations: int = 0
    num_resignations_evaluated_to_end: int = 0
    num_winnable_resignations: int = 0
    num_moves_after_resignation: int = 0

    def overwrite(
        self,
        num_samples: int | None = None,
        num_games: int | None = None,
        game_lengths: list[int] | None = None,
        total_generation_time: float | None = None,
        num_too_long_games: int | None = None,
        resignations: int | None = None,
        num_resignations_evaluated_to_end: int | None = None,
        num_winnable_resignations: int | None = None,
        num_moves_after_resignation: int | None = None,
    ) -> SelfPlayDatasetStats:
        return SelfPlayDatasetStats(
            num_samples=num_samples if num_samples is not None else self.num_samples,
            num_games=num_games if num_games is not None else self.num_games,
            game_lengths=game_lengths if game_lengths is not None else self.game_lengths,
            total_generation_time=total_generation_time
            if total_generation_time is not None
            else self.total_generation_time,
            num_too_long_games=num_too_long_games if num_too_long_games is not None else self.num_too_long_games,
            resignations=resignations if resignations is not None else self.resignations,
            num_resignations_evaluated_to_end=num_resignations_evaluated_to_end
            if num_resignations_evaluated_to_end is not None
            else self.num_resignations_evaluated_to_end,
            num_winnable_resignations=num_winnable_resignations
            if num_winnable_resignations is not None
            else self.num_winnable_resignations,
            num_moves_after_resignation=num_moves_after_resignation
            if num_moves_after_resignation is not None
            else self.num_moves_after_resignation,
        )

    def __repr__(self) -> str:
        return f"""Num samples: {self.num_samples}
Num games: {self.num_games}
Total generation time: {self.total_generation_time:.2f}s
Average game length: {np.mean(self.game_lengths):.2f}
Average generation time: {self.total_generation_time / self.num_games:.2f}s/game
Too long games: {self.num_too_long_games}
Resignations: {self.resignations / self.num_games * 100:.2f}% ({self.resignations})
Resignations evaluated to end: {self.num_resignations_evaluated_to_end / self.num_games * 100:.2f}% ({self.num_resignations_evaluated_to_end})
Winnable resignations: {self.num_winnable_resignations / self.num_resignations_evaluated_to_end * 100:.2f}% ({self.num_winnable_resignations})
Average moves after resignation: {self.num_moves_after_resignation / self.num_resignations_evaluated_to_end if self.num_resignations_evaluated_to_end > 0 else 0:.2f}"""

    def __add__(self, other: SelfPlayDatasetStats) -> SelfPlayDatasetStats:
        return SelfPlayDatasetStats(
            num_samples=self.num_samples + other.num_samples,
            num_games=self.num_games + other.num_games,
            game_lengths=self.game_lengths + other.game_lengths,
            total_generation_time=self.total_generation_time + other.total_generation_time,
            num_too_long_games=self.num_too_long_games + other.num_too_long_games,
            resignations=self.resignations + other.resignations,
            num_resignations_evaluated_to_end=self.num_resignations_evaluated_to_end
            + other.num_resignations_evaluated_to_end,
            num_winnable_resignations=self.num_winnable_resignations + other.num_winnable_resignations,
            num_moves_after_resignation=self.num_moves_after_resignation + other.num_moves_after_resignation,
        )

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class SelfPlayDatasetStats(NamedTuple):
    num_samples: int = 0
    num_games: int = 0
    game_lengths: list[int] = []
    total_generation_time: float = 0.0
    num_too_long_games: int = 0
    capped_game_material_scores: list[float] = []
    low_material_termination_evaluations: int = 0
    low_material_terminations: int = 0
    low_material_termination_declines: int = 0
    low_material_termination_material_scores: list[float] = []

    resignations: int = 0
    num_resignations_evaluated_to_end: int = 0
    num_winnable_resignations: int = 0
    num_moves_after_resignation: int = 0
    resignation_audit_games_started: int = 0
    resignation_audit_games_completed: int = 0
    hypothetical_resignations: int = 0
    actual_resignations: int = 0
    resignation_audit_natural_triggers: int = 0
    resignation_audit_capped_triggers: int = 0
    resignation_audit_recovered_wins: int = 0
    resignation_audit_recovered_draws: int = 0
    resignation_audit_recovered_losses: int = 0
    resignation_audit_white_triggers: int = 0
    resignation_audit_black_triggers: int = 0
    resignation_audit_white_false_non_losses: int = 0
    resignation_audit_black_false_non_losses: int = 0
    resignation_audit_root_value_abs_sum: float = 0.0
    resignation_audit_root_value_count: int = 0
    resignation_audit_continuation_plies: int = 0
    resignation_audit_estimated_searches_saved: int = 0

    def overwrite(
        self,
        num_samples: int | None = None,
        num_games: int | None = None,
        game_lengths: list[int] | None = None,
        total_generation_time: float | None = None,
        num_too_long_games: int | None = None,
        capped_game_material_scores: list[float] | None = None,
        low_material_termination_evaluations: int | None = None,
        low_material_terminations: int | None = None,
        low_material_termination_declines: int | None = None,
        low_material_termination_material_scores: list[float] | None = None,
        resignations: int | None = None,
        num_resignations_evaluated_to_end: int | None = None,
        num_winnable_resignations: int | None = None,
        num_moves_after_resignation: int | None = None,
        resignation_audit_games_started: int | None = None,
        resignation_audit_games_completed: int | None = None,
        hypothetical_resignations: int | None = None,
        actual_resignations: int | None = None,
        resignation_audit_natural_triggers: int | None = None,
        resignation_audit_capped_triggers: int | None = None,
        resignation_audit_recovered_wins: int | None = None,
        resignation_audit_recovered_draws: int | None = None,
        resignation_audit_recovered_losses: int | None = None,
        resignation_audit_white_triggers: int | None = None,
        resignation_audit_black_triggers: int | None = None,
        resignation_audit_white_false_non_losses: int | None = None,
        resignation_audit_black_false_non_losses: int | None = None,
        resignation_audit_root_value_abs_sum: float | None = None,
        resignation_audit_root_value_count: int | None = None,
        resignation_audit_continuation_plies: int | None = None,
        resignation_audit_estimated_searches_saved: int | None = None,
    ) -> SelfPlayDatasetStats:
        return SelfPlayDatasetStats(
            num_samples=num_samples if num_samples is not None else self.num_samples,
            num_games=num_games if num_games is not None else self.num_games,
            game_lengths=game_lengths if game_lengths is not None else self.game_lengths,
            total_generation_time=total_generation_time
            if total_generation_time is not None
            else self.total_generation_time,
            num_too_long_games=num_too_long_games if num_too_long_games is not None else self.num_too_long_games,
            capped_game_material_scores=(
                capped_game_material_scores
                if capped_game_material_scores is not None
                else self.capped_game_material_scores
            ),
            low_material_termination_evaluations=(
                low_material_termination_evaluations
                if low_material_termination_evaluations is not None
                else self.low_material_termination_evaluations
            ),
            low_material_terminations=(
                low_material_terminations if low_material_terminations is not None else self.low_material_terminations
            ),
            low_material_termination_declines=(
                low_material_termination_declines
                if low_material_termination_declines is not None
                else self.low_material_termination_declines
            ),
            low_material_termination_material_scores=(
                low_material_termination_material_scores
                if low_material_termination_material_scores is not None
                else self.low_material_termination_material_scores
            ),
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
            resignation_audit_games_started=(
                resignation_audit_games_started
                if resignation_audit_games_started is not None
                else self.resignation_audit_games_started
            ),
            resignation_audit_games_completed=(
                resignation_audit_games_completed
                if resignation_audit_games_completed is not None
                else self.resignation_audit_games_completed
            ),
            hypothetical_resignations=(
                hypothetical_resignations if hypothetical_resignations is not None else self.hypothetical_resignations
            ),
            actual_resignations=actual_resignations if actual_resignations is not None else self.actual_resignations,
            resignation_audit_natural_triggers=(
                resignation_audit_natural_triggers
                if resignation_audit_natural_triggers is not None
                else self.resignation_audit_natural_triggers
            ),
            resignation_audit_capped_triggers=(
                resignation_audit_capped_triggers
                if resignation_audit_capped_triggers is not None
                else self.resignation_audit_capped_triggers
            ),
            resignation_audit_recovered_wins=(
                resignation_audit_recovered_wins
                if resignation_audit_recovered_wins is not None
                else self.resignation_audit_recovered_wins
            ),
            resignation_audit_recovered_draws=(
                resignation_audit_recovered_draws
                if resignation_audit_recovered_draws is not None
                else self.resignation_audit_recovered_draws
            ),
            resignation_audit_recovered_losses=(
                resignation_audit_recovered_losses
                if resignation_audit_recovered_losses is not None
                else self.resignation_audit_recovered_losses
            ),
            resignation_audit_white_triggers=(
                resignation_audit_white_triggers
                if resignation_audit_white_triggers is not None
                else self.resignation_audit_white_triggers
            ),
            resignation_audit_black_triggers=(
                resignation_audit_black_triggers
                if resignation_audit_black_triggers is not None
                else self.resignation_audit_black_triggers
            ),
            resignation_audit_white_false_non_losses=(
                resignation_audit_white_false_non_losses
                if resignation_audit_white_false_non_losses is not None
                else self.resignation_audit_white_false_non_losses
            ),
            resignation_audit_black_false_non_losses=(
                resignation_audit_black_false_non_losses
                if resignation_audit_black_false_non_losses is not None
                else self.resignation_audit_black_false_non_losses
            ),
            resignation_audit_root_value_abs_sum=(
                resignation_audit_root_value_abs_sum
                if resignation_audit_root_value_abs_sum is not None
                else self.resignation_audit_root_value_abs_sum
            ),
            resignation_audit_root_value_count=(
                resignation_audit_root_value_count
                if resignation_audit_root_value_count is not None
                else self.resignation_audit_root_value_count
            ),
            resignation_audit_continuation_plies=(
                resignation_audit_continuation_plies
                if resignation_audit_continuation_plies is not None
                else self.resignation_audit_continuation_plies
            ),
            resignation_audit_estimated_searches_saved=(
                resignation_audit_estimated_searches_saved
                if resignation_audit_estimated_searches_saved is not None
                else self.resignation_audit_estimated_searches_saved
            ),
        )

    def __repr__(self) -> str:
        return f"""Num samples: {self.num_samples}
Num games: {self.num_games}
Total generation time: {self.total_generation_time:.2f}s
Average game length: {np.mean(self.game_lengths):.2f}
Average generation time: {self.total_generation_time / self.num_games:.2f}s/game
Too long games: {self.num_too_long_games}
Low-material terminations: {self.low_material_terminations} / {self.low_material_termination_evaluations}
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
            capped_game_material_scores=self.capped_game_material_scores + other.capped_game_material_scores,
            low_material_termination_evaluations=(
                self.low_material_termination_evaluations + other.low_material_termination_evaluations
            ),
            low_material_terminations=self.low_material_terminations + other.low_material_terminations,
            low_material_termination_declines=(
                self.low_material_termination_declines + other.low_material_termination_declines
            ),
            low_material_termination_material_scores=(
                self.low_material_termination_material_scores + other.low_material_termination_material_scores
            ),
            resignations=self.resignations + other.resignations,
            num_resignations_evaluated_to_end=self.num_resignations_evaluated_to_end
            + other.num_resignations_evaluated_to_end,
            num_winnable_resignations=self.num_winnable_resignations + other.num_winnable_resignations,
            num_moves_after_resignation=self.num_moves_after_resignation + other.num_moves_after_resignation,
            resignation_audit_games_started=(
                self.resignation_audit_games_started + other.resignation_audit_games_started
            ),
            resignation_audit_games_completed=(
                self.resignation_audit_games_completed + other.resignation_audit_games_completed
            ),
            hypothetical_resignations=self.hypothetical_resignations + other.hypothetical_resignations,
            actual_resignations=self.actual_resignations + other.actual_resignations,
            resignation_audit_natural_triggers=(
                self.resignation_audit_natural_triggers + other.resignation_audit_natural_triggers
            ),
            resignation_audit_capped_triggers=(
                self.resignation_audit_capped_triggers + other.resignation_audit_capped_triggers
            ),
            resignation_audit_recovered_wins=(
                self.resignation_audit_recovered_wins + other.resignation_audit_recovered_wins
            ),
            resignation_audit_recovered_draws=(
                self.resignation_audit_recovered_draws + other.resignation_audit_recovered_draws
            ),
            resignation_audit_recovered_losses=(
                self.resignation_audit_recovered_losses + other.resignation_audit_recovered_losses
            ),
            resignation_audit_white_triggers=(
                self.resignation_audit_white_triggers + other.resignation_audit_white_triggers
            ),
            resignation_audit_black_triggers=(
                self.resignation_audit_black_triggers + other.resignation_audit_black_triggers
            ),
            resignation_audit_white_false_non_losses=(
                self.resignation_audit_white_false_non_losses + other.resignation_audit_white_false_non_losses
            ),
            resignation_audit_black_false_non_losses=(
                self.resignation_audit_black_false_non_losses + other.resignation_audit_black_false_non_losses
            ),
            resignation_audit_root_value_abs_sum=(
                self.resignation_audit_root_value_abs_sum + other.resignation_audit_root_value_abs_sum
            ),
            resignation_audit_root_value_count=(
                self.resignation_audit_root_value_count + other.resignation_audit_root_value_count
            ),
            resignation_audit_continuation_plies=(
                self.resignation_audit_continuation_plies + other.resignation_audit_continuation_plies
            ),
            resignation_audit_estimated_searches_saved=(
                self.resignation_audit_estimated_searches_saved + other.resignation_audit_estimated_searches_saved
            ),
        )

from __future__ import annotations

from pydantic import Field, model_validator

from src.experiment.generation_schedule import (
    FloatGenerationSchedule,
    IntegerGenerationSchedule,
    defined_schedule_values,
)
from src.self_play.parameters import ResolvedSelfPlayParameters
from src.util.frozen_model import FrozenModel


class SelfPlaySearchParams(FrozenModel):
    full_searches: IntegerGenerationSchedule
    fast_searches: IntegerGenerationSchedule
    parallel_searches: int = Field(gt=0)
    dirichlet_epsilon: FloatGenerationSchedule
    dirichlet_alpha: FloatGenerationSchedule
    exploration_constant: FloatGenerationSchedule
    minimum_root_visits: IntegerGenerationSchedule

    @model_validator(mode='after')
    def validate_scheduled_values(self) -> SelfPlaySearchParams:
        if any(value <= self.parallel_searches for value in defined_schedule_values(self.full_searches)):
            raise ValueError('Every full-search budget must exceed the parallel-search count.')
        if any(value <= 0 for value in defined_schedule_values(self.fast_searches)):
            raise ValueError('Every fast-search budget must be positive.')
        if any(not 0.0 <= value <= 1.0 for value in defined_schedule_values(self.dirichlet_epsilon)):
            raise ValueError('Dirichlet epsilon must remain in [0, 1].')
        if any(value <= 0.0 for value in defined_schedule_values(self.dirichlet_alpha)):
            raise ValueError('Dirichlet alpha must remain positive.')
        if any(value <= 0.0 for value in defined_schedule_values(self.exploration_constant)):
            raise ValueError('Exploration constant must remain positive.')
        if any(value < 0 for value in defined_schedule_values(self.minimum_root_visits)):
            raise ValueError('Minimum root visits must remain nonnegative.')
        return self


class BatchedInferenceParams(FrozenModel):
    inference_workers: int = Field(gt=0)
    inference_batch_size: int = Field(gt=0)
    outstanding_batches_per_worker: int = Field(ge=1, le=2)


class SelfPlayConfiguration(FrozenModel):
    search: SelfPlaySearchParams
    inference: BatchedInferenceParams
    maximum_random_opening_plies: IntegerGenerationSchedule
    full_search_probability: FloatGenerationSchedule
    retained_root_visit_fraction: FloatGenerationSchedule
    greedy_after_ply: IntegerGenerationSchedule
    starting_temperature: FloatGenerationSchedule
    final_temperature: FloatGenerationSchedule
    primary_sample_weight: FloatGenerationSchedule
    detailed_statistics_workers: int = Field(default=1, ge=0)

    @model_validator(mode='after')
    def validate_temperatures(self) -> SelfPlayConfiguration:
        for schedule, name in (
            (self.starting_temperature, 'Starting temperature'),
            (self.final_temperature, 'Final temperature'),
        ):
            if any(value <= 0.0 for value in defined_schedule_values(schedule)):
                raise ValueError(f'{name} must remain positive.')
        if any(value < 0 for value in defined_schedule_values(self.maximum_random_opening_plies)):
            raise ValueError('Maximum random opening plies must remain nonnegative.')
        if any(value <= 0 for value in defined_schedule_values(self.greedy_after_ply)):
            raise ValueError('Greedy ply must remain positive.')
        if any(not 0.0 < value <= 1.0 for value in defined_schedule_values(self.full_search_probability)):
            raise ValueError('Full-search probability must remain in (0, 1].')
        if any(not 0.0 <= value <= 1.0 for value in defined_schedule_values(self.retained_root_visit_fraction)):
            raise ValueError('Retained-root fraction must remain in [0, 1].')
        if any(value <= 0.0 for value in defined_schedule_values(self.primary_sample_weight)):
            raise ValueError('Primary sample weight must remain positive.')
        return self

    def resolve(
        self,
        model_generation: int,
        maximum_game_plies: int | None,
    ) -> ResolvedSelfPlayParameters:
        search = self.search
        return ResolvedSelfPlayParameters(
            maximum_random_opening_plies=self.maximum_random_opening_plies.value_at(model_generation),
            full_search_probability=self.full_search_probability.value_at(model_generation),
            parallel_searches=search.parallel_searches,
            full_searches=search.full_searches.value_at(model_generation),
            fast_searches=search.fast_searches.value_at(model_generation),
            minimum_root_visits=search.minimum_root_visits.value_at(model_generation),
            exploration_constant=search.exploration_constant.value_at(model_generation),
            dirichlet_alpha=search.dirichlet_alpha.value_at(model_generation),
            dirichlet_epsilon=search.dirichlet_epsilon.value_at(model_generation),
            retained_root_visit_fraction=self.retained_root_visit_fraction.value_at(model_generation),
            starting_temperature=self.starting_temperature.value_at(model_generation),
            final_temperature=self.final_temperature.value_at(model_generation),
            greedy_after_ply=self.greedy_after_ply.value_at(model_generation),
            maximum_game_plies=maximum_game_plies,
            primary_sample_weight=self.primary_sample_weight.value_at(model_generation),
        )

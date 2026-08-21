from __future__ import annotations

from pydantic import Field, model_validator

from src.experiment.generation_schedule import IntegerGenerationSchedule, defined_schedule_values
from src.util.frozen_model import FrozenModel


class ReplayConfiguration(FrozenModel):
    capacity: IntegerGenerationSchedule
    maximum_capacity: int = Field(gt=0)
    maximum_policy_entries: int = Field(ge=1, le=255)
    materialization_processes: int = Field(default=1, ge=1)

    @model_validator(mode='after')
    def validate_capacity(self) -> ReplayConfiguration:
        capacities = defined_schedule_values(self.capacity)
        if any(capacity <= 0 for capacity in capacities):
            raise ValueError('Replay capacity must remain positive.')
        if any(capacity > self.maximum_capacity for capacity in capacities):
            raise ValueError('Scheduled replay capacity cannot exceed its static maximum capacity.')
        return self

    def capacity_at(self, model_generation: int) -> int:
        return self.capacity.value_at(model_generation)

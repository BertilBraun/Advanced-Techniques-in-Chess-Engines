from __future__ import annotations

from pydantic import Field, model_validator
from src.util.frozen_model import FrozenModel
from src.util.generation_schedule import IntegerGenerationSchedule, defined_schedule_values


class ReplayConfiguration(FrozenModel):
    capacity: IntegerGenerationSchedule
    maximum_capacity: int = Field(gt=0)
    maximum_policy_entries: int = Field(ge=1, le=255)
    materialization_processes: int = Field(default=1, ge=1)
    materialization_shard_maximum_games: int = Field(default=32, ge=1)
    materialization_shard_target_source_bytes: int = Field(default=16 * 1024 * 1024, ge=1)
    materialization_staging_shard_limit: int = Field(default=96, ge=1)
    materialization_inbox_rename_cap: int = Field(default=4096, ge=1)
    materialization_rejection_window_games: int = Field(default=512, ge=1)
    materialization_rejection_rate_ceiling: float = Field(default=0.05, ge=0.0, le=1.0)

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

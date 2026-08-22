from __future__ import annotations

from pydantic import Field
from src.util.frozen_model import FrozenModel


class TrainingProgress(FrozenModel):
    completed_optimizer_steps: int = Field(ge=0)
    optimizer_steps_per_generation: int = Field(gt=0)

    @property
    def model_generation(self) -> int:
        assert self.completed_optimizer_steps % self.optimizer_steps_per_generation == 0, (
            'Completed optimizer steps must be a multiple of optimizer steps per generation.'
        )
        return self.completed_optimizer_steps // self.optimizer_steps_per_generation

    @property
    def next_generation(self) -> 'TrainingProgress':
        return TrainingProgress(
            completed_optimizer_steps=self.completed_optimizer_steps + self.optimizer_steps_per_generation,
            optimizer_steps_per_generation=self.optimizer_steps_per_generation,
        )

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from src.az.config.base import FrozenModel
from src.az.games.go.configuration import ResidualGoModelConfiguration


class FixedModelSchedule(FrozenModel):
    kind: Literal['fixed']
    architecture: ResidualGoModelConfiguration


class ModelStage(FrozenModel):
    start_elapsed_seconds: int = Field(ge=0)
    architecture: ResidualGoModelConfiguration


class ProgressiveModelSchedule(FrozenModel):
    kind: Literal['progressive']
    stages: tuple[ModelStage, ...] = Field(min_length=2)

    @model_validator(mode='after')
    def validate_stages(self) -> ProgressiveModelSchedule:
        starts = tuple(stage.start_elapsed_seconds for stage in self.stages)
        if starts[0] != 0 or tuple(sorted(set(starts))) != starts:
            raise ValueError('Progressive model stages must start at zero and increase strictly.')
        return self


ModelSchedule = Annotated[FixedModelSchedule | ProgressiveModelSchedule, Field(discriminator='kind')]


class ModelConfiguration(FrozenModel):
    schedule: ModelSchedule

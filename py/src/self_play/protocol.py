from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from src.training.checkpoint import CheckpointReference
from src.util.frozen_model import FrozenModel


class StatisticsLevel(str, Enum):
    BASIC = 'basic'
    DETAILED = 'detailed'


class SelfPlayStatistics(FrozenModel):
    completed_generation: int
    level: StatisticsLevel
    completed_search_batches: int


class RunningSelfPlayState(FrozenModel):
    kind: Literal['running'] = 'running'
    checkpoint: CheckpointReference
    completed_generation_statistics: StatisticsLevel | None = None


class PausedSelfPlayState(FrozenModel):
    kind: Literal['paused'] = 'paused'


class StoppedSelfPlayState(FrozenModel):
    kind: Literal['stopped'] = 'stopped'


SelfPlayDesiredState: TypeAlias = Annotated[
    RunningSelfPlayState | PausedSelfPlayState | StoppedSelfPlayState,
    Field(discriminator='kind'),
]


class RunningSelfPlayStateApplied(FrozenModel):
    kind: Literal['running'] = 'running'
    worker_id: int
    loaded_generation: int
    loaded_inference_model_sha256: str
    completed_generation_statistics: SelfPlayStatistics | None


class StoppedSelfPlayStateApplied(FrozenModel):
    kind: Literal['stopped'] = 'stopped'
    worker_id: int


SelfPlayStateApplied: TypeAlias = Annotated[
    RunningSelfPlayStateApplied | StoppedSelfPlayStateApplied,
    Field(discriminator='kind'),
]

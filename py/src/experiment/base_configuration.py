from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from src.experiment.run_contract import (
    EnvironmentConfiguration,
    HardwareConfiguration,
    TrainingStage,
)
from src.training.configuration import TrainingArgs
from src.util.frozen_model import FrozenModel


class WeightsOnlyResumeConfiguration(FrozenModel):
    mode: Literal['weights_only']
    model_path: str


class RandomInitializationResumeConfiguration(FrozenModel):
    mode: Literal['random_initialization']


ResumeConfiguration = Annotated[
    WeightsOnlyResumeConfiguration | RandomInitializationResumeConfiguration,
    Field(discriminator='mode'),
]


class ExperimentRunConfiguration(FrozenModel):
    run_name: str = Field(min_length=1)
    tensorboard_run_directory: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    stage: TrainingStage
    requires_explicit_approval: bool
    resume: ResumeConfiguration
    hardware: HardwareConfiguration
    environment: EnvironmentConfiguration


class BaseExperimentConfiguration(FrozenModel):
    run: ExperimentRunConfiguration
    training: TrainingArgs

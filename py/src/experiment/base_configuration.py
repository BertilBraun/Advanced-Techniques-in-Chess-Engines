from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from src.evaluation.configuration import EvaluationConfiguration
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


class CheckpointResumeConfiguration(FrozenModel):
    mode: Literal['checkpoint']
    checkpoint_manifest_path: str
    generation: int = Field(gt=0)


class RandomInitializationResumeConfiguration(FrozenModel):
    mode: Literal['random_initialization']


ResumeConfiguration = Annotated[
    CheckpointResumeConfiguration | WeightsOnlyResumeConfiguration | RandomInitializationResumeConfiguration,
    Field(discriminator='mode'),
]


def initial_generation(configuration: ResumeConfiguration) -> int:
    match configuration:
        case CheckpointResumeConfiguration(generation=generation):
            return generation
        case WeightsOnlyResumeConfiguration() | RandomInitializationResumeConfiguration():
            return 0


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
    evaluation: EvaluationConfiguration

    @model_validator(mode='after')
    def validate_resume_progress(self) -> BaseExperimentConfiguration:
        starting_generation = initial_generation(self.run.resume)
        credit = self.training.lifecycle.credit
        maximum_optimizer_steps = credit.maximum_optimizer_steps
        starting_optimizer_steps = starting_generation * credit.optimizer_steps_per_quantum
        if maximum_optimizer_steps is not None and maximum_optimizer_steps < starting_optimizer_steps:
            raise ValueError('Maximum optimizer steps cannot precede the resumed checkpoint generation.')
        return self

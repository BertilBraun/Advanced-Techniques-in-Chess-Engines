from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator
from src.util.frozen_model import ConfigurationPath, FrozenModel
from src.util.generation_schedule import FloatGenerationSchedule, defined_schedule_values


class DisabledTrainingQuantization(FrozenModel):
    kind: Literal['disabled'] = 'disabled'


class TensorRtInt8QatConfiguration(FrozenModel):
    kind: Literal['tensorrt_int8_qat'] = 'tensorrt_int8_qat'
    fold_after_optimizer_steps: int = Field(default=1_000, gt=0)
    calibration_positions: int = Field(default=2_048, gt=0)
    recalibration_interval_generations: int = Field(default=1, gt=0)
    deployment_learning_rate: FloatGenerationSchedule | Literal['inherit']
    deployment_warmup_optimizer_steps: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_deployment_learning_rate(self) -> TensorRtInt8QatConfiguration:
        if self.deployment_learning_rate != 'inherit' and any(
            value <= 0.0 for value in defined_schedule_values(self.deployment_learning_rate)
        ):
            raise ValueError('Deployment learning-rate schedule values must be positive.')
        return self


TrainingQuantizationConfiguration: TypeAlias = Annotated[
    DisabledTrainingQuantization | TensorRtInt8QatConfiguration,
    Field(discriminator='kind'),
]


class QatCheckpointPhase(str, Enum):
    PRE_FOLD = 'pre_fold'
    DEPLOYMENT = 'deployment'


class QatStateIdentity(FrozenModel):
    phase: QatCheckpointPhase
    completed_optimizer_steps: int = Field(ge=0)
    path: ConfigurationPath
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


@dataclass(frozen=True)
class LearningRateWarmupProgress:
    warmup_optimizer_steps: int
    completed_optimizer_steps: int


def qat_phase_warmup_progress(
    initial_warmup_optimizer_steps: int,
    quantization: DisabledTrainingQuantization | TensorRtInt8QatConfiguration,
    qat_state: QatStateIdentity | None,
    completed_optimizer_steps: int,
) -> LearningRateWarmupProgress:
    match quantization:
        case DisabledTrainingQuantization():
            assert qat_state is None
            return LearningRateWarmupProgress(initial_warmup_optimizer_steps, completed_optimizer_steps)
        case TensorRtInt8QatConfiguration(
            fold_after_optimizer_steps=fold_after_optimizer_steps,
            deployment_warmup_optimizer_steps=deployment_warmup_optimizer_steps,
        ):
            assert qat_state is not None
            if qat_state.phase is QatCheckpointPhase.PRE_FOLD:
                return LearningRateWarmupProgress(initial_warmup_optimizer_steps, completed_optimizer_steps)
            assert completed_optimizer_steps >= fold_after_optimizer_steps
            return LearningRateWarmupProgress(
                deployment_warmup_optimizer_steps,
                completed_optimizer_steps - fold_after_optimizer_steps,
            )


def expected_qat_phase(
    configuration: TensorRtInt8QatConfiguration,
    completed_optimizer_steps: int,
) -> QatCheckpointPhase:
    if completed_optimizer_steps < 0:
        raise ValueError('Completed optimizer steps must be nonnegative.')
    if completed_optimizer_steps < configuration.fold_after_optimizer_steps:
        return QatCheckpointPhase.PRE_FOLD
    return QatCheckpointPhase.DEPLOYMENT

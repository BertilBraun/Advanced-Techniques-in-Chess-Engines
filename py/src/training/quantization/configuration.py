from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field
from src.util.frozen_model import ConfigurationPath, FrozenModel


class DisabledTrainingQuantization(FrozenModel):
    kind: Literal['disabled'] = 'disabled'


class TensorRtInt8QatConfiguration(FrozenModel):
    kind: Literal['tensorrt_int8_qat'] = 'tensorrt_int8_qat'
    fold_after_optimizer_steps: int = Field(default=1_000, gt=0)
    calibration_positions: int = Field(default=2_048, gt=0)
    recalibration_interval_generations: int = Field(default=1, gt=0)


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


def expected_qat_phase(
    configuration: TensorRtInt8QatConfiguration,
    completed_optimizer_steps: int,
) -> QatCheckpointPhase:
    if completed_optimizer_steps < 0:
        raise ValueError('Completed optimizer steps must be nonnegative.')
    if completed_optimizer_steps < configuration.fold_after_optimizer_steps:
        return QatCheckpointPhase.PRE_FOLD
    return QatCheckpointPhase.DEPLOYMENT

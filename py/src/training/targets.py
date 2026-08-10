from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from src.experiment.generation_schedule import FloatGenerationSchedule, defined_schedule_values
from src.util.frozen_model import FrozenModel


class NextPolicyTargetConfiguration(FrozenModel):
    kind: Literal['next_policy'] = 'next_policy'
    ply_offset: int = Field(gt=0)
    loss_weight: FloatGenerationSchedule

    @model_validator(mode='after')
    def validate_loss_weight(self) -> NextPolicyTargetConfiguration:
        if any(value < 0.0 for value in defined_schedule_values(self.loss_weight)):
            raise ValueError('Auxiliary loss weight must remain nonnegative.')
        return self


class RemainingGameLengthTargetConfiguration(FrozenModel):
    kind: Literal['remaining_game_length'] = 'remaining_game_length'
    normalization_scale: float = Field(gt=0.0)
    loss_weight: FloatGenerationSchedule

    @model_validator(mode='after')
    def validate_loss_weight(self) -> RemainingGameLengthTargetConfiguration:
        if any(value < 0.0 for value in defined_schedule_values(self.loss_weight)):
            raise ValueError('Auxiliary loss weight must remain nonnegative.')
        return self


AuxiliaryTargetConfiguration: TypeAlias = Annotated[
    NextPolicyTargetConfiguration | RemainingGameLengthTargetConfiguration,
    Field(discriminator='kind'),
]


@dataclass(frozen=True)
class NextPolicyHeadLayout:
    kind: Literal['next_policy']
    action_size: int
    ply_offset: int

    def __post_init__(self) -> None:
        if self.action_size <= 0 or self.ply_offset <= 0:
            raise ValueError('Next-policy head dimensions and offset must be positive.')


@dataclass(frozen=True)
class RemainingGameLengthHeadLayout:
    kind: Literal['remaining_game_length']
    normalization_scale: float
    output_size: Literal[1] = 1

    def __post_init__(self) -> None:
        if self.normalization_scale <= 0.0:
            raise ValueError('Remaining-game-length normalization scale must be positive.')


AuxiliaryHeadLayout: TypeAlias = NextPolicyHeadLayout | RemainingGameLengthHeadLayout


def auxiliary_head_output_size(head: AuxiliaryHeadLayout) -> int:
    match head:
        case NextPolicyHeadLayout(action_size=action_size):
            return action_size
        case RemainingGameLengthHeadLayout(output_size=output_size):
            return output_size


@dataclass(frozen=True)
class TrainingTargetLayout:
    action_size: int
    wdl_size: Literal[3]
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...]

    def __post_init__(self) -> None:
        if self.action_size <= 0:
            raise ValueError('Primary policy action size must be positive.')


def build_training_target_layout(
    action_size: int,
    auxiliary_targets: tuple[AuxiliaryTargetConfiguration, ...],
) -> TrainingTargetLayout:
    heads: list[AuxiliaryHeadLayout] = []
    for target in auxiliary_targets:
        match target:
            case NextPolicyTargetConfiguration(ply_offset=ply_offset):
                heads.append(NextPolicyHeadLayout(kind='next_policy', action_size=action_size, ply_offset=ply_offset))
            case RemainingGameLengthTargetConfiguration(normalization_scale=normalization_scale):
                heads.append(
                    RemainingGameLengthHeadLayout(
                        kind='remaining_game_length',
                        normalization_scale=normalization_scale,
                    )
                )
    return TrainingTargetLayout(action_size=action_size, wdl_size=3, auxiliary_heads=tuple(heads))

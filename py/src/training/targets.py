from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator
from src.util.frozen_model import FrozenModel
from src.util.generation_schedule import ConstantSchedule, FloatGenerationSchedule, defined_schedule_values


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
    loss_weight: FloatGenerationSchedule
    normalization_scale: float = Field(gt=0.0)

    @model_validator(mode='after')
    def validate_loss_weight(self) -> RemainingGameLengthTargetConfiguration:
        if any(value < 0.0 for value in defined_schedule_values(self.loss_weight)):
            raise ValueError('Auxiliary loss weight must remain nonnegative.')
        return self


class FutureSearchValueTargetConfiguration(FrozenModel):
    kind: Literal['future_search_value'] = 'future_search_value'
    ply_offset: int = Field(gt=0)
    smooth_l1_beta: float = Field(default=0.1, gt=0.0)
    loss_weight: FloatGenerationSchedule

    @model_validator(mode='after')
    def validate_loss_weight(self) -> FutureSearchValueTargetConfiguration:
        if any(value < 0.0 for value in defined_schedule_values(self.loss_weight)):
            raise ValueError('Auxiliary loss weight must remain nonnegative.')
        return self


class IrreversibleProgressTargetConfiguration(FrozenModel):
    kind: Literal['irreversible_progress'] = 'irreversible_progress'
    horizon_plies: int = Field(gt=0)
    loss_weight: FloatGenerationSchedule

    @model_validator(mode='after')
    def validate_loss_weight(self) -> IrreversibleProgressTargetConfiguration:
        if any(value < 0.0 for value in defined_schedule_values(self.loss_weight)):
            raise ValueError('Auxiliary loss weight must remain nonnegative.')
        return self


class LegalMovesTargetConfiguration(FrozenModel):
    kind: Literal['legal_moves'] = 'legal_moves'
    loss_weight: FloatGenerationSchedule

    @model_validator(mode='after')
    def validate_loss_weight(self) -> LegalMovesTargetConfiguration:
        if any(value < 0.0 for value in defined_schedule_values(self.loss_weight)):
            raise ValueError('Auxiliary loss weight must remain nonnegative.')
        return self


class SearchBudgetTargetConfiguration(FrozenModel):
    kind: Literal['search_budget'] = 'search_budget'
    loss_weight: FloatGenerationSchedule = ConstantSchedule[float](value=0.2)

    @model_validator(mode='after')
    def validate_loss_weight(self) -> SearchBudgetTargetConfiguration:
        if any(value < 0.0 for value in defined_schedule_values(self.loss_weight)):
            raise ValueError('Auxiliary loss weight must remain nonnegative.')
        return self


AuxiliaryTargetConfiguration: TypeAlias = Annotated[
    NextPolicyTargetConfiguration
    | RemainingGameLengthTargetConfiguration
    | FutureSearchValueTargetConfiguration
    | IrreversibleProgressTargetConfiguration
    | LegalMovesTargetConfiguration
    | SearchBudgetTargetConfiguration,
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


@dataclass(frozen=True)
class FutureSearchValueHeadLayout:
    kind: Literal['future_search_value']
    ply_offset: int
    smooth_l1_beta: float
    output_size: Literal[1] = 1

    def __post_init__(self) -> None:
        if self.ply_offset <= 0 or self.smooth_l1_beta <= 0.0:
            raise ValueError('Future-search-value offset and beta must be positive.')


@dataclass(frozen=True)
class IrreversibleProgressHeadLayout:
    kind: Literal['irreversible_progress']
    horizon_plies: int
    output_size: Literal[1] = 1

    def __post_init__(self) -> None:
        if self.horizon_plies <= 0:
            raise ValueError('Irreversible-progress horizon must be positive.')


@dataclass(frozen=True)
class LegalMovesHeadLayout:
    kind: Literal['legal_moves']
    action_size: int

    def __post_init__(self) -> None:
        if self.action_size <= 0:
            raise ValueError('Legal-moves action size must be positive.')


@dataclass(frozen=True)
class SearchBudgetHeadLayout:
    kind: Literal['search_budget']
    output_size: Literal[1] = 1


AuxiliaryHeadLayout: TypeAlias = (
    NextPolicyHeadLayout
    | RemainingGameLengthHeadLayout
    | FutureSearchValueHeadLayout
    | IrreversibleProgressHeadLayout
    | LegalMovesHeadLayout
    | SearchBudgetHeadLayout
)


def search_budget_auxiliary_index(auxiliary_heads: tuple[AuxiliaryHeadLayout, ...]) -> int | None:
    indices: list[int] = []
    for index, head in enumerate(auxiliary_heads):
        match head:
            case SearchBudgetHeadLayout():
                indices.append(index)
    if not indices:
        return None
    if len(indices) > 1:
        raise ValueError('A training layout may contain at most one search-budget head.')
    return indices[0]


def auxiliary_head_output_size(head: AuxiliaryHeadLayout) -> int:
    match head:
        case NextPolicyHeadLayout(action_size=action_size):
            return action_size
        case RemainingGameLengthHeadLayout(output_size=output_size):
            return output_size
        case FutureSearchValueHeadLayout(output_size=output_size):
            return output_size
        case IrreversibleProgressHeadLayout(output_size=output_size):
            return output_size
        case LegalMovesHeadLayout(action_size=action_size):
            return action_size
        case SearchBudgetHeadLayout(output_size=output_size):
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
            case FutureSearchValueTargetConfiguration(
                ply_offset=ply_offset,
                smooth_l1_beta=smooth_l1_beta,
            ):
                heads.append(
                    FutureSearchValueHeadLayout(
                        kind='future_search_value',
                        ply_offset=ply_offset,
                        smooth_l1_beta=smooth_l1_beta,
                    )
                )
            case IrreversibleProgressTargetConfiguration(horizon_plies=horizon_plies):
                heads.append(
                    IrreversibleProgressHeadLayout(
                        kind='irreversible_progress',
                        horizon_plies=horizon_plies,
                    )
                )
            case LegalMovesTargetConfiguration():
                heads.append(LegalMovesHeadLayout(kind='legal_moves', action_size=action_size))
            case SearchBudgetTargetConfiguration():
                heads.append(SearchBudgetHeadLayout(kind='search_budget'))
    return TrainingTargetLayout(action_size=action_size, wdl_size=3, auxiliary_heads=tuple(heads))

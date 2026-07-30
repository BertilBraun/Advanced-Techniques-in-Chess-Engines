from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from src.az.config.base import FrozenModel

NATIVE_INT32_MIN = -(2**31)
NATIVE_INT32_MAX = 2**31 - 1
MAXIMUM_HISTORY_LENGTH = 1024
GO_PAYLOAD_SCHEMA_VERSION = 1


class DisabledResignation(FrozenModel):
    kind: Literal['disabled']


class EnabledResignation(FrozenModel):
    kind: Literal['enabled']
    minimum_ply: PositiveInt
    value_threshold: float = Field(gt=-1, lt=0)
    consecutive_moves: PositiveInt
    false_positive_rate_limit: float = Field(gt=0, lt=1)


ResignationConfiguration = Annotated[
    DisabledResignation | EnabledResignation,
    Field(discriminator='kind'),
]


class GoGameConfiguration(FrozenModel):
    kind: Literal['go']
    board_size: Literal[7, 9]
    komi_half_points: int = Field(
        strict=True,
        ge=NATIVE_INT32_MIN,
        le=NATIVE_INT32_MAX,
    )
    scoring_rule: Literal['area']
    ko_rule: Literal['positional_superko']
    suicide_rule: Literal['illegal']
    pass_exempt_from_superko: Literal[True]
    score_comparison: Literal['doubled_integer_points']
    safety_ply_cap: int = Field(strict=True, ge=1, le=NATIVE_INT32_MAX)
    history_length: int = Field(strict=True, ge=1, le=MAXIMUM_HISTORY_LENGTH)
    history_planes_per_position: Literal[2]
    include_color_plane: Literal[True]
    pass_action: Literal['last']
    normal_termination: Literal['two_consecutive_passes']
    symmetry_group: Literal['dihedral_8']
    capped_game_value_target_weight: Literal[0]
    resignation: ResignationConfiguration

    @model_validator(mode='after')
    def validate_go_rules(self) -> GoGameConfiguration:
        if self.safety_ply_cap < self.board_size * self.board_size:
            raise ValueError('The safety ply cap must permit at least one move per board point.')
        return self

    @property
    def action_count(self) -> int:
        return self.board_size * self.board_size + 1

    @property
    def input_plane_count(self) -> int:
        return self.history_length * self.history_planes_per_position + 1


class ResidualGoModelConfiguration(FrozenModel):
    family: Literal['residual_go']
    channels: PositiveInt
    residual_blocks: PositiveInt
    policy_channels: PositiveInt
    value_hidden_size: PositiveInt
    normalization: Literal['batch']
    activation: Literal['relu']


class GoObjectiveConfiguration(FrozenModel):
    kind: Literal['go_policy_value']
    policy_loss_weight: PositiveFloat
    value_loss_weight: PositiveFloat
    l2_regularization_weight: float = Field(ge=0)


class RandomGoOpponent(FrozenModel):
    kind: Literal['random']


class CheckpointGoOpponent(FrozenModel):
    kind: Literal['checkpoint']
    checkpoint_path: PurePosixPath


GoOpponentConfiguration = Annotated[RandomGoOpponent | CheckpointGoOpponent, Field(discriminator='kind')]


class GoEvaluationSuite(FrozenModel):
    kind: Literal['go_paired']
    opponent: GoOpponentConfiguration
    alternate_colors: Literal[True]
    komi_half_points: int = Field(
        strict=True,
        ge=NATIVE_INT32_MIN,
        le=NATIVE_INT32_MAX,
    )


def validate_go_experiment_compatibility(
    objective: GoObjectiveConfiguration,
    optimizer_weight_decay: float,
    replay_payload_schema_version: int,
) -> None:
    if objective.l2_regularization_weight > 0 and optimizer_weight_decay > 0:
        raise ValueError(
            'Objective L2 and optimizer weight decay are intentionally mutually exclusive experimental choices.'
        )
    if replay_payload_schema_version != GO_PAYLOAD_SCHEMA_VERSION:
        raise ValueError(f'Go replay payload schema must be {GO_PAYLOAD_SCHEMA_VERSION}.')

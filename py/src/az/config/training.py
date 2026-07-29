from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from src.az.config.base import FrozenModel
from src.az.games.go.configuration import GoObjectiveConfiguration


class InitialStateOnly(FrozenModel):
    kind: Literal['initial_state_only']


StartStateConfiguration = InitialStateOnly


class SelfPlayConfiguration(FrozenModel):
    start_states: StartStateConfiguration
    concurrent_games_per_worker: PositiveInt
    games_per_shard: PositiveInt
    value_target_weight: PositiveFloat
    capped_game_policy_targets_remain_eligible: Literal[True]
    policy_target_source: Literal['search_budget']


class ReplayCreditConfiguration(FrozenModel):
    target_reuse: PositiveFloat
    optimizer_steps_per_quantum: PositiveInt
    minimum_positions_before_training: PositiveInt


class ReplayConfiguration(FrozenModel):
    capacity_positions: PositiveInt
    shard_directory: PurePosixPath
    maximum_positions_per_shard: PositiveInt
    payload_schema_version: PositiveInt
    compression: Literal['zstd', 'none']
    sampling: Literal['uniform']
    credits: ReplayCreditConfiguration


class AdamWOptimizerConfiguration(FrozenModel):
    kind: Literal['adamw']
    learning_rate: PositiveFloat
    beta_1: float = Field(gt=0, lt=1)
    beta_2: float = Field(gt=0, lt=1)
    epsilon: PositiveFloat
    weight_decay: float = Field(ge=0)


class SgdOptimizerConfiguration(FrozenModel):
    kind: Literal['sgd']
    learning_rate: PositiveFloat
    momentum: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)


OptimizerConfiguration = Annotated[
    AdamWOptimizerConfiguration | SgdOptimizerConfiguration,
    Field(discriminator='kind'),
]


class ConstantLearningRate(FrozenModel):
    kind: Literal['constant']
    multiplier: PositiveFloat


class LearningRateStage(FrozenModel):
    start_optimizer_step: int = Field(ge=0)
    multiplier: PositiveFloat


class PiecewiseLearningRate(FrozenModel):
    kind: Literal['piecewise']
    stages: tuple[LearningRateStage, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_stages(self) -> PiecewiseLearningRate:
        starts = tuple(stage.start_optimizer_step for stage in self.stages)
        if starts[0] != 0 or tuple(sorted(set(starts))) != starts:
            raise ValueError('Learning-rate stages must start at zero and increase strictly.')
        return self


LearningRateConfiguration = Annotated[
    ConstantLearningRate | PiecewiseLearningRate,
    Field(discriminator='kind'),
]


ObjectiveConfiguration = GoObjectiveConfiguration


class TrainingConfiguration(FrozenModel):
    global_batch_size: PositiveInt
    local_batch_size: PositiveInt
    maximum_optimizer_steps: PositiveInt
    optimizer: OptimizerConfiguration
    learning_rate_schedule: LearningRateConfiguration
    precision: Literal['float32', 'bfloat16', 'float16']
    objective: ObjectiveConfiguration
    checkpoint_every_optimizer_steps: PositiveInt
    gradient_clip_norm: PositiveFloat

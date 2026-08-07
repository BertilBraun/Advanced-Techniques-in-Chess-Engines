from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.experiment.base_configuration import BaseExperimentConfiguration
from src.neural_network import NetworkDimensions
from src.training.configuration import SelfPlayConfiguration
from src.util.frozen_model import FrozenModel


class GoRulesConfiguration(FrozenModel):
    scoring: Literal['area'] = 'area'
    komi_half_points: int
    maximum_moves: int = Field(gt=0)


class GoRepresentationConfiguration(FrozenModel):
    board_size: Literal[7, 9]
    history_length: Literal[8] = 8
    binary_channel_count: Literal[16] = 16
    scalar_channel_count: Literal[1] = 1
    action_encoding: Literal['go-point-pass-v1'] = 'go-point-pass-v1'
    canonical_player_perspective: bool = True

    @property
    def channel_count(self) -> int:
        return self.binary_channel_count + self.scalar_channel_count

    @property
    def action_count(self) -> int:
        return self.board_size * self.board_size + 1


class GoTrainingObjectiveConfiguration(FrozenModel):
    policy_loss_weight: float = Field(default=1.0, ge=0.0)
    outcome_value_loss_weight: float = Field(default=1.0, ge=0.0)
    root_value_loss_weight: float = Field(default=0.0, ge=0.0)

    @model_validator(mode='after')
    def validate_value_weights(self) -> GoTrainingObjectiveConfiguration:
        if abs(self.outcome_value_loss_weight + self.root_value_loss_weight - 1.0) > 1e-9:
            raise ValueError('Go value-objective component weights must sum to 1.')
        return self


class GoEvaluationConfiguration(FrozenModel):
    num_searches_per_turn: int = Field(gt=0)
    num_games: int = Field(gt=0)
    every_n_model_versions: int = Field(gt=0)
    max_concurrent_tasks: int = Field(gt=0)
    inference_workers: int = Field(gt=0)
    inference_batch_size: int = Field(gt=0)
    previous_model_offsets: tuple[int, ...]
    historical_model_versions: tuple[int, ...]
    evaluate_random: bool = True


class GoConfiguration(FrozenModel):
    rules: GoRulesConfiguration
    representation: GoRepresentationConfiguration
    self_play: SelfPlayConfiguration
    objective: GoTrainingObjectiveConfiguration = GoTrainingObjectiveConfiguration()
    evaluation: GoEvaluationConfiguration

    @model_validator(mode='after')
    def validate_board_dependent_rules(self) -> GoConfiguration:
        point_count = self.representation.board_size**2
        if self.rules.maximum_moves < point_count * 2:
            raise ValueError('Go maximum moves must be at least twice the board point count.')
        if abs(self.rules.komi_half_points) > point_count * 2:
            raise ValueError('Go komi magnitude cannot exceed the board area.')
        return self


class GoExperimentConfiguration(BaseExperimentConfiguration):
    game: Literal['go'] = 'go'
    go: GoConfiguration

    @property
    def network_dimensions(self) -> NetworkDimensions:
        representation = self.go.representation
        return NetworkDimensions(
            representation.channel_count,
            representation.board_size,
            representation.board_size,
            representation.action_count,
        )

    @model_validator(mode='after')
    def validate_experiment(self) -> GoExperimentConfiguration:
        evaluation = self.go.evaluation
        retention = self.training.lifecycle.inference_retention
        if (
            evaluation.previous_model_offsets
            and max(evaluation.previous_model_offsets) >= retention.recent_checkpoint_count
        ):
            raise ValueError('Recent inference-checkpoint retention must exceed every previous-model offset.')
        if any(
            model_generation % retention.milestone_interval != 0
            for model_generation in evaluation.historical_model_versions
        ):
            raise ValueError('Historical model generations must align with retained milestone checkpoints.')
        return self

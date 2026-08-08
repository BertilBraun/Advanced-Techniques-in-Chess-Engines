from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.experiment.base_configuration import BaseExperimentConfiguration
from src.neural_network import NetworkDimensions
from src.training.configuration import SelfPlayConfiguration, TrainingObjectiveConfiguration
from src.util.frozen_model import FrozenModel


class GoRulesConfiguration(FrozenModel):
    komi_half_points: int
    maximum_moves: int = Field(gt=0)


class GoRepresentationConfiguration(FrozenModel):
    board_size: Literal[7, 9]
    history_length: Literal[8] = 8
    binary_channel_count: Literal[16] = 16
    scalar_channel_count: Literal[1] = 1
    canonical_player_perspective: bool = True

    @property
    def channel_count(self) -> int:
        return self.binary_channel_count + self.scalar_channel_count

    @property
    def action_count(self) -> int:
        return self.board_size * self.board_size + 1


class GoConfiguration(FrozenModel):
    rules: GoRulesConfiguration
    representation: GoRepresentationConfiguration
    self_play: SelfPlayConfiguration
    objective: TrainingObjectiveConfiguration

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
        self.training.validate_game(self.network_dimensions.actions, self.go.self_play)
        if self.evaluation.engine.kind != 'katago':
            raise ValueError('Go evaluation requires the KataGo engine configuration.')
        return self

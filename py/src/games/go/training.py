from __future__ import annotations

from src.games.go.configuration import GoExperimentConfiguration
from src.games.go.contract import GoStateContract, NativeGoPosition
from src.games.go.self_play import (
    GoSelfPlayGame,
    GoSelfPlayPolicy,
    NativeGoSearchRequest,
    NativeGoSearchResult,
)
from src.games.implementation import GameImplementation
from src.neural_network import NetworkDimensions
from src.training.configuration import SelfPlayConfiguration
from src.training.objective import ResolvedTrainingObjective
from src.training.targets import TrainingTargetLayout, build_training_target_layout


class GoImplementation(
    GameImplementation[
        NativeGoPosition,
        GoSelfPlayGame,
        NativeGoSearchRequest,
        NativeGoSearchResult,
        None,
    ]
):
    def __init__(self, configuration: GoExperimentConfiguration) -> None:
        self._configuration = configuration
        self._state = GoStateContract(
            configuration.go.representation.board_size,
            configuration.go.representation.history_length,
            configuration.go.rules.komi_half_points,
            configuration.go.rules.maximum_moves,
        )

    @property
    def configuration(self) -> GoExperimentConfiguration:
        return self._configuration

    @property
    def network_dimensions(self) -> NetworkDimensions:
        return self.configuration.network_dimensions

    @property
    def state(self) -> GoStateContract:
        return self._state

    @property
    def self_play_configuration(self) -> SelfPlayConfiguration:
        return self.configuration.go.self_play

    @property
    def target_layout(self) -> TrainingTargetLayout:
        return build_training_target_layout(
            self.network_dimensions.actions,
            self.configuration.go.objective.auxiliary_targets,
        )

    def training_objective_at(self, model_generation: int) -> ResolvedTrainingObjective:
        configuration = self.configuration.go.objective
        return ResolvedTrainingObjective(
            policy_loss_weight=configuration.policy_loss_weight.value_at(model_generation),
            value_loss_weight=configuration.value_loss_weight.value_at(model_generation),
            root_value_blend=configuration.root_value_blend.value_at(model_generation),
            auxiliary_loss_weights=tuple(
                target.loss_weight.value_at(model_generation) for target in configuration.auxiliary_targets
            ),
        )

    def create_self_play_policy(self, device_id: int, worker_id: int) -> GoSelfPlayPolicy:
        return GoSelfPlayPolicy(self.configuration, worker_id, device_id)

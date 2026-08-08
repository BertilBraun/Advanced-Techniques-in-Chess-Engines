from __future__ import annotations

from typing import TYPE_CHECKING

from src.games.chess.configuration import ChessExperimentConfiguration, ChessSelfPlayConfiguration
from src.games.chess.contract import CHESS_STATE_CONTRACT, ChessPosition, ChessStateContract
from src.games.implementation import GameImplementation
from src.neural_network import NetworkDimensions
from src.self_play.parameters import ResolvedSelfPlayParameters
from src.self_play.worker import NativeSelfPlaySearch
from src.training.checkpoint import CheckpointReference
from src.training.objective import ResolvedTrainingObjective
from src.training.targets import TrainingTargetLayout, build_training_target_layout


if TYPE_CHECKING:
    from AlphaZeroCpp import ChessSelfPlaySearch


class ChessImplementation(GameImplementation[ChessPosition, NativeSelfPlaySearch]):
    def __init__(self, configuration: ChessExperimentConfiguration) -> None:
        self._configuration = configuration

    @property
    def configuration(self) -> ChessExperimentConfiguration:
        return self._configuration

    @property
    def network_dimensions(self) -> NetworkDimensions:
        return self.configuration.network_dimensions

    @property
    def state(self) -> ChessStateContract:
        return CHESS_STATE_CONTRACT

    @property
    def self_play_configuration(self) -> ChessSelfPlayConfiguration:
        return self.configuration.chess.self_play

    @property
    def target_layout(self) -> TrainingTargetLayout:
        return build_training_target_layout(
            self.network_dimensions.actions,
            self.configuration.chess.objective.auxiliary_targets,
        )

    def self_play_parameters_at(self, model_generation: int) -> ResolvedSelfPlayParameters:
        configuration = self.self_play_configuration
        maximum_game_plies = (
            None
            if configuration.maximum_game_plies is None
            else configuration.maximum_game_plies.value_at(model_generation)
        )
        return configuration.resolve(model_generation, maximum_game_plies)

    def create_native_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        parameters: ResolvedSelfPlayParameters,
    ) -> ChessSelfPlaySearch:
        from AlphaZeroCpp import (
            BatchedInferenceParameters,
            ChessSelfPlaySearch,
            InferenceConfiguration,
            InferenceDevice,
        )

        inference = self.self_play_configuration.inference
        device = InferenceDevice.CPU if self.training.topology.trainer.device_type == 'cpu' else InferenceDevice.CUDA
        return ChessSelfPlaySearch(
            InferenceConfiguration(device_id, str(checkpoint.inference_model_path), device),
            self.native_search_parameters(parameters),
            BatchedInferenceParameters(
                inference.inference_workers,
                inference.inference_batch_size,
                inference.outstanding_batches_per_worker,
            ),
            checkpoint.generation,
        )

    def training_objective_at(self, model_generation: int) -> ResolvedTrainingObjective:
        configuration = self.configuration.chess.objective
        return ResolvedTrainingObjective(
            policy_loss_weight=configuration.policy_loss_weight.value_at(model_generation),
            value_loss_weight=configuration.value_loss_weight.value_at(model_generation),
            root_value_blend=configuration.root_value_blend.value_at(model_generation),
            auxiliary_loss_weights=tuple(
                target.loss_weight.value_at(model_generation) for target in configuration.auxiliary_targets
            ),
        )

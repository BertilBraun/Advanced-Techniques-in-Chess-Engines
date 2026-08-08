from __future__ import annotations

from src.games.go.configuration import GoExperimentConfiguration
from src.games.go.contract import GoStateContract, NativeGoPosition
from src.games.implementation import GameImplementation
from src.neural_network import NetworkDimensions
from src.self_play.parameters import ResolvedSelfPlayParameters
from src.self_play.worker import NativeSelfPlaySearch
from src.training.checkpoint import CheckpointReference
from src.training.configuration import SelfPlayConfiguration
from src.training.objective import ResolvedTrainingObjective
from src.training.targets import TrainingTargetLayout, build_training_target_layout


class GoImplementation(GameImplementation[NativeGoPosition, NativeSelfPlaySearch]):
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

    def self_play_parameters_at(self, model_generation: int) -> ResolvedSelfPlayParameters:
        return self.self_play_configuration.resolve(
            model_generation,
            self.configuration.go.rules.maximum_moves,
        )

    def create_native_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        parameters: ResolvedSelfPlayParameters,
    ) -> NativeSelfPlaySearch:
        from AlphaZeroCpp import (
            BatchedInferenceParameters,
            GoSelfPlaySearch7,
            GoSelfPlaySearch9,
            InferenceConfiguration,
            InferenceDevice,
        )

        search_type = GoSelfPlaySearch7 if self.state.board_size == 7 else GoSelfPlaySearch9
        dimensions = search_type.inference_dimensions()
        expected = self.network_dimensions
        if (dimensions.channels, dimensions.rows, dimensions.columns, dimensions.actions, dimensions.outcomes) != (
            expected.channels,
            expected.rows,
            expected.columns,
            expected.actions,
            expected.outcomes,
        ):
            raise ValueError('Resolved Go representation disagrees with the native template dimensions.')
        inference = self.self_play_configuration.inference
        device = InferenceDevice.CPU if self.training.topology.trainer.device_type == 'cpu' else InferenceDevice.CUDA
        return search_type(
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
        configuration = self.configuration.go.objective
        return ResolvedTrainingObjective(
            policy_loss_weight=configuration.policy_loss_weight.value_at(model_generation),
            value_loss_weight=configuration.value_loss_weight.value_at(model_generation),
            root_value_blend=configuration.root_value_blend.value_at(model_generation),
            auxiliary_loss_weights=tuple(
                target.loss_weight.value_at(model_generation) for target in configuration.auxiliary_targets
            ),
        )

from __future__ import annotations

from dataclasses import replace

from src.evaluation.configuration import EvaluationSearchConfiguration
from src.experiment.generation_schedule import FloatGenerationSchedule
from src.games.go.configuration import GoExperimentConfiguration
from src.games.go.contract import GoStateContract, NativeGoPosition
from src.games.implementation import GameImplementation
from src.games.representation import NetworkDimensions
from src.self_play.configuration import BatchedInferenceParams, SelfPlayConfiguration
from src.self_play.native_configuration import native_sdpa_backend
from src.self_play.native_search import NativeSelfPlaySearch
from src.self_play.parameters import (
    FixedFullSearchBudget,
    ResolvedSelfPlayParameters,
    ZeroFirstPlayUrgencyParameters,
)
from src.training.checkpoint import CheckpointReference
from src.training.objective import ResolvedTrainingObjective, resolve_auxiliary_losses
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

    @property
    def value_discount_per_ply(self) -> FloatGenerationSchedule:
        return self.configuration.go.objective.value_discount_per_ply

    def self_play_parameters_at(self, model_generation: int) -> ResolvedSelfPlayParameters:
        return self.self_play_configuration.resolve(
            model_generation,
            self.configuration.go.rules.maximum_moves,
            self.value_discount_per_ply.value_at(model_generation),
        )

    def create_native_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        parameters: ResolvedSelfPlayParameters,
    ) -> NativeSelfPlaySearch:
        return self._create_search(device_id, checkpoint, parameters, self.self_play_configuration.inference)

    def create_evaluation_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        configuration: EvaluationSearchConfiguration,
    ) -> NativeSelfPlaySearch:
        parameters = replace(
            self.self_play_parameters_at(checkpoint.generation),
            parallel_searches=configuration.parallel_searches,
            full_search_budget=FixedFullSearchBudget(kind='fixed', visits=configuration.searches_per_move),
            fast_searches=configuration.searches_per_move,
            forced_playout_coefficient=0.0,
            exploration_constant=configuration.exploration_constant,
            first_play_urgency=ZeroFirstPlayUrgencyParameters(),
            dirichlet_alpha=1.0,
            dirichlet_epsilon=0.0,
        )
        return self._create_search(device_id, checkpoint, parameters, configuration.inference)

    def _create_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        parameters: ResolvedSelfPlayParameters,
        inference: BatchedInferenceParams,
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
        device = InferenceDevice.CPU if self.training.topology.trainer.device_type == 'cpu' else InferenceDevice.CUDA
        return search_type(
            InferenceConfiguration(
                device_id,
                str(checkpoint.inference_model_path),
                device,
                native_sdpa_backend(inference.sdpa_backend),
            ),
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
            auxiliary_losses=resolve_auxiliary_losses(configuration.auxiliary_targets, model_generation),
        )

from __future__ import annotations

from src.evaluation.configuration import EvaluationSearchConfiguration, EvaluationTreeSearchOverrides
from src.games.go.configuration import GoExperimentConfiguration
from src.games.go.contract import GoStateContract, NativeGoPosition
from src.games.implementation import GameImplementation, resolved_evaluation_parameters
from src.games.representation import NetworkDimensions
from src.search_budget.policy import SearchBudgetPolicy, disabled_policy
from src.self_play.configuration import BatchedInferenceParams, SelfPlayConfiguration
from src.self_play.native_search import NativeSelfPlaySearch
from src.self_play.parameters import (
    ResolvedSelfPlayParameters,
)
from src.training.checkpoint import CheckpointReference
from src.training.objective import ResolvedTrainingObjective, resolve_auxiliary_losses
from src.training.targets import TrainingTargetLayout, build_training_target_layout
from src.util.generation_schedule import FloatGenerationSchedule


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

    def self_play_parameters_at(
        self,
        model_generation: int,
        search_budget_policy: SearchBudgetPolicy,
    ) -> ResolvedSelfPlayParameters:
        objective = self.configuration.go.objective
        return self.self_play_configuration.resolve(
            model_generation,
            search_budget_policy,
            self.configuration.go.rules.maximum_moves,
            objective.effective_search_value_discount_per_ply.value_at(model_generation),
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
        tree_search: EvaluationTreeSearchOverrides | None = None,
    ) -> NativeSelfPlaySearch:
        parameters = self.evaluation_parameters_at(checkpoint.generation, configuration, tree_search)
        return self._create_search(device_id, checkpoint, parameters, configuration.inference)

    def evaluation_parameters_at(
        self,
        model_generation: int,
        configuration: EvaluationSearchConfiguration,
        tree_search: EvaluationTreeSearchOverrides | None = None,
    ) -> ResolvedSelfPlayParameters:
        """Evaluation inherits the self-play first-play urgency; only the listed fields are overridden."""
        return resolved_evaluation_parameters(
            self.self_play_parameters_at(model_generation, disabled_policy()),
            configuration,
            model_generation,
            tree_search,
        )

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
        )

        search_type = GoSelfPlaySearch7 if self.state.board_size == 7 else GoSelfPlaySearch9
        self.validate_native_dimensions(search_type.inference_dimensions())
        return search_type(
            self.native_inference_configuration(device_id, checkpoint.inference_model_path, inference),
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
            auxiliary_losses=resolve_auxiliary_losses(
                configuration.auxiliary_targets,
                model_generation,
                self.configuration.training.lifecycle.search_budget.head_training.dedicated_batches,
            ),
        )

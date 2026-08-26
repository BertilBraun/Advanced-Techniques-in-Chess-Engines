from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from src.evaluation.configuration import EvaluationSearchConfiguration, EvaluationTreeSearchOverrides
from src.games.chess.configuration import ChessExperimentConfiguration, ChessSelfPlayConfiguration
from src.games.chess.contract import CHESS_STATE_CONTRACT, ChessPosition, ChessStateContract
from src.games.implementation import GameImplementation, resolved_evaluation_parameters
from src.games.representation import NetworkDimensions
from src.self_play.configuration import BatchedInferenceParams
from src.self_play.native_search import NativeSelfPlaySearch
from src.self_play.parameters import (
    ResolvedSelfPlayParameters,
)
from src.self_play.resignation import CalibratedResignationConfiguration
from src.training.checkpoint import CheckpointReference
from src.training.objective import ResolvedTrainingObjective, resolve_auxiliary_losses
from src.training.targets import TrainingTargetLayout, build_training_target_layout
from src.util.generation_schedule import FloatGenerationSchedule

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
    def terminal_oracle(self) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def self_play_configuration(self) -> ChessSelfPlayConfiguration:
        return self.configuration.chess.self_play

    @property
    def target_layout(self) -> TrainingTargetLayout:
        return build_training_target_layout(
            self.network_dimensions.actions,
            self.configuration.chess.objective.auxiliary_targets,
        )

    @property
    def value_discount_per_ply(self) -> FloatGenerationSchedule:
        return self.configuration.chess.objective.value_discount_per_ply

    def self_play_parameters_at(self, model_generation: int) -> ResolvedSelfPlayParameters:
        configuration = self.self_play_configuration
        objective = self.configuration.chess.objective
        early_termination = configuration.early_termination
        return replace(
            configuration.resolve(
                model_generation,
                configuration.maximum_game_plies_at(model_generation),
                objective.effective_search_value_discount_per_ply.value_at(model_generation),
            ),
            bootstrap_cut_game_value=(
                early_termination is not None and early_termination.value_target == 'search_root_value'
            ),
        )

    @property
    def censor_remaining_game_length_on_cut_games(self) -> bool:
        early_termination = self.self_play_configuration.early_termination
        return early_termination is not None and early_termination.censor_remaining_game_length_target

    @property
    def resignation_configuration(self) -> CalibratedResignationConfiguration | None:
        configuration = self.self_play_configuration.resignation
        match configuration:
            case CalibratedResignationConfiguration():
                return configuration
            case _:
                return None

    def create_native_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        parameters: ResolvedSelfPlayParameters,
    ) -> ChessSelfPlaySearch:
        return self._create_search(device_id, checkpoint, parameters, self.self_play_configuration.inference)

    def create_evaluation_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        configuration: EvaluationSearchConfiguration,
        tree_search: EvaluationTreeSearchOverrides | None = None,
    ) -> ChessSelfPlaySearch:
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
            self.self_play_parameters_at(model_generation),
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
    ) -> ChessSelfPlaySearch:
        from AlphaZeroCpp import (
            BatchedInferenceParameters,
            ChessSelfPlaySearch,
        )

        self.validate_native_dimensions(ChessSelfPlaySearch.inference_dimensions())
        return ChessSelfPlaySearch(
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
        configuration = self.configuration.chess.objective
        return ResolvedTrainingObjective(
            policy_loss_weight=configuration.policy_loss_weight.value_at(model_generation),
            value_loss_weight=configuration.value_loss_weight.value_at(model_generation),
            root_value_blend=configuration.root_value_blend.value_at(model_generation),
            auxiliary_losses=resolve_auxiliary_losses(configuration.auxiliary_targets, model_generation),
        )

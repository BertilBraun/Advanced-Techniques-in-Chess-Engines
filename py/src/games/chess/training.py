from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from src.evaluation.configuration import EvaluationSearchConfiguration
from src.util.generation_schedule import FloatGenerationSchedule
from src.games.chess.configuration import ChessExperimentConfiguration, ChessSelfPlayConfiguration
from src.games.chess.contract import CHESS_STATE_CONTRACT, ChessPosition, ChessStateContract
from src.games.chess.syzygy import SyzygyTerminalOracle
from src.games.implementation import GameImplementation
from src.games.representation import NetworkDimensions
from src.self_play.configuration import BatchedInferenceParams
from src.self_play.native_search import NativeSelfPlaySearch
from src.self_play.parameters import (
    FixedFullSearchBudget,
    ResolvedSelfPlayParameters,
    ZeroFirstPlayUrgencyParameters,
)
from src.self_play.resignation import CalibratedResignationConfiguration
from src.training.checkpoint import CheckpointReference
from src.training.objective import ResolvedTrainingObjective, resolve_auxiliary_losses
from src.training.targets import TrainingTargetLayout, build_training_target_layout

if TYPE_CHECKING:
    from AlphaZeroCpp import ChessSelfPlaySearch


class ChessImplementation(GameImplementation[ChessPosition, NativeSelfPlaySearch]):
    def __init__(self, configuration: ChessExperimentConfiguration) -> None:
        self._configuration = configuration
        self._terminal_oracle: SyzygyTerminalOracle | None = None

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
    def terminal_oracle(self) -> SyzygyTerminalOracle | None:
        paths = self.self_play_configuration.maximum_ply_syzygy_paths
        if paths is None:
            return None
        if self._terminal_oracle is None:
            self._terminal_oracle = SyzygyTerminalOracle(paths)
        return self._terminal_oracle

    def close(self) -> None:
        if self._terminal_oracle is not None:
            self._terminal_oracle.close()
            self._terminal_oracle = None

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
        return configuration.resolve(
            model_generation,
            configuration.maximum_game_plies_at(model_generation),
            self.value_discount_per_ply.value_at(model_generation),
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
    ) -> ChessSelfPlaySearch:
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

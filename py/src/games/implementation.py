from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from src.experiment.configuration import ExperimentConfiguration
from src.games.contracts import GameStateContract, TerminalOracle
from src.games.representation import NetworkDimensions
from src.self_play.configuration import BatchedInferenceParams, SelfPlayConfiguration
from src.self_play.native_configuration import native_execution_options
from src.self_play.parameters import (
    ParentValueFirstPlayUrgencyParameters,
    ReducedParentValueFirstPlayUrgencyParameters,
    ResolvedSelfPlayParameters,
    ZeroFirstPlayUrgencyParameters,
)
from src.training.configuration import TrainingArgs
from src.training.objective import ResolvedTrainingObjective
from src.training.targets import TrainingTargetLayout
from src.util.generation_schedule import FloatGenerationSchedule

if TYPE_CHECKING:
    from AlphaZeroCpp import InferenceConfiguration, SelfPlaySearchParameters
    from src.evaluation.configuration import EvaluationSearchConfiguration, EvaluationTreeSearchOverrides
    from src.self_play.native_search import NativeSelfPlaySearch
    from src.self_play.resignation import CalibratedResignationConfiguration
    from src.training.checkpoint import CheckpointReference


def resolved_evaluation_parameters(
    baseline: ResolvedSelfPlayParameters,
    configuration: EvaluationSearchConfiguration,
    model_generation: int,
    overrides: EvaluationTreeSearchOverrides | None,
) -> ResolvedSelfPlayParameters:
    parameters = replace(
        baseline,
        baseline_visits=configuration.searches_per_move,
        search_budget_blend=0.0,
        forced_playout_coefficient=0.0,
        exploration_constant=configuration.resolved_exploration_constant,
        first_play_urgency=(
            baseline.first_play_urgency if overrides is None else overrides.first_play_urgency.resolve(model_generation)
        ),
        dirichlet_alpha=1.0,
        dirichlet_epsilon=0.0,
    )
    if overrides is None:
        return parameters
    return replace(
        parameters,
        virtual_loss_weight=overrides.virtual_loss_weight,
        value_discount_per_ply=overrides.value_discount_per_ply,
    )


PositionT = TypeVar('PositionT')
NativeSearchT = TypeVar('NativeSearchT', bound='NativeSelfPlaySearch')


class GameImplementation(ABC, Generic[PositionT, NativeSearchT]):
    @property
    @abstractmethod
    def configuration(self) -> ExperimentConfiguration:
        raise NotImplementedError

    @property
    def training(self) -> TrainingArgs:
        return self.configuration.training

    @property
    @abstractmethod
    def network_dimensions(self) -> NetworkDimensions:
        raise NotImplementedError

    @property
    @abstractmethod
    def state(self) -> GameStateContract[PositionT]:
        raise NotImplementedError

    @property
    def terminal_oracle(self) -> TerminalOracle[PositionT] | None:
        return None

    def validate_native_dimensions(self, native_dimensions: object) -> None:
        expected = self.network_dimensions
        actual = (
            native_dimensions.channels,  # type: ignore[attr-defined]
            native_dimensions.rows,  # type: ignore[attr-defined]
            native_dimensions.columns,  # type: ignore[attr-defined]
            native_dimensions.actions,  # type: ignore[attr-defined]
            native_dimensions.outcomes,  # type: ignore[attr-defined]
        )
        if actual != (expected.channels, expected.rows, expected.columns, expected.actions, expected.outcomes):
            raise ValueError(
                f'Resolved {self.state.name} representation {expected} disagrees with the native template {actual}.'
            )

    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def self_play_configuration(self) -> SelfPlayConfiguration:
        raise NotImplementedError

    @property
    @abstractmethod
    def target_layout(self) -> TrainingTargetLayout:
        raise NotImplementedError

    @property
    @abstractmethod
    def value_discount_per_ply(self) -> FloatGenerationSchedule:
        raise NotImplementedError

    @abstractmethod
    def self_play_parameters_at(
        self,
        model_generation: int,
        search_budget_blend: float,
    ) -> ResolvedSelfPlayParameters:
        raise NotImplementedError

    @property
    def resignation_configuration(self) -> CalibratedResignationConfiguration | None:
        return None

    @property
    def censor_remaining_game_length_on_cut_games(self) -> bool:
        return False

    def native_inference_configuration(
        self,
        device_id: int,
        model_path: Path,
        inference: BatchedInferenceParams | None = None,
    ) -> InferenceConfiguration:
        from AlphaZeroCpp import InferenceConfiguration, InferenceDevice

        # Defaults to the self-play inference parameters; evaluation passes its own so backends can differ.
        effective = self.self_play_configuration.inference if inference is None else inference
        return InferenceConfiguration(
            device_id=device_id,
            model_path=str(model_path),
            device=InferenceDevice.CPU if self.training.topology.trainer.device_type == 'cpu' else InferenceDevice.CUDA,
            execution_options=native_execution_options(effective),
        )

    def native_search_parameters(self, parameters: ResolvedSelfPlayParameters) -> SelfPlaySearchParameters:
        from AlphaZeroCpp import (
            FirstPlayUrgencyKind,
            FirstPlayUrgencyParameters,
            SelfPlaySearchParameters,
            TreeSearchParameters,
        )

        match parameters.first_play_urgency:
            case ZeroFirstPlayUrgencyParameters():
                first_play_urgency = FirstPlayUrgencyParameters(FirstPlayUrgencyKind.ZERO)
            case ParentValueFirstPlayUrgencyParameters():
                first_play_urgency = FirstPlayUrgencyParameters(FirstPlayUrgencyKind.PARENT_VALUE)
            case ReducedParentValueFirstPlayUrgencyParameters(reduction=reduction):
                first_play_urgency = FirstPlayUrgencyParameters(
                    FirstPlayUrgencyKind.REDUCED_PARENT_VALUE,
                    reduction,
                )

        return SelfPlaySearchParameters(
            baseline_visits=parameters.baseline_visits,
            search_budget_blend=parameters.search_budget_blend,
            tree_search=TreeSearchParameters(
                exploration_constant=parameters.exploration_constant,
                first_play_urgency=first_play_urgency,
                forced_playout_coefficient=parameters.forced_playout_coefficient,
                value_discount_per_ply=parameters.value_discount_per_ply,
                virtual_loss_weight=parameters.virtual_loss_weight,
            ),
            dirichlet_alpha=parameters.dirichlet_alpha,
            dirichlet_epsilon=parameters.dirichlet_epsilon,
        )

    @abstractmethod
    def create_native_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        parameters: ResolvedSelfPlayParameters,
    ) -> NativeSearchT:
        raise NotImplementedError

    @abstractmethod
    def create_evaluation_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        configuration: EvaluationSearchConfiguration,
        tree_search: EvaluationTreeSearchOverrides | None = None,
    ) -> NativeSearchT:
        raise NotImplementedError

    @abstractmethod
    def training_objective_at(self, model_generation: int) -> ResolvedTrainingObjective:
        raise NotImplementedError

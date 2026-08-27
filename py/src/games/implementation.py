from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from src.experiment.configuration import ExperimentConfiguration
from src.games.contracts import GameStateContract, TerminalOracle
from src.games.representation import NetworkDimensions
from src.self_play.configuration import BatchedInferenceParams, SelfPlayConfiguration
from src.self_play.native_configuration import native_execution_options
from src.self_play.parameters import (
    AdaptiveFullSearchBudget,
    FixedFullSearchBudget,
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
    from src.evaluation.configuration import EvaluationSearchConfiguration
    from src.self_play.native_search import NativeSelfPlaySearch
    from src.self_play.resignation import CalibratedResignationConfiguration
    from src.training.checkpoint import CheckpointReference


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
    def self_play_parameters_at(self, model_generation: int) -> ResolvedSelfPlayParameters:
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
            AdaptiveSearchLimit,
            DisabledSearchCorrectionGate,
            FirstPlayUrgencyKind,
            FirstPlayUrgencyParameters,
            FixedSearchLimit,
            SearchCorrectionGate,
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

        match parameters.full_search_budget:
            case FixedFullSearchBudget(visits=visits):
                full_search_budget = FixedSearchLimit(visits)
            case AdaptiveFullSearchBudget() as adaptive:
                gate_visit = adaptive.minimum_visits + (adaptive.maximum_visits - adaptive.minimum_visits) // 2
                gate = (
                    DisabledSearchCorrectionGate()
                    if adaptive.minimum_search_correction_to_unlock_tail is None
                    else SearchCorrectionGate(
                        gate_visit,
                        adaptive.minimum_search_correction_to_unlock_tail,
                    )
                )
                full_search_budget = AdaptiveSearchLimit(
                    minimum_visits=adaptive.minimum_visits,
                    maximum_visits=adaptive.maximum_visits,
                    observation_interval=adaptive.observation_interval,
                    leader_stability_window=adaptive.leader_stability_window,
                    root_value_tolerance=adaptive.root_value_tolerance,
                    initial_top_visit_share=adaptive.initial_top_visit_share,
                    final_top_visit_share=adaptive.final_top_visit_share,
                    initial_top_two_margin=adaptive.initial_top_two_margin,
                    final_top_two_margin=adaptive.final_top_two_margin,
                    threshold_relaxation_visits=adaptive.threshold_relaxation_visits,
                    search_correction_gate=gate,
                )

        return SelfPlaySearchParameters(
            parallel_searches=parameters.parallel_searches,
            full_search_budget=full_search_budget,
            fast_searches=parameters.fast_searches,
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
    ) -> NativeSearchT:
        raise NotImplementedError

    @abstractmethod
    def training_objective_at(self, model_generation: int) -> ResolvedTrainingObjective:
        raise NotImplementedError

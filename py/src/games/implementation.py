from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from src.experiment.configuration import ExperimentConfiguration
from src.experiment.generation_schedule import FloatGenerationSchedule
from src.games.contracts import GameStateContract, TerminalOracle
from src.games.representation import NetworkDimensions
from src.self_play.configuration import SelfPlayConfiguration
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

if TYPE_CHECKING:
    from AlphaZeroCpp import SelfPlaySearchParameters
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

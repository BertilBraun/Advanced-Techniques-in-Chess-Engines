from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from src.experiment.configuration import ExperimentConfiguration
from src.games.contracts import GameStateContract
from src.games.representation import NetworkDimensions
from src.self_play.configuration import SelfPlayConfiguration
from src.self_play.parameters import (
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

    from src.self_play.native_search import NativeSelfPlaySearch
    from src.evaluation.configuration import EvaluationSearchConfiguration
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
    @abstractmethod
    def self_play_configuration(self) -> SelfPlayConfiguration:
        raise NotImplementedError

    @property
    @abstractmethod
    def target_layout(self) -> TrainingTargetLayout:
        raise NotImplementedError

    @abstractmethod
    def self_play_parameters_at(self, model_generation: int) -> ResolvedSelfPlayParameters:
        raise NotImplementedError

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
            parallel_searches=parameters.parallel_searches,
            full_searches=parameters.full_searches,
            fast_searches=parameters.fast_searches,
            tree_search=TreeSearchParameters(
                exploration_constant=parameters.exploration_constant,
                first_play_urgency=first_play_urgency,
                forced_playout_coefficient=parameters.forced_playout_coefficient,
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

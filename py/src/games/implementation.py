from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.experiment.configuration import ExperimentConfiguration
from src.games.contracts import GameStateContract
from src.neural_network import NetworkDimensions
from src.self_play.worker import GameSelfPlayPolicy
from src.training.configuration import SelfPlayConfiguration, TrainingArgs
from src.training.objective import ResolvedTrainingObjective
from src.training.targets import TrainingTargetLayout


PositionT = TypeVar('PositionT')
ActiveGameT = TypeVar('ActiveGameT')
SearchRequestT = TypeVar('SearchRequestT')
SearchResultT = TypeVar('SearchResultT')
StatisticsT = TypeVar('StatisticsT')


class GameImplementation(
    ABC,
    Generic[PositionT, ActiveGameT, SearchRequestT, SearchResultT, StatisticsT],
):
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
    def training_objective_at(self, model_generation: int) -> ResolvedTrainingObjective:
        raise NotImplementedError

    @abstractmethod
    def create_self_play_policy(
        self,
        device_id: int,
        worker_id: int,
    ) -> GameSelfPlayPolicy[ActiveGameT, SearchRequestT, SearchResultT, StatisticsT]:
        raise NotImplementedError

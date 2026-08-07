from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.experiment.configuration import ExperimentConfiguration
from src.neural_network import NetworkDimensions
from src.self_play.completed_game import CompletedGamePublisher, CompletedGameRecord
from src.self_play.worker import GameSelfPlayPolicy
from src.train.Replay import ReplayGameImplementation
from src.train.Trainer import TrainingObjective
from src.train.TrainingArgs import TrainingArgs


CompletedGameT = TypeVar('CompletedGameT', bound=CompletedGameRecord)
ActiveGameT = TypeVar('ActiveGameT')
SearchRequestT = TypeVar('SearchRequestT')
SearchResultT = TypeVar('SearchResultT')
StatisticsT = TypeVar('StatisticsT')


class GameImplementation(
    ABC,
    Generic[
        CompletedGameT,
        ActiveGameT,
        SearchRequestT,
        SearchResultT,
        StatisticsT,
    ],
):
    """Compose the game-owned components consumed by shared training infrastructure."""

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
    def replay(self) -> ReplayGameImplementation[CompletedGameT]:
        raise NotImplementedError

    @abstractmethod
    def objective(self, optimizer_step: int) -> TrainingObjective:
        raise NotImplementedError

    @abstractmethod
    def create_self_play_policy(
        self,
        device_id: int,
        publisher: CompletedGamePublisher,
    ) -> GameSelfPlayPolicy[
        ActiveGameT,
        SearchRequestT,
        SearchResultT,
        StatisticsT,
    ]:
        raise NotImplementedError

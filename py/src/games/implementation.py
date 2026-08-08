from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.experiment.configuration import ExperimentConfiguration
from src.games.contracts import GameStateContract
from src.neural_network import NetworkDimensions
from src.self_play.completed_game import RuntimeCompletedGameRecord
from src.self_play.worker import GameSelfPlayPolicy
from src.training.configuration import SelfPlayConfiguration, TrainingArgs
from src.training.replay import ReplayGameImplementation
from src.training.targets import TrainingTargetLayout
from src.training.objective import ResolvedTrainingObjective
from src.training.trainer import RuntimeTrainingObjective


PositionT = TypeVar('PositionT')
CompletedGameT = TypeVar('CompletedGameT', bound=RuntimeCompletedGameRecord)
ActiveGameT = TypeVar('ActiveGameT')
SearchRequestT = TypeVar('SearchRequestT')
SearchResultT = TypeVar('SearchResultT')
StatisticsT = TypeVar('StatisticsT')


class GameImplementation(
    ABC,
    Generic[
        PositionT,
        CompletedGameT,
        ActiveGameT,
        SearchRequestT,
        SearchResultT,
        StatisticsT,
    ],
):
    """Root composition for the selected concrete game."""

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
    def runtime_training_objective_at(self, model_generation: int) -> RuntimeTrainingObjective:
        """Transitional objective consumed only by the pre-Phase-2 trainer."""
        raise NotImplementedError

    @property
    @abstractmethod
    def replay(self) -> ReplayGameImplementation[CompletedGameT]:
        """Current in-memory runtime boundary, replaced as one unit in Phase 2."""
        raise NotImplementedError

    @abstractmethod
    def create_self_play_policy(
        self,
        device_id: int,
        worker_id: int,
    ) -> GameSelfPlayPolicy[ActiveGameT, SearchRequestT, SearchResultT, StatisticsT]:
        """Current worker boundary, replaced as one unit in Phase 2."""
        raise NotImplementedError

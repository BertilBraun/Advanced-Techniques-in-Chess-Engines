from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from src.neural_network import NetworkDimensions
from src.experiment.configuration import ExperimentConfiguration
from src.games.chess.completed_game import ChessCompletedGame
from src.self_play.completed_game import CompletedGamePublisher
from src.games.go.completed_game import GoCompletedGame
from src.train.Replay import ReplayGameImplementation
from src.train.Trainer import TrainingObjective
from src.train.TrainingArgs import TrainingArgs


class SelfPlayWorker(Protocol):
    def run_batch(self) -> None: ...

    def refresh_published_model(self, model_generation: int, model_path: Path) -> None: ...

    def snapshot_statistics(self, tensorboard_step: int) -> object: ...


class TrainingGameImplementation(ABC):
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
    def replay(
        self,
    ) -> ReplayGameImplementation[ChessCompletedGame] | ReplayGameImplementation[GoCompletedGame]:
        raise NotImplementedError

    @abstractmethod
    def objective(self, optimizer_step: int) -> TrainingObjective:
        raise NotImplementedError

    @abstractmethod
    def create_self_play(
        self,
        device_id: int,
        model_generation: int,
        model_path: Path,
        publisher: CompletedGamePublisher,
    ) -> SelfPlayWorker:
        raise NotImplementedError

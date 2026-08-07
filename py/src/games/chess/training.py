from pathlib import Path

from src.neural_network import NetworkDimensions
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.training_contract import SelfPlayWorker, TrainingGameImplementation
from src.games.chess.self_play import SelfPlay
from src.games.chess.completed_game import ChessCompletedGame
from src.self_play.completed_game import CompletedGamePublisher
from src.games.chess.replay import CHESS_REPLAY_IMPLEMENTATION
from src.train.Replay import ReplayGameImplementation
from src.train.Trainer import ChessTrainingObjective, TrainingObjective


class ChessTrainingGame(TrainingGameImplementation):
    def __init__(self, configuration: ChessExperimentConfiguration) -> None:
        self._configuration = configuration

    @property
    def configuration(self) -> ChessExperimentConfiguration:
        return self._configuration

    @property
    def network_dimensions(self) -> NetworkDimensions:
        return self.configuration.network_dimensions

    @property
    def replay(self) -> ReplayGameImplementation[ChessCompletedGame]:
        return CHESS_REPLAY_IMPLEMENTATION

    def objective(self, optimizer_step: int) -> TrainingObjective:
        return ChessTrainingObjective(self.training.trainer, optimizer_step)

    def create_self_play(
        self,
        device_id: int,
        model_generation: int,
        model_path: Path,
        publisher: CompletedGamePublisher,
    ) -> SelfPlayWorker:
        return SelfPlay(device_id, self.training, publisher)

from pathlib import Path

from src.Network import NetworkDimensions
from src.experiment.chess_experiment import GoExperimentConfiguration
from src.games.go.contract import GoStateContract
from src.games.training_contract import SelfPlayWorker, TrainingGameImplementation
from src.self_play.GoSelfPlay import GoSelfPlay
from src.self_play.completed_game import CompletedGamePublisher
from src.self_play.go_completed_game import GoCompletedGame
from src.train.GoReplay import GoReplayImplementation
from src.train.Replay import ReplayGameImplementation
from src.train.Trainer import GoTrainingObjective, TrainingObjective


class GoTrainingGame(TrainingGameImplementation):
    def __init__(self, configuration: GoExperimentConfiguration) -> None:
        self._configuration = configuration
        self.state = GoStateContract(
            configuration.go.representation.board_size,
            configuration.go.representation.history_length,
        )
        self._replay = GoReplayImplementation(self.state)

    @property
    def configuration(self) -> GoExperimentConfiguration:
        return self._configuration

    @property
    def network_dimensions(self) -> NetworkDimensions:
        return self.state.network_dimensions

    @property
    def replay(self) -> ReplayGameImplementation[GoCompletedGame]:
        return self._replay

    def objective(self, optimizer_step: int) -> TrainingObjective:
        return GoTrainingObjective(self.configuration.go.objective)

    def create_self_play(
        self,
        device_id: int,
        model_generation: int,
        model_path: Path,
        publisher: CompletedGamePublisher,
    ) -> SelfPlayWorker:
        return GoSelfPlay(
            self.configuration,
            model_path,
            model_generation,
            publisher,
            device_id,
        )

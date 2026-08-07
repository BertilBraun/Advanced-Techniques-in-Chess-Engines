from src.neural_network import NetworkDimensions
from src.games.go.configuration import GoExperimentConfiguration
from src.games.go.contract import GoStateContract
from src.games.training_contract import GameImplementation
from src.games.go.self_play import (
    GoSelfPlayGame,
    GoSelfPlayPolicy,
    NativeGoSearchRequest,
    NativeGoSearchResult,
)
from src.self_play.completed_game import CompletedGamePublisher
from src.games.go.completed_game import GoCompletedGame
from src.games.go.replay import GoReplayImplementation
from src.train.Replay import ReplayGameImplementation
from src.train.Trainer import GoTrainingObjective, TrainingObjective


class GoImplementation(
    GameImplementation[
        GoCompletedGame,
        GoSelfPlayGame,
        NativeGoSearchRequest,
        NativeGoSearchResult,
        None,
    ]
):
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
        return self.configuration.network_dimensions

    @property
    def replay(self) -> ReplayGameImplementation[GoCompletedGame]:
        return self._replay

    def objective(self, optimizer_step: int) -> TrainingObjective:
        return GoTrainingObjective(self.configuration.go.objective)

    def create_self_play_policy(
        self,
        device_id: int,
        publisher: CompletedGamePublisher,
    ) -> GoSelfPlayPolicy:
        return GoSelfPlayPolicy(
            self.configuration,
            publisher,
            device_id,
        )

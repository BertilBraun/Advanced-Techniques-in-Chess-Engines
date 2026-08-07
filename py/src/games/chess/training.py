from src.neural_network import NetworkDimensions
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.training_contract import GameImplementation
from src.games.chess.self_play import (
    ChessSelfPlayPolicy,
    SelfPlayGame,
    SelfPlayStatisticsSnapshot,
)
from src.games.chess.completed_game import ChessCompletedGame
from src.self_play.completed_game import CompletedGamePublisher
from src.games.chess.replay import CHESS_REPLAY_IMPLEMENTATION
from src.training.replay import ReplayGameImplementation
from src.training.trainer import ChessTrainingObjective, TrainingObjective


class ChessImplementation(
    GameImplementation[
        ChessCompletedGame,
        SelfPlayGame,
        'ChessSelfPlaySearchRequest',
        'ChessSelfPlaySearchResult',
        SelfPlayStatisticsSnapshot | None,
    ]
):
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

    def create_self_play_policy(
        self,
        device_id: int,
        publisher: CompletedGamePublisher,
    ) -> ChessSelfPlayPolicy:
        return ChessSelfPlayPolicy(device_id, self.training, publisher)

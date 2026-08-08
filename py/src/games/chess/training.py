from src.games.chess.configuration import ChessExperimentConfiguration, ChessSelfPlayConfiguration
from src.games.chess.contract import CHESS_STATE_CONTRACT, ChessPosition, ChessStateContract
from src.games.chess.self_play import ChessSelfPlayGame, ChessSelfPlayPolicy, SelfPlayStatisticsSnapshot
from src.games.implementation import GameImplementation
from src.neural_network import NetworkDimensions
from src.training.objective import ResolvedTrainingObjective
from src.training.targets import TrainingTargetLayout, build_training_target_layout


class ChessImplementation(
    GameImplementation[
        ChessPosition,
        ChessSelfPlayGame,
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
    def state(self) -> ChessStateContract:
        return CHESS_STATE_CONTRACT

    @property
    def self_play_configuration(self) -> ChessSelfPlayConfiguration:
        return self.configuration.chess.self_play

    @property
    def target_layout(self) -> TrainingTargetLayout:
        return build_training_target_layout(
            self.network_dimensions.actions,
            self.configuration.chess.objective.auxiliary_targets,
        )

    def training_objective_at(self, model_generation: int) -> ResolvedTrainingObjective:
        configuration = self.configuration.chess.objective
        return ResolvedTrainingObjective(
            policy_loss_weight=configuration.policy_loss_weight.value_at(model_generation),
            value_loss_weight=configuration.value_loss_weight.value_at(model_generation),
            root_value_blend=configuration.root_value_blend.value_at(model_generation),
            auxiliary_loss_weights=tuple(
                target.loss_weight.value_at(model_generation) for target in configuration.auxiliary_targets
            ),
        )

    def create_self_play_policy(self, device_id: int, worker_id: int) -> ChessSelfPlayPolicy:
        return ChessSelfPlayPolicy(
            device_id,
            self.configuration.chess.self_play,
            worker_id,
            self.training.random_seed,
        )

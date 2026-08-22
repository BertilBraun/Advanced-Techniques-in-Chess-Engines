from __future__ import annotations

from src.experiment.configuration import ExperimentConfiguration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.games.go.configuration import GoExperimentConfiguration
from src.games.go.training import GoImplementation

ConfiguredGame = ChessImplementation | GoImplementation


def create_game_implementation(configuration: ExperimentConfiguration) -> ConfiguredGame:
    match configuration:
        case ChessExperimentConfiguration():
            return ChessImplementation(configuration)
        case GoExperimentConfiguration():
            return GoImplementation(configuration)

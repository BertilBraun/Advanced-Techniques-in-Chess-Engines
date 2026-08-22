from __future__ import annotations

from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.training.configuration import TrainingArgs
from test_helpers.configuration_paths import TEST_CONFIG_DIRECTORY

CHESS_EXPERIMENT_TEMPLATE_PATH = TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml'
CHESS_EXPERIMENT: ChessExperimentConfiguration = load_chess_experiment_configuration(CHESS_EXPERIMENT_TEMPLATE_PATH)
CHESS_TRAINING: TrainingArgs = CHESS_EXPERIMENT.training
CHESS_SELF_PLAY = CHESS_EXPERIMENT.chess.self_play

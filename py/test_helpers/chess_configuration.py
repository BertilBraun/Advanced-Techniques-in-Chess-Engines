from pathlib import Path

from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.training.configuration import TrainingArgs


CHESS_EXPERIMENT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / 'test' / 'configs' / 'chess-experiment.yaml'
CHESS_EXPERIMENT: ChessExperimentConfiguration = load_chess_experiment_configuration(CHESS_EXPERIMENT_TEMPLATE_PATH)
CHESS_TRAINING: TrainingArgs = CHESS_EXPERIMENT.training
CHESS_SELF_PLAY = CHESS_EXPERIMENT.chess.self_play

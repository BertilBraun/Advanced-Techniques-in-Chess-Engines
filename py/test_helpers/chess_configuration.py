from pathlib import Path

from src.experiment.configuration import ChessExperimentConfiguration, load_chess_experiment_configuration
from src.train.TrainingArgs import TrainingArgs


DEFAULT_CHESS_EXPERIMENT_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'chess-default-experiment.yaml'
CHESS_EXPERIMENT: ChessExperimentConfiguration = load_chess_experiment_configuration(DEFAULT_CHESS_EXPERIMENT_PATH)
CHESS_TRAINING: TrainingArgs = CHESS_EXPERIMENT.training

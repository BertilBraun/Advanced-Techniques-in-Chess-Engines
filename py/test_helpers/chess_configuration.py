from pathlib import Path

from src.experiment.configuration import ChessExperimentConfiguration, load_chess_experiment_configuration
from src.train.TrainingArgs import TrainingArgs


CHESS_EXPERIMENT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'chess-experiment-template.yaml'
CHESS_EXPERIMENT: ChessExperimentConfiguration = load_chess_experiment_configuration(CHESS_EXPERIMENT_TEMPLATE_PATH)
CHESS_TRAINING: TrainingArgs = CHESS_EXPERIMENT.training

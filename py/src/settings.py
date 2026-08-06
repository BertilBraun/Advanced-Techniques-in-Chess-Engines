from __future__ import annotations

import os
from pathlib import Path

from src.settings_common import *
from src.experiment.chess_experiment import ChessExperimentConfiguration, load_experiment_configuration
from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.ChessGame import ChessGame, ChessMove
from src.games.chess.ChessVisuals import ChessVisuals


DEFAULT_CHESS_EXPERIMENT_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'chess-default-experiment.yaml'


def _default_experiment_path() -> Path:
    configured_path = os.environ.get('ALPHAZERO_EXPERIMENT_PATH')
    return Path(configured_path) if configured_path is not None else DEFAULT_CHESS_EXPERIMENT_PATH


CurrentGameMove = ChessMove
CurrentGame = ChessGame()
CurrentBoard = ChessBoard
CurrentGameVisuals = ChessVisuals()

EXPERIMENT = load_experiment_configuration(_default_experiment_path())
CHESS_EXPERIMENT = EXPERIMENT if isinstance(EXPERIMENT, ChessExperimentConfiguration) else None
TRAINING_ARGS = EXPERIMENT.training
PLAY_C_PARAM = CHESS_EXPERIMENT.chess.evaluation.search_exploration_constant if CHESS_EXPERIMENT is not None else 1.0

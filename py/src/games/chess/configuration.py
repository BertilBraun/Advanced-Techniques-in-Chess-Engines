from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from src.experiment.base_configuration import BaseExperimentConfiguration
from src.games.chess.ChessGame import BINARY_CHANNELS, SCALAR_CHANNELS
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.neural_network import NetworkDimensions
from src.train.TrainingArgs import EvaluationParams
from src.util.frozen_model import FrozenModel


class ChessRulesConfiguration(FrozenModel):
    variant: Literal['standard'] = 'standard'
    chess960: bool = False
    automatic_fifty_move_draw: bool = True
    automatic_threefold_repetition_draw: bool = True


class ChessRepresentationConfiguration(FrozenModel):
    board_length: Literal[8] = 8
    binary_channels: tuple[int, ...] = BINARY_CHANNELS
    scalar_channels: tuple[int, ...] = SCALAR_CHANNELS
    action_encoding: Literal['chess-move2index-v1'] = 'chess-move2index-v1'
    canonical_player_perspective: bool = True

    @model_validator(mode='after')
    def validate_channels(self) -> ChessRepresentationConfiguration:
        if set(self.binary_channels) & set(self.scalar_channels):
            raise ValueError('Chess binary and scalar channels must be disjoint.')
        expected_channels = tuple(range(len(self.binary_channels) + len(self.scalar_channels)))
        actual_channels = tuple(sorted(self.binary_channels + self.scalar_channels))
        if actual_channels != expected_channels:
            raise ValueError('Chess representation channels must form one dense range starting at zero.')
        return self


class ChessConfiguration(FrozenModel):
    rules: ChessRulesConfiguration = ChessRulesConfiguration()
    representation: ChessRepresentationConfiguration = ChessRepresentationConfiguration()
    evaluation: EvaluationParams


class ChessExperimentConfiguration(BaseExperimentConfiguration):
    game: Literal['chess'] = 'chess'
    chess: ChessConfiguration

    @property
    def network_dimensions(self) -> NetworkDimensions:
        return CHESS_NETWORK_DIMENSIONS

    @model_validator(mode='after')
    def validate_experiment(self) -> ChessExperimentConfiguration:
        evaluation = self.chess.evaluation
        retention = self.training.lifecycle.inference_retention
        if evaluation.previous_model_offsets and (
            max(evaluation.previous_model_offsets) >= retention.recent_checkpoint_count
        ):
            raise ValueError('Recent inference-checkpoint retention must exceed every previous-model offset.')
        if any(
            model_version % retention.milestone_interval != 0 for model_version in evaluation.historical_model_versions
        ):
            raise ValueError('Historical model versions must align with retained milestone checkpoints.')
        return self

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.experiment.base_configuration import BaseExperimentConfiguration
from src.games.chess.game import BINARY_CHANNELS, SCALAR_CHANNELS
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.neural_network import NetworkDimensions
from src.training.configuration import BatchedInferenceParams
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


class ChessEvaluationConfiguration(FrozenModel):
    num_searches_per_turn: int = Field(gt=0)
    num_games: int = Field(gt=0)
    every_n_model_versions: int = Field(gt=0)
    max_concurrent_tasks: int = Field(gt=0)
    inference: BatchedInferenceParams
    dataset_path: str | None
    reference_model_path: str | None
    opening_suite_path: str | None
    raw_results_path: str | None
    maximum_game_plies: int | None = Field(default=None, gt=0)
    bootstrap_seed: int = Field(ge=0)
    bootstrap_samples: int = Field(gt=0)
    previous_model_offsets: tuple[int, ...]
    historical_model_versions: tuple[int, ...]
    historical_model_rotation_period: int = Field(gt=0)
    stockfish_skill_levels: tuple[int, ...]
    stockfish_binary_path: str | None
    stockfish_nodes_per_move: int = Field(gt=0)
    stockfish_threads: int = Field(gt=0)
    stockfish_hash_mib: int = Field(gt=0)
    evaluate_random: bool
    search_exploration_constant: float = Field(default=1.0, gt=0.0)
    parallel_searches: int = Field(default=1, gt=0)
    teacher_searches_per_turn: int = Field(default=600, gt=0)
    teacher_evaluation_games: int = Field(default=16, gt=0)


class ChessConfiguration(FrozenModel):
    rules: ChessRulesConfiguration = ChessRulesConfiguration()
    representation: ChessRepresentationConfiguration = ChessRepresentationConfiguration()
    evaluation: ChessEvaluationConfiguration


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

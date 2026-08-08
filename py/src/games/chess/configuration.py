from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.experiment.base_configuration import BaseExperimentConfiguration
from src.experiment.generation_schedule import IntegerGenerationSchedule
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.neural_network import NetworkDimensions
from src.training.configuration import (
    BatchedInferenceParams,
    SelfPlayConfiguration,
    TrainingObjectiveConfiguration,
)
from src.util.frozen_model import FrozenModel


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


class ChessSelfPlayConfiguration(SelfPlayConfiguration):
    maximum_game_plies: IntegerGenerationSchedule | None = None


class ChessConfiguration(FrozenModel):
    self_play: ChessSelfPlayConfiguration
    objective: TrainingObjectiveConfiguration
    evaluation: ChessEvaluationConfiguration


class ChessExperimentConfiguration(BaseExperimentConfiguration):
    game: Literal['chess'] = 'chess'
    chess: ChessConfiguration

    @property
    def network_dimensions(self) -> NetworkDimensions:
        return CHESS_NETWORK_DIMENSIONS

    @model_validator(mode='after')
    def validate_experiment(self) -> ChessExperimentConfiguration:
        self.training.validate_game(self.network_dimensions.actions, self.chess.self_play)
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

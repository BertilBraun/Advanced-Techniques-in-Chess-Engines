from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import yaml
from pydantic import Field, TypeAdapter, model_validator

from src.experiment.run_contract import EnvironmentConfiguration, HardwareConfiguration, TrainingStage
from src.games.chess.ChessGame import BINARY_CHANNELS, SCALAR_CHANNELS
from src.train.TrainingArgs import EvaluationParams, TrainingArgs
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class WeightsOnlyResumeConfiguration(FrozenModel):
    mode: Literal['weights_only']
    model_path: str


class RandomInitializationResumeConfiguration(FrozenModel):
    mode: Literal['random_initialization']


ResumeConfiguration = Annotated[
    WeightsOnlyResumeConfiguration | RandomInitializationResumeConfiguration,
    Field(discriminator='mode'),
]


class ExperimentRunConfiguration(FrozenModel):
    run_name: str = Field(min_length=1)
    tensorboard_run_directory: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    stage: TrainingStage
    requires_explicit_approval: bool
    resume: ResumeConfiguration
    hardware: HardwareConfiguration
    environment: EnvironmentConfiguration


class BaseExperimentConfiguration(FrozenModel):
    run: ExperimentRunConfiguration
    training: TrainingArgs


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


class GoRulesConfiguration(FrozenModel):
    scoring: Literal['area'] = 'area'
    komi_half_points: int
    maximum_moves: int = Field(gt=0)


class GoRepresentationConfiguration(FrozenModel):
    board_size: Literal[7, 9]
    history_length: Literal[8] = 8
    binary_channel_count: Literal[16] = 16
    scalar_channel_count: Literal[1] = 1
    action_encoding: Literal['go-point-pass-v1'] = 'go-point-pass-v1'
    canonical_player_perspective: bool = True

    @property
    def channel_count(self) -> int:
        return self.binary_channel_count + self.scalar_channel_count

    @property
    def action_count(self) -> int:
        return self.board_size * self.board_size + 1


class GoTrainingObjectiveConfiguration(FrozenModel):
    policy_loss_weight: float = Field(default=1.0, ge=0.0)
    outcome_value_loss_weight: float = Field(default=1.0, ge=0.0)
    root_value_loss_weight: float = Field(default=0.0, ge=0.0)

    @model_validator(mode='after')
    def validate_value_weights(self) -> GoTrainingObjectiveConfiguration:
        if abs(self.outcome_value_loss_weight + self.root_value_loss_weight - 1.0) > 1e-9:
            raise ValueError('Go value-objective component weights must sum to 1.')
        return self


class GoEvaluationConfiguration(FrozenModel):
    num_searches_per_turn: int = Field(gt=0)
    num_games: int = Field(gt=0)
    every_n_model_versions: int = Field(gt=0)
    max_concurrent_tasks: int = Field(gt=0)
    inference_workers: int = Field(gt=0)
    inference_batch_size: int = Field(gt=0)
    previous_model_offsets: tuple[int, ...]
    historical_model_versions: tuple[int, ...]
    evaluate_random: bool = True


class GoConfiguration(FrozenModel):
    rules: GoRulesConfiguration
    representation: GoRepresentationConfiguration
    objective: GoTrainingObjectiveConfiguration = GoTrainingObjectiveConfiguration()
    evaluation: GoEvaluationConfiguration

    @model_validator(mode='after')
    def validate_board_dependent_rules(self) -> GoConfiguration:
        point_count = self.representation.board_size**2
        if self.rules.maximum_moves < point_count * 2:
            raise ValueError('Go maximum moves must be at least twice the board point count.')
        if abs(self.rules.komi_half_points) > point_count * 2:
            raise ValueError('Go komi magnitude cannot exceed the board area.')
        return self


class GoExperimentConfiguration(BaseExperimentConfiguration):
    game: Literal['go'] = 'go'
    go: GoConfiguration

    @model_validator(mode='after')
    def validate_experiment(self) -> GoExperimentConfiguration:
        evaluation = self.go.evaluation
        retention = self.training.lifecycle.inference_retention
        if (
            evaluation.previous_model_offsets
            and max(evaluation.previous_model_offsets) >= retention.recent_checkpoint_count
        ):
            raise ValueError('Recent inference-checkpoint retention must exceed every previous-model offset.')
        if any(
            model_generation % retention.milestone_interval != 0
            for model_generation in evaluation.historical_model_versions
        ):
            raise ValueError('Historical model generations must align with retained milestone checkpoints.')
        if self.training.self_play.maximum_game_plies not in (None, self.go.rules.maximum_moves):
            raise ValueError('Go self-play maximum plies must be absent or equal the rules maximum moves.')
        return self


ExperimentConfiguration: TypeAlias = Annotated[
    ChessExperimentConfiguration | GoExperimentConfiguration,
    Field(discriminator='game'),
]


def load_experiment_configuration(path: Path) -> ExperimentConfiguration:
    payload = path.read_text(encoding='utf-8')
    parsed = yaml.safe_load(payload) if path.suffix.casefold() in {'.yaml', '.yml'} else json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f'Experiment file must contain a mapping: {path}')
    return TypeAdapter(ExperimentConfiguration).validate_python(parsed)


def load_chess_experiment_configuration(path: Path) -> ChessExperimentConfiguration:
    configuration = load_experiment_configuration(path)
    if not isinstance(configuration, ChessExperimentConfiguration):
        raise ValueError(f'Expected a chess experiment configuration: {path}')
    return configuration


def validate_experiment_queue(paths: tuple[Path, ...]) -> tuple[ExperimentConfiguration, ...]:
    if not paths:
        raise ValueError('Experiment queue validation requires at least one configuration path.')
    return tuple(load_experiment_configuration(path) for path in paths)


def write_resolved_experiment(path: Path, configuration: ExperimentConfiguration) -> None:
    write_text_atomically(path, configuration.model_dump_json(indent=2) + '\n')


def write_resolved_chess_experiment(path: Path, configuration: ChessExperimentConfiguration) -> None:
    write_resolved_experiment(path, configuration)

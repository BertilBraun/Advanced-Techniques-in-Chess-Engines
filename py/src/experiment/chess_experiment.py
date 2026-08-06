from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

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


class ChessRunConfiguration(FrozenModel):
    run_name: str = Field(min_length=1)
    tensorboard_run_directory: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    stage: TrainingStage
    requires_explicit_approval: bool
    resume: ResumeConfiguration
    hardware: HardwareConfiguration
    environment: EnvironmentConfiguration


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
    game: Literal['chess'] = 'chess'
    rules: ChessRulesConfiguration = ChessRulesConfiguration()
    representation: ChessRepresentationConfiguration = ChessRepresentationConfiguration()
    evaluation: EvaluationParams


class ChessExperimentConfiguration(FrozenModel):
    run: ChessRunConfiguration
    training: TrainingArgs
    chess: ChessConfiguration

    @model_validator(mode='after')
    def validate_experiment(self) -> ChessExperimentConfiguration:
        training = self.training
        world_size = len(training.topology.trainer.ddp_device_ids)
        expected_global_batch_size = training.trainer.local_batch_size * world_size
        if training.trainer.global_batch_size != expected_global_batch_size:
            raise ValueError(
                f'Global training batch size {training.trainer.global_batch_size} must equal '
                f'local batch size {training.trainer.local_batch_size} times world size {world_size}.'
            )
        training.lifecycle.evaluation.validate_for_optimizer_quantum(
            training.lifecycle.credit.optimizer_steps_per_quantum
        )
        evaluation = self.chess.evaluation
        retention = training.lifecycle.inference_retention
        if evaluation.previous_model_offsets and (
            max(evaluation.previous_model_offsets) >= retention.recent_checkpoint_count
        ):
            raise ValueError('Recent inference-checkpoint retention must exceed every previous-model offset.')
        if any(
            model_version % retention.milestone_interval != 0 for model_version in evaluation.historical_model_versions
        ):
            raise ValueError('Historical model versions must align with retained milestone checkpoints.')
        return self


def load_chess_experiment_configuration(path: Path) -> ChessExperimentConfiguration:
    payload = path.read_text(encoding='utf-8')
    parsed = yaml.safe_load(payload) if path.suffix.casefold() in {'.yaml', '.yml'} else json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f'Chess experiment file must contain a mapping: {path}')
    return ChessExperimentConfiguration.model_validate(parsed)


def validate_experiment_queue(paths: tuple[Path, ...]) -> tuple[ChessExperimentConfiguration, ...]:
    if not paths:
        raise ValueError('Experiment queue validation requires at least one configuration path.')
    return tuple(load_chess_experiment_configuration(path) for path in paths)


def write_resolved_chess_experiment(path: Path, configuration: ChessExperimentConfiguration) -> None:
    write_text_atomically(path, configuration.model_dump_json(indent=2) + '\n')

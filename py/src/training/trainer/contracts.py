from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import Field
from src.replay.description import ReplayDescription
from src.training.checkpoint import CheckpointReference
from src.training.distributions import TrainingDistributionSnapshot
from src.training.network import NetworkConfiguration
from src.training.objective import ResolvedTrainingObjective
from src.training.progress import TrainingProgress
from src.util.frozen_model import FrozenModel


class ResolvedTrainingParameters(FrozenModel):
    learning_rate: float
    objective: ResolvedTrainingObjective


class TrainerStartup(FrozenModel):
    network: NetworkConfiguration
    save_path: Path
    starting_generation: int = Field(ge=0)


@dataclass(frozen=True)
class TrainerQuantum:
    replay: ReplayDescription
    model_progress: TrainingProgress
    replay_source_progress: TrainingProgress


class TrainQuantumCommand(FrozenModel):
    kind: Literal['train_quantum'] = 'train_quantum'
    replay: ReplayDescription
    source_progress: TrainingProgress
    target_progress: TrainingProgress
    parameters: ResolvedTrainingParameters
    replay_source_optimizer_steps: int


class StopTrainerCommand(FrozenModel):
    kind: Literal['stop'] = 'stop'


TrainerCommand: TypeAlias = TrainQuantumCommand | StopTrainerCommand


class RankTrainingResult(FrozenModel):
    kind: Literal['trained'] = 'trained'
    rank: int
    completed_optimizer_steps: int
    policy_loss: float
    wdl_loss: float
    auxiliary_losses: tuple[float, ...]
    total_loss: float
    gradient_norm: float
    elapsed_seconds: float
    checkpoint: CheckpointReference | None
    distributions: TrainingDistributionSnapshot | None


class RankTrainingFailure(FrozenModel):
    kind: Literal['failed'] = 'failed'
    rank: int
    error: str


class TrainerStopped(FrozenModel):
    kind: Literal['stopped'] = 'stopped'
    rank: int


TrainerResponse: TypeAlias = RankTrainingResult | RankTrainingFailure | TrainerStopped


@dataclass(frozen=True)
class TrainingStatistics:
    policy_loss: float
    wdl_loss: float
    auxiliary_losses: tuple[float, ...]
    total_loss: float
    gradient_norm: float
    training_samples_per_second: float
    elapsed_seconds: float
    distributions: TrainingDistributionSnapshot


@dataclass(frozen=True)
class TrainingQuantumResult:
    completed_optimizer_steps: int
    checkpoint: CheckpointReference
    statistics: TrainingStatistics

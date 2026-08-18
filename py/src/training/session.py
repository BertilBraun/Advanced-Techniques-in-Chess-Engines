from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from src.experiment.configuration import ExperimentConfiguration
from src.games.implementation import GameImplementation
from src.replay.description import ReplayDescription
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.paths import checkpoint_manifest_path
from src.training.checkpoint.persistence import import_checkpoint, publish_checkpoint
from src.training.progress import TrainingProgress
from src.training.progressive import (
    CompletedCandidateTraining,
    ProgressiveTrainingStateStore,
    retain_progressive_candidate_checkpoints,
)
from src.training.trainer import TrainerGroup, TrainingQuantumResult, TrainingStatistics
from src.training.trainer.contracts import TrainerQuantum, TrainerStartup


@dataclass(frozen=True)
class ModelTrainingResult:
    model_id: str
    result: TrainingQuantumResult


@dataclass(frozen=True)
class TrainingPublication:
    completed_optimizer_steps: int
    checkpoint: CheckpointReference


@dataclass(frozen=True)
class TrainingSessionQuantum:
    replay: ReplayDescription
    progress: TrainingProgress
    active_checkpoint: CheckpointReference
    elapsed_seconds: float


@dataclass(frozen=True)
class FixedTrainingSessionResult:
    publication: TrainingPublication
    statistics: TrainingStatistics


@dataclass(frozen=True)
class ProgressiveTrainingSessionResult:
    publication: TrainingPublication
    active_model_id: str
    active_model_index: int
    model_results: tuple[ModelTrainingResult, ...]


TrainingSessionResult = FixedTrainingSessionResult | ProgressiveTrainingSessionResult


class TrainingSession(ABC):
    @property
    def has_pending_quantum(self) -> bool:
        return False

    @property
    def pauses_all_self_play_workers(self) -> bool:
        return False

    def recover_published_checkpoint(self, progress: TrainingProgress) -> CheckpointReference | None:
        return None

    @abstractmethod
    def train_quantum(self, quantum: TrainingSessionQuantum) -> TrainingSessionResult: ...

    def close(self) -> None:
        return None


class FixedTrainingSession(TrainingSession):
    def __init__(
        self,
        configuration: ExperimentConfiguration,
        game: GameImplementation,
        starting_checkpoint: CheckpointReference,
    ) -> None:
        self.trainer = TrainerGroup(
            configuration,
            game,
            TrainerStartup(
                network=configuration.training.initial_model.network,
                save_path=Path(configuration.training.save_path),
                starting_generation=starting_checkpoint.generation,
            ),
        )

    def train_quantum(self, quantum: TrainingSessionQuantum) -> FixedTrainingSessionResult:
        result = self.trainer.train_quantum(
            TrainerQuantum(
                replay=quantum.replay,
                model_progress=quantum.progress,
                replay_source_progress=quantum.progress,
            )
        )
        return FixedTrainingSessionResult(
            publication=TrainingPublication(
                completed_optimizer_steps=result.completed_optimizer_steps,
                checkpoint=result.checkpoint,
            ),
            statistics=result.statistics,
        )

    def close(self) -> None:
        self.trainer.close()


class ProgressiveTrainingSession(TrainingSession):
    def __init__(self, configuration: ExperimentConfiguration, game: GameImplementation) -> None:
        progressive_configuration = configuration.training.progressive_model_sizing
        self.configuration = configuration
        self.game = game
        self.progressive_configuration = progressive_configuration
        self.run_path = Path(configuration.training.save_path)
        self.optimizer_steps_per_quantum = configuration.training.lifecycle.credit.optimizer_steps_per_quantum
        self.state = ProgressiveTrainingStateStore(
            self.run_path / 'progressive-training.json',
            progressive_configuration,
        )

    @property
    def has_pending_quantum(self) -> bool:
        return self.state.state.pending_quantum is not None

    @property
    def pauses_all_self_play_workers(self) -> bool:
        return True

    def recover_published_checkpoint(self, progress: TrainingProgress) -> CheckpointReference | None:
        if self.has_pending_quantum:
            return None
        generation = progress.model_generation + 1
        if not checkpoint_manifest_path(generation, self.run_path).exists():
            return None
        return CheckpointReference.load(self.run_path, generation)

    def train_quantum(self, quantum: TrainingSessionQuantum) -> ProgressiveTrainingSessionResult:
        pending = self.state.begin_quantum(
            quantum.elapsed_seconds,
            quantum.replay,
            quantum.progress.completed_optimizer_steps,
            self.optimizer_steps_per_quantum,
        )
        model_results: list[ModelTrainingResult] = []
        while pending.next_model_id is not None:
            model_result = self._train_candidate(
                pending.next_model_id,
                quantum.replay,
                quantum.progress,
                quantum.active_checkpoint,
            )
            model_results.append(model_result)
            pending = self.state.state.pending_quantum
            assert pending is not None

        active_model_id = self.state.preview_active_model_id()
        active_candidate = self.state.completed_result(active_model_id)
        published_checkpoint = publish_checkpoint(
            active_candidate.checkpoint,
            quantum.progress.model_generation + 1,
            self.run_path,
        )
        self.state.complete_quantum()
        retain_progressive_candidate_checkpoints(self.run_path, self.state.state)
        return ProgressiveTrainingSessionResult(
            publication=TrainingPublication(
                completed_optimizer_steps=pending.target_global_optimizer_steps,
                checkpoint=published_checkpoint,
            ),
            active_model_id=active_model_id,
            active_model_index=self._model_index(active_model_id),
            model_results=tuple(model_results),
        )

    def _train_candidate(
        self,
        model_id: str,
        replay: ReplayDescription,
        replay_source_progress: TrainingProgress,
        active_checkpoint: CheckpointReference,
    ) -> ModelTrainingResult:
        definition = self.progressive_configuration.model(model_id)
        candidate = self.state.candidate(model_id)
        model_path = self.run_path / 'models' / model_id
        if candidate.checkpoint is None and model_id == self.state.state.active_model_id:
            imported = import_checkpoint(
                active_checkpoint.manifest_path,
                active_checkpoint.generation,
                model_path,
            )
            self.state.initialize_candidate(
                model_id,
                replay_source_progress.completed_optimizer_steps,
                imported,
            )
            candidate = self.state.candidate(model_id)
        model_progress = TrainingProgress(
            completed_optimizer_steps=candidate.completed_optimizer_steps,
            optimizer_steps_per_generation=self.optimizer_steps_per_quantum,
        )
        trainer = TrainerGroup(
            self.configuration,
            self.game,
            TrainerStartup(
                network=definition.network,
                save_path=model_path,
                starting_generation=0 if candidate.checkpoint is None else candidate.checkpoint.generation,
            ),
        )
        try:
            result = trainer.train_quantum(
                TrainerQuantum(
                    replay=replay,
                    model_progress=model_progress,
                    replay_source_progress=replay_source_progress,
                )
            )
        finally:
            trainer.close()
        self.state.record_candidate(
            CompletedCandidateTraining(
                model_id=model_id,
                completed_optimizer_steps=result.completed_optimizer_steps,
                checkpoint=result.checkpoint,
                comparable_total_loss=result.statistics.total_loss,
            )
        )
        return ModelTrainingResult(model_id=model_id, result=result)

    def _model_index(self, model_id: str) -> int:
        return tuple(model.model_id for model in self.progressive_configuration.models).index(model_id)


def create_training_session(
    configuration: ExperimentConfiguration,
    game: GameImplementation,
    starting_checkpoint: CheckpointReference,
) -> TrainingSession:
    if not configuration.training.progressive_model_sizing.is_progressive:
        return FixedTrainingSession(configuration, game, starting_checkpoint)
    return ProgressiveTrainingSession(configuration, game)

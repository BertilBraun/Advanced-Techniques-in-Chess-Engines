from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from src.training.checkpoint import CheckpointReference
from src.training.configuration import CreditTrainingParams
from src.training.progress import TrainingProgress
from src.training.trainer import TrainingQuantumResult
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class CreditLedgerState(FrozenModel):
    schema_version: Literal[1] = 1
    completed_optimizer_steps: int = Field(ge=0)
    earned_credits: Decimal = Field(ge=0)
    consumed_credits: Decimal = Field(ge=0)
    active_checkpoint: CheckpointReference

    @model_validator(mode='after')
    def validate_credits(self) -> CreditLedgerState:
        if self.consumed_credits > self.earned_credits:
            raise ValueError('Consumed training credits cannot exceed earned credits.')
        return self

    @property
    def available_credits(self) -> Decimal:
        return self.earned_credits - self.consumed_credits


class CreditLedger:
    def __init__(
        self,
        run_path: Path,
        parameters: CreditTrainingParams,
        global_batch_size: int,
        starting_checkpoint: CheckpointReference,
    ) -> None:
        self.path = run_path / 'credit-ledger.json'
        self.parameters = parameters
        self.global_batch_size = global_batch_size
        if self.path.exists():
            self._state = CreditLedgerState.model_validate_json(self.path.read_text(encoding='utf-8'))
        else:
            self._state = CreditLedgerState(
                completed_optimizer_steps=0,
                earned_credits=Decimal(0),
                consumed_credits=Decimal(0),
                active_checkpoint=starting_checkpoint,
            )
            self.save()
        if self._state.active_checkpoint.generation != self.model_generation:
            raise ValueError('Active checkpoint generation disagrees with completed optimizer progress.')
        self._adopt_completed_quantum(run_path)

    @property
    def state(self) -> CreditLedgerState:
        return self._state

    @property
    def progress(self) -> TrainingProgress:
        return TrainingProgress(
            completed_optimizer_steps=self._state.completed_optimizer_steps,
            optimizer_steps_per_generation=self.parameters.optimizer_steps_per_quantum,
        )

    @property
    def model_generation(self) -> int:
        steps = self._state.completed_optimizer_steps
        quantum = self.parameters.optimizer_steps_per_quantum
        if steps % quantum:
            raise ValueError('Optimizer progress must align with complete training quanta.')
        return steps // quantum

    @property
    def training_complete(self) -> bool:
        maximum_optimizer_steps = self.parameters.maximum_optimizer_steps
        return maximum_optimizer_steps is not None and self._state.completed_optimizer_steps >= maximum_optimizer_steps

    def can_train_quantum(self, live_samples: int) -> bool:
        required = self.parameters.presentation_credits_per_quantum(self.global_batch_size)
        return (
            not self.training_complete
            and live_samples >= self.global_batch_size
            and self._state.available_credits >= Decimal(required)
        )

    def add_samples(self, sample_count: int, model_generation: int) -> None:
        if sample_count < 0:
            raise ValueError('Ingested sample count cannot be negative.')
        if model_generation != self.model_generation:
            raise ValueError('Replay samples must be credited to the active model generation.')
        if sample_count == 0:
            return
        self._state = self._state.model_copy(
            update={'earned_credits': self._state.earned_credits + Decimal(sample_count) * self.parameters.replay_ratio}
        )
        self.save()

    def commit_quantum(self, result: TrainingQuantumResult) -> None:
        expected_steps = self._state.completed_optimizer_steps + self.parameters.optimizer_steps_per_quantum
        if result.completed_optimizer_steps != expected_steps:
            raise ValueError('Training result does not advance exactly one configured quantum.')
        expected_generation = self.model_generation + 1
        if result.checkpoint.generation != expected_generation:
            raise ValueError('Training checkpoint does not advance exactly one generation.')
        required = Decimal(self.parameters.presentation_credits_per_quantum(self.global_batch_size))
        if self._state.available_credits < required:
            raise ValueError('Training result cannot consume unavailable credits.')
        self._state = self._state.model_copy(
            update={
                'completed_optimizer_steps': result.completed_optimizer_steps,
                'consumed_credits': self._state.consumed_credits + required,
                'active_checkpoint': result.checkpoint,
            }
        )
        self.save()

    def save(self) -> None:
        write_text_atomically(self.path, self._state.model_dump_json(indent=2) + '\n')

    def _adopt_completed_quantum(self, run_path: Path) -> None:
        newer_generations = sorted(
            int(path.stem.removeprefix('checkpoint_'))
            for path in run_path.glob('checkpoint_*.json')
            if path.stem.removeprefix('checkpoint_').isdigit()
            and int(path.stem.removeprefix('checkpoint_')) > self.model_generation
        )
        if not newer_generations:
            return
        expected_generation = self.model_generation + 1
        if newer_generations != [expected_generation]:
            raise ValueError('Checkpoint manifests do not contain exactly the next uncommitted quantum.')
        required = Decimal(self.parameters.presentation_credits_per_quantum(self.global_batch_size))
        if self._state.available_credits < required:
            raise ValueError('Completed checkpoint cannot be adopted without its training credits.')
        checkpoint = CheckpointReference.load(run_path, expected_generation)
        self._state = self._state.model_copy(
            update={
                'completed_optimizer_steps': (
                    self._state.completed_optimizer_steps + self.parameters.optimizer_steps_per_quantum
                ),
                'consumed_credits': self._state.consumed_credits + required,
                'active_checkpoint': checkpoint,
            }
        )
        self.save()

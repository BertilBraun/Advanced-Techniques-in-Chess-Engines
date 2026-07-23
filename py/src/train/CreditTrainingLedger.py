from __future__ import annotations

import hashlib
import os
import time
import uuid
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.train.TrainingArgs import CreditTrainingParams


CREDIT_LEDGER_SCHEMA_VERSION = 1


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary_path.open('x', encoding='utf-8') as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class CreditTrainingProgress(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schema_version: int = CREDIT_LEDGER_SCHEMA_VERSION
    credited_unique_samples: int = Field(ge=0)
    earned_position_credits: Decimal = Field(ge=0)
    consumed_position_credits: Decimal = Field(ge=0)
    available_position_credits: Decimal = Field(ge=0)
    completed_optimizer_steps: int = Field(ge=0)
    completed_training_quanta: int = Field(ge=0)
    model_version: int = Field(ge=0)
    sampler_global_step: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_counters(self) -> CreditTrainingProgress:
        if self.schema_version != CREDIT_LEDGER_SCHEMA_VERSION:
            raise ValueError(f'Unsupported credit-ledger schema {self.schema_version}.')
        if self.available_position_credits != self.earned_position_credits - self.consumed_position_credits:
            raise ValueError('Available credits must equal earned credits minus consumed credits.')
        if self.model_version != self.completed_training_quanta:
            raise ValueError('Model version must equal the number of committed training quanta.')
        return self

    @classmethod
    def initial(cls) -> CreditTrainingProgress:
        return cls(
            credited_unique_samples=0,
            earned_position_credits=Decimal(0),
            consumed_position_credits=Decimal(0),
            available_position_credits=Decimal(0),
            completed_optimizer_steps=0,
            completed_training_quanta=0,
            model_version=0,
            sampler_global_step=0,
        )

    def reconcile_credited_samples(
        self,
        credited_unique_samples: int,
        replay_ratio: Decimal,
    ) -> CreditTrainingProgress:
        if credited_unique_samples < self.credited_unique_samples:
            raise ValueError('Durable replay credit count cannot move backwards.')
        earned_position_credits = Decimal(credited_unique_samples) * replay_ratio
        if earned_position_credits < self.consumed_position_credits:
            raise ValueError('Durable replay does not cover already consumed training credits.')
        return self.model_copy(
            update={
                'credited_unique_samples': credited_unique_samples,
                'earned_position_credits': earned_position_credits,
                'available_position_credits': earned_position_credits - self.consumed_position_credits,
            }
        )

    def can_train(self, required_position_credits: int) -> bool:
        if required_position_credits <= 0:
            raise ValueError('Required position credits must be positive.')
        return self.available_position_credits >= Decimal(required_position_credits)

    def consume_quantum(
        self,
        parameters: CreditTrainingParams,
        global_batch_size: int,
    ) -> CreditTrainingProgress:
        required_position_credits = parameters.presentation_credits_per_quantum(global_batch_size)
        if not self.can_train(required_position_credits):
            raise ValueError('Insufficient credits for a complete training quantum.')
        completed_optimizer_steps = self.completed_optimizer_steps + parameters.optimizer_steps_per_quantum
        if completed_optimizer_steps > parameters.maximum_optimizer_steps:
            raise ValueError('Training quantum exceeds the configured optimizer-step limit.')
        consumed_position_credits = self.consumed_position_credits + Decimal(required_position_credits)
        return self.model_copy(
            update={
                'consumed_position_credits': consumed_position_credits,
                'available_position_credits': self.earned_position_credits - consumed_position_credits,
                'completed_optimizer_steps': completed_optimizer_steps,
                'completed_training_quanta': self.completed_training_quanta + 1,
                'model_version': self.model_version + 1,
                'sampler_global_step': completed_optimizer_steps,
            }
        )


class PreparedTrainingQuantum(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schema_version: int = CREDIT_LEDGER_SCHEMA_VERSION
    previous_progress: CreditTrainingProgress
    prepared_progress: CreditTrainingProgress
    checkpoint_manifest_path: str
    checkpoint_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    prepared_at_seconds: float

    @model_validator(mode='after')
    def validate_transition(self) -> PreparedTrainingQuantum:
        if self.schema_version != CREDIT_LEDGER_SCHEMA_VERSION:
            raise ValueError(f'Unsupported prepared-quantum schema {self.schema_version}.')
        if self.prepared_progress.completed_training_quanta != self.previous_progress.completed_training_quanta + 1:
            raise ValueError('A prepared quantum must advance exactly one training quantum.')
        if self.prepared_progress.model_version != self.previous_progress.model_version + 1:
            raise ValueError('A prepared quantum must advance exactly one model version.')
        return self


class CommittedTrainingQuantum(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schema_version: int = CREDIT_LEDGER_SCHEMA_VERSION
    progress: CreditTrainingProgress
    checkpoint_manifest_path: str
    checkpoint_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    committed_at_seconds: float


class CreditTrainingLedger:
    """Crash-safe credit accounting with a two-phase quantum publication boundary."""

    def __init__(
        self,
        run_path: Path,
        parameters: CreditTrainingParams,
        global_batch_size: int,
    ) -> None:
        self.run_path = run_path
        self.parameters = parameters
        self.global_batch_size = global_batch_size
        self.progress_path = run_path / 'credit-training-progress.json'
        self.prepared_path = run_path / 'prepared-training-quantum.json'
        self.committed_directory = run_path / 'training_quanta'
        self._progress = self._load_last_committed_progress()
        self._recover_stale_prepared_quantum()
        self._persist_progress()

    @property
    def progress(self) -> CreditTrainingProgress:
        return self._progress

    @property
    def prepared_quantum(self) -> PreparedTrainingQuantum | None:
        if not self.prepared_path.exists():
            return None
        prepared = PreparedTrainingQuantum.model_validate_json(self.prepared_path.read_text(encoding='utf-8'))
        checkpoint_manifest = self.run_path / prepared.checkpoint_manifest_path
        if not checkpoint_manifest.is_file() or _sha256(checkpoint_manifest) != prepared.checkpoint_manifest_sha256:
            raise ValueError('Prepared training quantum references an invalid checkpoint manifest.')
        if prepared.previous_progress.completed_training_quanta != self._progress.completed_training_quanta:
            raise ValueError('Prepared training quantum does not follow the last committed quantum.')
        self._validate_quantum_transition(prepared.previous_progress, prepared.prepared_progress)
        return prepared

    def reconcile_credited_samples(self, credited_unique_samples: int) -> CreditTrainingProgress:
        if self.prepared_path.exists():
            raise RuntimeError('Cannot reconcile replay credits while a training quantum is prepared.')
        self._progress = self._progress.reconcile_credited_samples(
            credited_unique_samples,
            self.parameters.replay_ratio,
        )
        self._persist_progress()
        return self._progress

    def prepare_quantum(self, checkpoint_manifest_path: Path) -> PreparedTrainingQuantum:
        if self.prepared_path.exists():
            raise RuntimeError('A training quantum is already prepared.')
        if not checkpoint_manifest_path.is_file():
            raise ValueError(f'Checkpoint manifest does not exist: {checkpoint_manifest_path}')
        prepared_progress = self._progress.consume_quantum(self.parameters, self.global_batch_size)
        self._validate_quantum_transition(self._progress, prepared_progress)
        prepared = PreparedTrainingQuantum(
            previous_progress=self._progress,
            prepared_progress=prepared_progress,
            checkpoint_manifest_path=checkpoint_manifest_path.relative_to(self.run_path).as_posix(),
            checkpoint_manifest_sha256=_sha256(checkpoint_manifest_path),
            prepared_at_seconds=time.time(),
        )
        _atomic_write(self.prepared_path, prepared.model_dump_json(indent=2) + '\n')
        return prepared

    def commit_prepared_quantum(self) -> CreditTrainingProgress:
        prepared = self.prepared_quantum
        if prepared is None:
            raise RuntimeError('No training quantum is prepared.')
        committed = CommittedTrainingQuantum(
            progress=prepared.prepared_progress,
            checkpoint_manifest_path=prepared.checkpoint_manifest_path,
            checkpoint_manifest_sha256=prepared.checkpoint_manifest_sha256,
            committed_at_seconds=time.time(),
        )
        commit_path = self._commit_path(prepared.prepared_progress.completed_training_quanta)
        if commit_path.exists():
            raise RuntimeError(f'Training quantum is already committed: {commit_path}')
        _atomic_write(commit_path, committed.model_dump_json(indent=2) + '\n')
        self._progress = prepared.prepared_progress
        self._persist_progress()
        self.prepared_path.unlink()
        return self._progress

    def _load_last_committed_progress(self) -> CreditTrainingProgress:
        if not self.committed_directory.exists():
            return CreditTrainingProgress.initial()
        committed_quanta = tuple(
            CommittedTrainingQuantum.model_validate_json(path.read_text(encoding='utf-8'))
            for path in sorted(self.committed_directory.glob('quantum_*.json'))
        )
        if not committed_quanta:
            return CreditTrainingProgress.initial()
        previous_progress = CreditTrainingProgress.initial()
        for expected_quantum, committed in enumerate(committed_quanta, start=1):
            if committed.progress.completed_training_quanta != expected_quantum:
                raise ValueError('Committed training-quantum sequence is not contiguous.')
            self._validate_quantum_transition(previous_progress, committed.progress)
            checkpoint_manifest = self.run_path / committed.checkpoint_manifest_path
            if (
                not checkpoint_manifest.is_file()
                or _sha256(checkpoint_manifest) != committed.checkpoint_manifest_sha256
            ):
                raise ValueError(f'Committed quantum {expected_quantum} references an invalid checkpoint.')
            previous_progress = committed.progress
        return committed_quanta[-1].progress

    def _recover_stale_prepared_quantum(self) -> None:
        if not self.prepared_path.exists():
            return
        prepared = PreparedTrainingQuantum.model_validate_json(self.prepared_path.read_text(encoding='utf-8'))
        if prepared.prepared_progress == self._progress:
            self.prepared_path.unlink()
            return
        self._validate_quantum_transition(prepared.previous_progress, prepared.prepared_progress)
        committed_counters_match = (
            prepared.previous_progress.consumed_position_credits == self._progress.consumed_position_credits
            and prepared.previous_progress.completed_optimizer_steps == self._progress.completed_optimizer_steps
            and prepared.previous_progress.completed_training_quanta == self._progress.completed_training_quanta
            and prepared.previous_progress.model_version == self._progress.model_version
            and prepared.previous_progress.sampler_global_step == self._progress.sampler_global_step
        )
        if not committed_counters_match:
            raise ValueError('Prepared training quantum does not follow the last committed quantum.')
        self._progress = prepared.previous_progress

    def _validate_quantum_transition(
        self,
        previous: CreditTrainingProgress,
        current: CreditTrainingProgress,
    ) -> None:
        expected_credit_increment = Decimal(self.parameters.presentation_credits_per_quantum(self.global_batch_size))
        if current.completed_optimizer_steps - previous.completed_optimizer_steps != (
            self.parameters.optimizer_steps_per_quantum
        ):
            raise ValueError('Training quantum does not advance the configured optimizer-step count.')
        if current.completed_training_quanta != previous.completed_training_quanta + 1:
            raise ValueError('Training quantum does not advance exactly one quantum.')
        if current.model_version != previous.model_version + 1:
            raise ValueError('Training quantum does not advance exactly one model version.')
        if current.sampler_global_step != current.completed_optimizer_steps:
            raise ValueError('Replay sampler step must equal the completed optimizer-step count.')
        if current.credited_unique_samples < previous.credited_unique_samples:
            raise ValueError('Training quantum cannot reduce credited unique samples.')
        if current.earned_position_credits < previous.earned_position_credits:
            raise ValueError('Training quantum cannot reduce earned credits.')
        if current.earned_position_credits != Decimal(current.credited_unique_samples) * self.parameters.replay_ratio:
            raise ValueError('Training quantum earned credits do not match the configured replay ratio.')
        if previous.earned_position_credits != (
            Decimal(previous.credited_unique_samples) * self.parameters.replay_ratio
        ):
            raise ValueError('Previous earned credits do not match the configured replay ratio.')
        if current.consumed_position_credits - previous.consumed_position_credits != expected_credit_increment:
            raise ValueError('Training quantum consumes the wrong number of credits.')

    def _persist_progress(self) -> None:
        _atomic_write(self.progress_path, self._progress.model_dump_json(indent=2) + '\n')

    def _commit_path(self, quantum: int) -> Path:
        return self.committed_directory / f'quantum_{quantum:010d}.json'

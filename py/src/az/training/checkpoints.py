from __future__ import annotations

import hashlib
import os
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from src.az.config.base import FrozenModel, Sha256
from src.az.replay.credits import ReplayCreditState
from src.az.replay.sampling import ReplaySamplerState
from src.az.training.optimizer import LearningRateState


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_POINTER_NAME = 'current-checkpoint.json'


class CheckpointArtifactKind(str, Enum):
    MODEL = 'model'
    OPTIMIZER = 'optimizer'
    TORCH_RANDOM_STATE = 'torch_random_state'


class CheckpointArtifactFormat(str, Enum):
    TORCH_STATE_DICT_V1 = 'torch_state_dict_v1'
    TORCH_OPTIMIZER_STATE_V1 = 'torch_optimizer_state_v1'
    TORCH_CPU_RANDOM_STATE_V1 = 'torch_cpu_random_state_v1'


class CheckpointPurpose(str, Enum):
    CREDIT_COMMIT = 'credit_commit'
    SCHEDULED = 'scheduled'
    FINAL = 'final'


class CheckpointArtifact(FrozenModel):
    kind: CheckpointArtifactKind
    format: CheckpointArtifactFormat
    filename: str = Field(min_length=1, pattern=r'^[a-z0-9][a-z0-9._-]*$')
    byte_count: int = Field(gt=0)
    sha256: Sha256


class TrainerCheckpointState(FrozenModel):
    replay_credits: ReplayCreditState
    replay_sampler: ReplaySamplerState
    learning_rate: LearningRateState

    @model_validator(mode='after')
    def validate_steps(self) -> TrainerCheckpointState:
        optimizer_steps = self.replay_credits.completed_optimizer_steps
        if self.replay_sampler.next_optimizer_step != optimizer_steps:
            raise ValueError('Replay sampler step must equal the completed optimizer-step count.')
        if self.learning_rate.completed_optimizer_steps != optimizer_steps:
            raise ValueError('Learning-rate step must equal the completed optimizer-step count.')
        return self


class CheckpointManifest(FrozenModel):
    schema_version: Literal[1]
    run_id: UUID
    resolved_configuration_sha256: Sha256
    checkpoint_id: UUID
    created_at: datetime
    purpose: CheckpointPurpose
    state: TrainerCheckpointState
    model: CheckpointArtifact
    optimizer: CheckpointArtifact
    torch_random_state: CheckpointArtifact

    @field_validator('created_at')
    @classmethod
    def validate_utc_timestamp(cls, created_at: datetime) -> datetime:
        if created_at.tzinfo is None or created_at.utcoffset() != timezone.utc.utcoffset(created_at):
            raise ValueError('Checkpoint creation time must be timezone-aware UTC.')
        return created_at

    @model_validator(mode='after')
    def validate_manifest(self) -> CheckpointManifest:
        artifacts = (self.model, self.optimizer, self.torch_random_state)
        if tuple((artifact.kind, artifact.format) for artifact in artifacts) != (
            (CheckpointArtifactKind.MODEL, CheckpointArtifactFormat.TORCH_STATE_DICT_V1),
            (CheckpointArtifactKind.OPTIMIZER, CheckpointArtifactFormat.TORCH_OPTIMIZER_STATE_V1),
            (
                CheckpointArtifactKind.TORCH_RANDOM_STATE,
                CheckpointArtifactFormat.TORCH_CPU_RANDOM_STATE_V1,
            ),
        ):
            raise ValueError('Checkpoint artifacts do not match their manifest roles.')
        if len({artifact.filename for artifact in artifacts}) != len(artifacts):
            raise ValueError('Checkpoint artifact filenames must be unique.')
        return self


class CheckpointPointer(FrozenModel):
    schema_version: Literal[1]
    run_id: UUID
    model_version: int = Field(ge=0)
    checkpoint_directory: str = Field(pattern=r'^checkpoint-[0-9]{10}-[0-9a-f]{32}$')
    manifest_sha256: Sha256


@dataclass(frozen=True)
class LoadedCheckpoint:
    manifest: CheckpointManifest
    model_artifact: bytes
    optimizer_artifact: bytes
    torch_random_state_artifact: bytes


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _write_durable(path: Path, contents: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_replace(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid4().hex}.partial')
    try:
        _write_durable(temporary, contents)
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


class CheckpointRepository:
    def __init__(self, directory: Path, run_id: UUID, resolved_configuration_sha256: str) -> None:
        if not directory.is_absolute():
            raise ValueError('Checkpoint repository directory must be absolute.')
        self._directory = directory
        self._run_id = run_id
        self._configuration_sha256 = resolved_configuration_sha256
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def pointer_path(self) -> Path:
        return self._directory / CHECKPOINT_POINTER_NAME

    def publish(
        self,
        state: TrainerCheckpointState,
        purpose: CheckpointPurpose,
        model_artifact: bytes,
        optimizer_artifact: bytes,
        torch_random_state_artifact: bytes,
    ) -> LoadedCheckpoint:
        artifacts = (
            self._artifact(
                CheckpointArtifactKind.MODEL,
                CheckpointArtifactFormat.TORCH_STATE_DICT_V1,
                'model.pt',
                model_artifact,
            ),
            self._artifact(
                CheckpointArtifactKind.OPTIMIZER,
                CheckpointArtifactFormat.TORCH_OPTIMIZER_STATE_V1,
                'optimizer.pt',
                optimizer_artifact,
            ),
            self._artifact(
                CheckpointArtifactKind.TORCH_RANDOM_STATE,
                CheckpointArtifactFormat.TORCH_CPU_RANDOM_STATE_V1,
                'torch-random-state.bin',
                torch_random_state_artifact,
            ),
        )
        checkpoint_id = uuid4()
        if self.has_current():
            current_version = self.load_current().manifest.state.replay_credits.model_version
            if state.replay_credits.model_version != current_version + 1:
                raise ValueError('Published checkpoint model versions must increase by exactly one.')
        elif state.replay_credits.model_version != 1:
            raise ValueError('The first published training checkpoint must have model version one.')
        directory_name = f'checkpoint-{state.replay_credits.model_version:010d}-{checkpoint_id.hex}'
        destination = self._directory / directory_name
        temporary = self._directory / f'.{directory_name}.partial'
        if destination.exists() or temporary.exists():
            raise ValueError('Checkpoint destination already exists.')
        temporary.mkdir()
        try:
            for artifact, contents in zip(
                artifacts,
                (model_artifact, optimizer_artifact, torch_random_state_artifact),
                strict=True,
            ):
                _write_durable(temporary / artifact.filename, contents)
            manifest = CheckpointManifest(
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                run_id=self._run_id,
                resolved_configuration_sha256=self._configuration_sha256,
                checkpoint_id=checkpoint_id,
                created_at=datetime.now(timezone.utc),
                purpose=purpose,
                state=state,
                model=artifacts[0],
                optimizer=artifacts[1],
                torch_random_state=artifacts[2],
            )
            manifest_contents = (manifest.model_dump_json(indent=2) + '\n').encode()
            _write_durable(temporary / 'manifest.json', manifest_contents)
            temporary.rename(destination)
            _sync_directory(self._directory)
            pointer = CheckpointPointer(
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                run_id=self._run_id,
                model_version=state.replay_credits.model_version,
                checkpoint_directory=directory_name,
                manifest_sha256=_sha256(manifest_contents),
            )
            _atomic_replace(self.pointer_path, (pointer.model_dump_json(indent=2) + '\n').encode())
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return self.load_current()

    def load_current(self) -> LoadedCheckpoint:
        if not self.pointer_path.is_file():
            raise ValueError('No published checkpoint pointer exists.')
        try:
            pointer = CheckpointPointer.model_validate_json(self.pointer_path.read_bytes())
        except ValueError as error:
            raise ValueError('Published checkpoint pointer is invalid or torn.') from error
        if pointer.run_id != self._run_id:
            raise ValueError('Published checkpoint belongs to a different run.')
        checkpoint_directory = self._directory / pointer.checkpoint_directory
        if checkpoint_directory.parent.resolve() != self._directory.resolve() or not checkpoint_directory.is_dir():
            raise ValueError('Published checkpoint directory does not exist.')
        manifest_path = checkpoint_directory / 'manifest.json'
        if not manifest_path.is_file():
            raise ValueError('Published checkpoint manifest does not exist.')
        manifest_contents = manifest_path.read_bytes()
        if _sha256(manifest_contents) != pointer.manifest_sha256:
            raise ValueError('Published checkpoint manifest checksum mismatch.')
        try:
            manifest = CheckpointManifest.model_validate_json(manifest_contents)
        except ValueError as error:
            raise ValueError('Published checkpoint manifest is invalid.') from error
        if (
            manifest.run_id != self._run_id
            or manifest.resolved_configuration_sha256 != self._configuration_sha256
            or manifest.state.replay_credits.model_version != pointer.model_version
            or pointer.checkpoint_directory != f'checkpoint-{pointer.model_version:010d}-{manifest.checkpoint_id.hex}'
        ):
            raise ValueError('Published checkpoint identity does not match its repository or pointer.')
        artifact_contents = tuple(
            self._read_artifact(checkpoint_directory, artifact)
            for artifact in (manifest.model, manifest.optimizer, manifest.torch_random_state)
        )
        return LoadedCheckpoint(
            manifest=manifest,
            model_artifact=artifact_contents[0],
            optimizer_artifact=artifact_contents[1],
            torch_random_state_artifact=artifact_contents[2],
        )

    def has_current(self) -> bool:
        return self.pointer_path.exists()

    def _read_artifact(self, checkpoint_directory: Path, artifact: CheckpointArtifact) -> bytes:
        path = (checkpoint_directory / artifact.filename).resolve()
        if path.parent != checkpoint_directory.resolve() or not path.is_file():
            raise ValueError(f'Checkpoint {artifact.kind.value} artifact does not exist.')
        contents = path.read_bytes()
        if len(contents) != artifact.byte_count or _sha256(contents) != artifact.sha256:
            raise ValueError(f'Checkpoint {artifact.kind.value} artifact checksum mismatch.')
        return contents

    @staticmethod
    def _artifact(
        kind: CheckpointArtifactKind,
        artifact_format: CheckpointArtifactFormat,
        filename: str,
        contents: bytes,
    ) -> CheckpointArtifact:
        if not contents:
            raise ValueError(f'Checkpoint {kind.value} artifact cannot be empty.')
        return CheckpointArtifact(
            kind=kind,
            format=artifact_format,
            filename=filename,
            byte_count=len(contents),
            sha256=_sha256(contents),
        )


def _sync_directory(directory: Path) -> None:
    if os.name == 'nt':
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

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
from src.az.training.distributed import ProcessGroupLifecycle, TrainingDeterminism


CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_POINTER_NAME = 'current-checkpoint.json'


class CheckpointArtifactKind(str, Enum):
    MODEL = 'model'
    OPTIMIZER = 'optimizer'
    TORCH_RANDOM_STATE = 'torch_random_state'
    CUDA_RANDOM_STREAM = 'cuda_random_stream'
    GRADIENT_SCALER = 'gradient_scaler'


class CheckpointArtifactFormat(str, Enum):
    TORCH_STATE_DICT_V1 = 'torch_state_dict_v1'
    TORCH_OPTIMIZER_STATE_V1 = 'torch_optimizer_state_v1'
    TORCH_CPU_RANDOM_STATE_V1 = 'torch_cpu_random_state_v1'
    ASSIGNED_CUDA_RANDOM_STREAM_V1 = 'assigned_cuda_random_stream_v1'
    TORCH_GRADIENT_SCALER_V1 = 'torch_gradient_scaler_v1'


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
    process_group: ProcessGroupLifecycle
    training_determinism: TrainingDeterminism

    @model_validator(mode='after')
    def validate_steps(self) -> TrainerCheckpointState:
        optimizer_steps = self.replay_credits.completed_optimizer_steps
        if self.replay_sampler.next_optimizer_step != optimizer_steps:
            raise ValueError('Replay sampler step must equal the completed optimizer-step count.')
        if self.learning_rate.completed_optimizer_steps != optimizer_steps:
            raise ValueError('Learning-rate step must equal the completed optimizer-step count.')
        return self


class CheckpointManifest(FrozenModel):
    schema_version: Literal[2]
    run_id: UUID
    resolved_configuration_sha256: Sha256
    checkpoint_id: UUID
    created_at: datetime
    purpose: CheckpointPurpose
    state: TrainerCheckpointState
    model: CheckpointArtifact
    optimizer: CheckpointArtifact
    torch_random_state: CheckpointArtifact
    cuda_random_stream: CheckpointArtifact
    gradient_scaler: CheckpointArtifact

    @field_validator('created_at')
    @classmethod
    def validate_utc_timestamp(cls, created_at: datetime) -> datetime:
        if created_at.tzinfo is None or created_at.utcoffset() != timezone.utc.utcoffset(created_at):
            raise ValueError('Checkpoint creation time must be timezone-aware UTC.')
        return created_at

    @model_validator(mode='after')
    def validate_manifest(self) -> CheckpointManifest:
        artifacts = (
            self.model,
            self.optimizer,
            self.torch_random_state,
            self.cuda_random_stream,
            self.gradient_scaler,
        )
        if tuple((artifact.kind, artifact.format) for artifact in artifacts) != (
            (CheckpointArtifactKind.MODEL, CheckpointArtifactFormat.TORCH_STATE_DICT_V1),
            (CheckpointArtifactKind.OPTIMIZER, CheckpointArtifactFormat.TORCH_OPTIMIZER_STATE_V1),
            (
                CheckpointArtifactKind.TORCH_RANDOM_STATE,
                CheckpointArtifactFormat.TORCH_CPU_RANDOM_STATE_V1,
            ),
            (
                CheckpointArtifactKind.CUDA_RANDOM_STREAM,
                CheckpointArtifactFormat.ASSIGNED_CUDA_RANDOM_STREAM_V1,
            ),
            (
                CheckpointArtifactKind.GRADIENT_SCALER,
                CheckpointArtifactFormat.TORCH_GRADIENT_SCALER_V1,
            ),
        ):
            raise ValueError('Checkpoint artifacts do not match their manifest roles.')
        if len({artifact.filename for artifact in artifacts}) != len(artifacts):
            raise ValueError('Checkpoint artifact filenames must be unique.')
        return self


class CheckpointPointer(FrozenModel):
    schema_version: Literal[2]
    run_id: UUID
    model_version: int = Field(ge=0)
    checkpoint_directory: str = Field(pattern=r'^(checkpoint-[0-9]{10}-[0-9a-f]{32}|distributed-[0-9]{10})$')
    manifest_sha256: Sha256


@dataclass(frozen=True)
class LoadedCheckpoint:
    manifest: CheckpointManifest
    model_artifact: bytes
    optimizer_artifact: bytes
    torch_random_state_artifact: bytes
    cuda_random_stream_artifact: bytes
    gradient_scaler_artifact: bytes


@dataclass(frozen=True)
class LoadedModelCheckpoint:
    manifest: ModelCheckpointManifest
    model_artifact: bytes


class DistributedRankStage(FrozenModel):
    state: TrainerCheckpointState
    model_sha256: Sha256
    optimizer_sha256: Sha256
    gradient_scaler_sha256: Sha256
    torch_random_state: CheckpointArtifact
    cuda_random_stream: CheckpointArtifact


class DistributedRankCheckpoint(FrozenModel):
    rank: int = Field(ge=0)
    state: TrainerCheckpointState
    torch_random_state: CheckpointArtifact
    cuda_random_stream: CheckpointArtifact


class DistributedCheckpointManifest(FrozenModel):
    schema_version: Literal[1]
    run_id: UUID
    resolved_configuration_sha256: Sha256
    checkpoint_id: UUID
    created_at: datetime
    purpose: CheckpointPurpose
    model_version: int = Field(gt=0)
    model: CheckpointArtifact
    optimizer: CheckpointArtifact
    gradient_scaler: CheckpointArtifact
    ranks: tuple[DistributedRankCheckpoint, ...] = Field(min_length=1)

    @field_validator('created_at')
    @classmethod
    def validate_utc_timestamp(cls, created_at: datetime) -> datetime:
        if created_at.tzinfo is None or created_at.utcoffset() != timezone.utc.utcoffset(created_at):
            raise ValueError('Distributed checkpoint creation time must be timezone-aware UTC.')
        return created_at

    @model_validator(mode='after')
    def validate_ranks(self) -> DistributedCheckpointManifest:
        ranks = tuple(rank.rank for rank in self.ranks)
        if ranks != tuple(range(len(ranks))):
            raise ValueError('Distributed checkpoint ranks must be contiguous from zero.')
        if any(rank.rank != rank.state.process_group.rank for rank in self.ranks):
            raise ValueError('Distributed checkpoint rank identity does not match process-group state.')
        lifecycles = {rank.state.process_group for rank in self.ranks}
        if len(lifecycles) != len(self.ranks):
            raise ValueError('Distributed checkpoint process-group rank state must be unique.')
        first_lifecycle = self.ranks[0].state.process_group
        if (
            first_lifecycle.world_size != len(self.ranks)
            or not first_lifecycle.initialized
            or any(
                rank.state.process_group.world_size != first_lifecycle.world_size
                or rank.state.process_group.backend != first_lifecycle.backend
                or not rank.state.process_group.initialized
                for rank in self.ranks
            )
        ):
            raise ValueError('Distributed checkpoint process-group lifecycle is inconsistent.')
        ledgers = {rank.state.replay_credits for rank in self.ranks}
        learning_rates = {rank.state.learning_rate for rank in self.ranks}
        determinism = {rank.state.training_determinism for rank in self.ranks}
        if len(ledgers) != 1 or len(learning_rates) != 1 or len(determinism) != 1:
            raise ValueError('Distributed checkpoint common trainer state must match across ranks.')
        if next(iter(ledgers)).model_version != self.model_version:
            raise ValueError('Distributed checkpoint model version does not match rank ledgers.')
        common_artifacts = (self.model, self.optimizer, self.gradient_scaler)
        if tuple((artifact.kind, artifact.format) for artifact in common_artifacts) != (
            (CheckpointArtifactKind.MODEL, CheckpointArtifactFormat.TORCH_STATE_DICT_V1),
            (CheckpointArtifactKind.OPTIMIZER, CheckpointArtifactFormat.TORCH_OPTIMIZER_STATE_V1),
            (
                CheckpointArtifactKind.GRADIENT_SCALER,
                CheckpointArtifactFormat.TORCH_GRADIENT_SCALER_V1,
            ),
        ):
            raise ValueError('Distributed common artifacts do not match their manifest roles.')
        if any(
            (
                rank.torch_random_state.kind,
                rank.torch_random_state.format,
                rank.cuda_random_stream.kind,
                rank.cuda_random_stream.format,
            )
            != (
                CheckpointArtifactKind.TORCH_RANDOM_STATE,
                CheckpointArtifactFormat.TORCH_CPU_RANDOM_STATE_V1,
                CheckpointArtifactKind.CUDA_RANDOM_STREAM,
                CheckpointArtifactFormat.ASSIGNED_CUDA_RANDOM_STREAM_V1,
            )
            for rank in self.ranks
        ):
            raise ValueError('Distributed rank artifacts do not match their manifest roles.')
        artifact_paths = tuple(artifact.filename for artifact in common_artifacts) + tuple(
            f'rank-{rank.rank:05d}/{artifact.filename}'
            for rank in self.ranks
            for artifact in (rank.torch_random_state, rank.cuda_random_stream)
        )
        if len(set(artifact_paths)) != len(artifact_paths):
            raise ValueError('Distributed checkpoint artifact paths must be unique.')
        return self


class ModelCheckpointManifest(FrozenModel):
    run_id: UUID
    resolved_configuration_sha256: Sha256
    checkpoint_id: UUID
    created_at: datetime
    purpose: CheckpointPurpose
    model_version: int = Field(gt=0)
    model: CheckpointArtifact

    @field_validator('created_at')
    @classmethod
    def validate_utc_timestamp(cls, created_at: datetime) -> datetime:
        if created_at.tzinfo is None or created_at.utcoffset() != timezone.utc.utcoffset(created_at):
            raise ValueError('Model checkpoint creation time must be timezone-aware UTC.')
        return created_at

    @model_validator(mode='after')
    def validate_model(self) -> ModelCheckpointManifest:
        if (
            self.model.kind is not CheckpointArtifactKind.MODEL
            or self.model.format is not CheckpointArtifactFormat.TORCH_STATE_DICT_V1
        ):
            raise ValueError('Model checkpoint artifact does not match its manifest role.')
        return self


@dataclass(frozen=True)
class LoadedDistributedCheckpoint:
    manifest: DistributedCheckpointManifest
    rank: DistributedRankCheckpoint
    model_artifact: bytes
    optimizer_artifact: bytes
    gradient_scaler_artifact: bytes
    torch_random_state_artifact: bytes
    cuda_random_stream_artifact: bytes


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
        cuda_random_stream_artifact: bytes,
        gradient_scaler_artifact: bytes,
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
            self._artifact(
                CheckpointArtifactKind.CUDA_RANDOM_STREAM,
                CheckpointArtifactFormat.ASSIGNED_CUDA_RANDOM_STREAM_V1,
                'cuda-random-stream.json',
                cuda_random_stream_artifact,
            ),
            self._artifact(
                CheckpointArtifactKind.GRADIENT_SCALER,
                CheckpointArtifactFormat.TORCH_GRADIENT_SCALER_V1,
                'gradient-scaler.pt',
                gradient_scaler_artifact,
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
                (
                    model_artifact,
                    optimizer_artifact,
                    torch_random_state_artifact,
                    cuda_random_stream_artifact,
                    gradient_scaler_artifact,
                ),
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
                cuda_random_stream=artifacts[3],
                gradient_scaler=artifacts[4],
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

    def stage_distributed_rank(
        self,
        state: TrainerCheckpointState,
        model_artifact: bytes,
        optimizer_artifact: bytes,
        torch_random_state_artifact: bytes,
        cuda_random_stream_artifact: bytes,
        gradient_scaler_artifact: bytes,
    ) -> None:
        lifecycle = state.process_group
        if lifecycle.world_size <= 1 or not lifecycle.initialized:
            raise ValueError('Distributed rank staging requires an initialized multi-rank process group.')
        generation = self._distributed_staging_directory(state.replay_credits.model_version)
        rank_directory = generation / f'rank-{lifecycle.rank:05d}'
        rank_directory.mkdir(parents=True, exist_ok=True)
        torch_artifact = self._artifact(
            CheckpointArtifactKind.TORCH_RANDOM_STATE,
            CheckpointArtifactFormat.TORCH_CPU_RANDOM_STATE_V1,
            'torch-random-state.bin',
            torch_random_state_artifact,
        )
        cuda_artifact = self._artifact(
            CheckpointArtifactKind.CUDA_RANDOM_STREAM,
            CheckpointArtifactFormat.ASSIGNED_CUDA_RANDOM_STREAM_V1,
            'cuda-random-stream.json',
            cuda_random_stream_artifact,
        )
        _atomic_replace(rank_directory / torch_artifact.filename, torch_random_state_artifact)
        _atomic_replace(rank_directory / cuda_artifact.filename, cuda_random_stream_artifact)
        stage = DistributedRankStage(
            state=state,
            model_sha256=_sha256(model_artifact),
            optimizer_sha256=_sha256(optimizer_artifact),
            gradient_scaler_sha256=_sha256(gradient_scaler_artifact),
            torch_random_state=torch_artifact,
            cuda_random_stream=cuda_artifact,
        )
        _atomic_replace(
            rank_directory / 'stage.json',
            (stage.model_dump_json(indent=2) + '\n').encode(),
        )

    def commit_distributed_generation(
        self,
        state: TrainerCheckpointState,
        purpose: CheckpointPurpose,
        model_artifact: bytes,
        optimizer_artifact: bytes,
        gradient_scaler_artifact: bytes,
    ) -> LoadedDistributedCheckpoint:
        lifecycle = state.process_group
        if lifecycle.rank != 0 or lifecycle.world_size <= 1:
            raise ValueError('Only distributed rank zero can commit a multi-rank generation.')
        model_version = state.replay_credits.model_version
        destination = self._directory / f'distributed-{model_version:010d}'
        if destination.exists():
            return self._recover_distributed_generation(
                destination=destination,
                state=state,
                purpose=purpose,
                model_artifact=model_artifact,
                optimizer_artifact=optimizer_artifact,
                gradient_scaler_artifact=gradient_scaler_artifact,
            )
        staging = self._distributed_staging_directory(model_version)
        stages = tuple(self._read_rank_stage(staging, rank) for rank in range(lifecycle.world_size))
        for rank, rank_stage in enumerate(stages):
            rank_directory = staging / f'rank-{rank:05d}'
            self._read_artifact(rank_directory, rank_stage.torch_random_state)
            self._read_artifact(rank_directory, rank_stage.cuda_random_stream)
        expected_digests = (
            _sha256(model_artifact),
            _sha256(optimizer_artifact),
            _sha256(gradient_scaler_artifact),
        )
        if any(
            (
                rank_stage.model_sha256,
                rank_stage.optimizer_sha256,
                rank_stage.gradient_scaler_sha256,
            )
            != expected_digests
            for rank_stage in stages
        ):
            raise ValueError('Distributed ranks do not agree on common checkpoint artifacts.')
        current_version = self.current_model_version()
        if current_version is not None and model_version != current_version + 1:
            raise ValueError('Distributed checkpoint generation must advance by exactly one.')
        if current_version is None and model_version != 1:
            raise ValueError('First distributed checkpoint generation must be one.')
        common_artifacts = (
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
                CheckpointArtifactKind.GRADIENT_SCALER,
                CheckpointArtifactFormat.TORCH_GRADIENT_SCALER_V1,
                'gradient-scaler.pt',
                gradient_scaler_artifact,
            ),
        )
        for artifact, contents in zip(
            common_artifacts,
            (model_artifact, optimizer_artifact, gradient_scaler_artifact),
            strict=True,
        ):
            _atomic_replace(staging / artifact.filename, contents)
        ranks = tuple(
            DistributedRankCheckpoint(
                rank=rank,
                state=rank_stage.state,
                torch_random_state=rank_stage.torch_random_state,
                cuda_random_stream=rank_stage.cuda_random_stream,
            )
            for rank, rank_stage in enumerate(stages)
        )
        manifest = DistributedCheckpointManifest(
            schema_version=1,
            run_id=self._run_id,
            resolved_configuration_sha256=self._configuration_sha256,
            checkpoint_id=uuid4(),
            created_at=datetime.now(timezone.utc),
            purpose=purpose,
            model_version=model_version,
            model=common_artifacts[0],
            optimizer=common_artifacts[1],
            gradient_scaler=common_artifacts[2],
            ranks=ranks,
        )
        manifest_contents = (manifest.model_dump_json(indent=2) + '\n').encode()
        _atomic_replace(staging / 'distributed-manifest.json', manifest_contents)
        staging.rename(destination)
        _sync_directory(self._directory)
        pointer = CheckpointPointer(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            run_id=self._run_id,
            model_version=model_version,
            checkpoint_directory=destination.name,
            manifest_sha256=_sha256(manifest_contents),
        )
        _atomic_replace(
            self.pointer_path,
            (pointer.model_dump_json(indent=2) + '\n').encode(),
        )
        return self.load_distributed(rank=0)

    def load_distributed(self, rank: int) -> LoadedDistributedCheckpoint:
        pointer = self._load_pointer()
        if not pointer.checkpoint_directory.startswith('distributed-'):
            raise ValueError('Current checkpoint is not a distributed generation.')
        directory = self._directory / pointer.checkpoint_directory
        manifest = self._load_distributed_manifest(directory, pointer)
        if not 0 <= rank < len(manifest.ranks):
            raise ValueError('Distributed checkpoint rank is outside the committed world.')
        rank_manifest = manifest.ranks[rank]
        rank_directory = directory / f'rank-{rank:05d}'
        return LoadedDistributedCheckpoint(
            manifest=manifest,
            rank=rank_manifest,
            model_artifact=self._read_artifact(directory, manifest.model),
            optimizer_artifact=self._read_artifact(directory, manifest.optimizer),
            gradient_scaler_artifact=self._read_artifact(
                directory,
                manifest.gradient_scaler,
            ),
            torch_random_state_artifact=self._read_artifact(
                rank_directory,
                rank_manifest.torch_random_state,
            ),
            cuda_random_stream_artifact=self._read_artifact(
                rank_directory,
                rank_manifest.cuda_random_stream,
            ),
        )

    def _recover_distributed_generation(
        self,
        destination: Path,
        state: TrainerCheckpointState,
        purpose: CheckpointPurpose,
        model_artifact: bytes,
        optimizer_artifact: bytes,
        gradient_scaler_artifact: bytes,
    ) -> LoadedDistributedCheckpoint:
        model_version = state.replay_credits.model_version
        manifest, manifest_contents = self._validate_distributed_directory(destination, model_version)
        if (
            manifest.purpose != purpose
            or manifest.ranks[0].state != state
            or manifest.model.sha256 != _sha256(model_artifact)
            or manifest.optimizer.sha256 != _sha256(optimizer_artifact)
            or manifest.gradient_scaler.sha256 != _sha256(gradient_scaler_artifact)
        ):
            raise ValueError('Existing distributed generation does not match the attempted commit.')
        current_version = self.current_model_version()
        if current_version == model_version:
            pointer = self._load_pointer()
            if pointer.checkpoint_directory != destination.name:
                raise ValueError('Published distributed generation directory does not match its pointer.')
            return self.load_distributed(rank=0)
        if current_version is not None and current_version != model_version - 1:
            raise ValueError('Orphan distributed generation does not immediately follow the current pointer.')
        if current_version is None and model_version != 1:
            raise ValueError('First recoverable distributed generation must be one.')
        pointer = CheckpointPointer(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            run_id=self._run_id,
            model_version=model_version,
            checkpoint_directory=destination.name,
            manifest_sha256=_sha256(manifest_contents),
        )
        _atomic_replace(
            self.pointer_path,
            (pointer.model_dump_json(indent=2) + '\n').encode(),
        )
        return self.load_distributed(rank=0)

    def _validate_distributed_directory(
        self,
        directory: Path,
        expected_model_version: int,
    ) -> tuple[DistributedCheckpointManifest, bytes]:
        if directory.resolve().parent != self._directory.resolve() or not directory.is_dir():
            raise ValueError('Distributed checkpoint directory is outside its repository.')
        manifest_path = directory / 'distributed-manifest.json'
        if not manifest_path.is_file():
            raise ValueError('Distributed checkpoint manifest does not exist.')
        manifest_contents = manifest_path.read_bytes()
        try:
            manifest = DistributedCheckpointManifest.model_validate_json(manifest_contents)
        except ValueError as error:
            raise ValueError('Distributed checkpoint manifest is invalid.') from error
        if (
            manifest.run_id != self._run_id
            or manifest.resolved_configuration_sha256 != self._configuration_sha256
            or manifest.model_version != expected_model_version
            or directory.name != f'distributed-{expected_model_version:010d}'
        ):
            raise ValueError('Distributed checkpoint identity does not match its repository.')
        for artifact in (manifest.model, manifest.optimizer, manifest.gradient_scaler):
            self._read_artifact(directory, artifact)
        for rank in manifest.ranks:
            rank_directory = directory / f'rank-{rank.rank:05d}'
            self._read_artifact(rank_directory, rank.torch_random_state)
            self._read_artifact(rank_directory, rank.cuda_random_stream)
        return manifest, manifest_contents

    def _load_distributed_manifest(
        self,
        directory: Path,
        pointer: CheckpointPointer,
    ) -> DistributedCheckpointManifest:
        if directory.resolve().parent != self._directory.resolve() or not directory.is_dir():
            raise ValueError('Distributed checkpoint directory does not exist.')
        manifest_path = directory / 'distributed-manifest.json'
        if not manifest_path.is_file():
            raise ValueError('Distributed checkpoint manifest does not exist.')
        manifest_contents = manifest_path.read_bytes()
        if _sha256(manifest_contents) != pointer.manifest_sha256:
            raise ValueError('Distributed checkpoint manifest checksum mismatch.')
        try:
            manifest = DistributedCheckpointManifest.model_validate_json(manifest_contents)
        except ValueError as error:
            raise ValueError('Distributed checkpoint manifest is invalid.') from error
        if (
            manifest.run_id != self._run_id
            or manifest.resolved_configuration_sha256 != self._configuration_sha256
            or manifest.model_version != pointer.model_version
            or directory.name != f'distributed-{pointer.model_version:010d}'
        ):
            raise ValueError('Distributed checkpoint identity does not match its repository.')
        return manifest

    def load_current(self) -> LoadedCheckpoint:
        checkpoint_directory, manifest = self._load_current_manifest()
        artifact_contents = tuple(
            self._read_artifact(checkpoint_directory, artifact)
            for artifact in (
                manifest.model,
                manifest.optimizer,
                manifest.torch_random_state,
                manifest.cuda_random_stream,
                manifest.gradient_scaler,
            )
        )
        return LoadedCheckpoint(
            manifest=manifest,
            model_artifact=artifact_contents[0],
            optimizer_artifact=artifact_contents[1],
            torch_random_state_artifact=artifact_contents[2],
            cuda_random_stream_artifact=artifact_contents[3],
            gradient_scaler_artifact=artifact_contents[4],
        )

    def load_current_model(self) -> LoadedModelCheckpoint:
        pointer = self._load_pointer()
        if pointer.checkpoint_directory.startswith('distributed-'):
            checkpoint_directory = self._directory / pointer.checkpoint_directory
            distributed_manifest = self._load_distributed_manifest(checkpoint_directory, pointer)
            return LoadedModelCheckpoint(
                manifest=self._model_manifest(
                    distributed_manifest.run_id,
                    distributed_manifest.resolved_configuration_sha256,
                    distributed_manifest.checkpoint_id,
                    distributed_manifest.created_at,
                    distributed_manifest.purpose,
                    distributed_manifest.model_version,
                    distributed_manifest.model,
                ),
                model_artifact=self._read_artifact(checkpoint_directory, distributed_manifest.model),
            )
        checkpoint_directory, manifest = self._load_current_manifest()
        return LoadedModelCheckpoint(
            manifest=self._model_manifest(
                manifest.run_id,
                manifest.resolved_configuration_sha256,
                manifest.checkpoint_id,
                manifest.created_at,
                manifest.purpose,
                manifest.state.replay_credits.model_version,
                manifest.model,
            ),
            model_artifact=self._read_artifact(checkpoint_directory, manifest.model),
        )

    def current_model_version(self) -> int | None:
        if not self.pointer_path.is_file():
            return None
        return self._load_pointer().model_version

    def _load_pointer(self) -> CheckpointPointer:
        if not self.pointer_path.is_file():
            raise ValueError('No published checkpoint pointer exists.')
        try:
            pointer = CheckpointPointer.model_validate_json(self.pointer_path.read_bytes())
        except ValueError as error:
            raise ValueError('Published checkpoint pointer is invalid or torn.') from error
        if pointer.run_id != self._run_id:
            raise ValueError('Published checkpoint belongs to a different run.')
        return pointer

    def _load_current_manifest(self) -> tuple[Path, CheckpointManifest]:
        pointer = self._load_pointer()
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
        return checkpoint_directory, manifest

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
    def _model_manifest(
        run_id: UUID,
        resolved_configuration_sha256: str,
        checkpoint_id: UUID,
        created_at: datetime,
        purpose: CheckpointPurpose,
        model_version: int,
        model: CheckpointArtifact,
    ) -> ModelCheckpointManifest:
        return ModelCheckpointManifest(
            run_id=run_id,
            resolved_configuration_sha256=resolved_configuration_sha256,
            checkpoint_id=checkpoint_id,
            created_at=created_at,
            purpose=purpose,
            model_version=model_version,
            model=model,
        )

    def _distributed_staging_directory(self, model_version: int) -> Path:
        if model_version <= 0:
            raise ValueError('Distributed checkpoint model version must be positive.')
        staging = self._directory / f'.distributed-{model_version:010d}.partial'
        staging.mkdir(parents=True, exist_ok=True)
        return staging

    @staticmethod
    def _read_rank_stage(
        staging: Path,
        rank: int,
    ) -> DistributedRankStage:
        path = staging / f'rank-{rank:05d}' / 'stage.json'
        if not path.is_file():
            raise ValueError(f'Distributed checkpoint rank {rank} has not staged its state.')
        try:
            return DistributedRankStage.model_validate_json(path.read_bytes())
        except ValueError as error:
            raise ValueError(f'Distributed checkpoint rank {rank} stage is invalid.') from error

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

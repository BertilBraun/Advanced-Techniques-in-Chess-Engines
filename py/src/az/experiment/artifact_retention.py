from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from src.az.config.runtime import RetentionConfiguration
from src.az.training.checkpoints import (
    CheckpointArtifact,
    CheckpointManifest,
    CheckpointPointer,
    DistributedCheckpointManifest,
)


@dataclass(frozen=True)
class RetentionResult:
    retained_checkpoint_directories: tuple[str, ...]
    deleted_checkpoint_directories: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedCheckpointMetadata:
    path: Path
    run_id: UUID
    model_version: int
    completed_optimizer_steps: int
    manifest_sha256: str


def apply_checkpoint_retention(
    checkpoint_repository_directory: Path,
    configuration: RetentionConfiguration,
) -> RetentionResult:
    pointer_path = checkpoint_repository_directory / 'current-checkpoint.json'
    if not pointer_path.is_file():
        raise ValueError('Checkpoint retention requires a published current checkpoint.')
    try:
        pointer = CheckpointPointer.model_validate_json(pointer_path.read_bytes())
    except ValueError as error:
        raise ValueError('Checkpoint retention cannot use an invalid current pointer.') from error
    checkpoint_directories = tuple(
        sorted(
            (
                path
                for path in checkpoint_repository_directory.iterdir()
                if path.is_dir() and (path.name.startswith('checkpoint-') or path.name.startswith('distributed-'))
            ),
            key=lambda path: (_model_version(path), path.name),
        )
    )
    if not any(path.name == pointer.checkpoint_directory for path in checkpoint_directories):
        raise ValueError('Checkpoint retention cannot find the currently published checkpoint.')
    metadata = tuple(
        _validate_checkpoint_metadata(path, checkpoint_repository_directory) for path in checkpoint_directories
    )
    current = next(item for item in metadata if item.path.name == pointer.checkpoint_directory)
    if (
        current.manifest_sha256 != pointer.manifest_sha256
        or current.run_id != pointer.run_id
        or current.model_version != pointer.model_version
    ):
        raise ValueError('Current checkpoint pointer does not authenticate its manifest identity.')
    recent = metadata[-configuration.recent_checkpoint_count :]
    retained_names = {item.path.name for item in recent}
    retained_names.add(pointer.checkpoint_directory)
    for item in metadata:
        if item.completed_optimizer_steps % configuration.milestone_every_optimizer_steps == 0:
            retained_names.add(item.path.name)
    deletion_targets = tuple(item.path for item in metadata if item.path.name not in retained_names)
    deleted: list[str] = []
    for path in deletion_targets:
        shutil.rmtree(path)
        deleted.append(path.name)
    return RetentionResult(
        retained_checkpoint_directories=tuple(sorted(retained_names)),
        deleted_checkpoint_directories=tuple(deleted),
    )


def _model_version(path: Path) -> int:
    fields = path.name.split('-')
    is_single = len(fields) == 3 and fields[0] == 'checkpoint' and len(fields[2]) == 32
    is_distributed = len(fields) == 2 and fields[0] == 'distributed'
    if (not is_single and not is_distributed) or len(fields[1]) != 10:
        raise ValueError(f'Invalid checkpoint directory name: {path.name}.')
    try:
        model_version = int(fields[1])
        if is_single:
            int(fields[2], 16)
    except ValueError as error:
        raise ValueError(f'Invalid checkpoint directory name: {path.name}.') from error
    return model_version


def _validate_checkpoint_metadata(
    path: Path,
    checkpoint_repository_directory: Path,
) -> ValidatedCheckpointMetadata:
    if path.resolve().parent != checkpoint_repository_directory.resolve():
        raise ValueError('Checkpoint retention target escaped its repository.')
    distributed = path.name.startswith('distributed-')
    manifest_path = path / ('distributed-manifest.json' if distributed else 'manifest.json')
    if not manifest_path.is_file():
        raise ValueError(f'Checkpoint directory has no manifest: {path.name}.')
    manifest_contents = manifest_path.read_bytes()
    try:
        if distributed:
            manifest = DistributedCheckpointManifest.model_validate_json(manifest_contents)
        else:
            manifest = CheckpointManifest.model_validate_json(manifest_contents)
    except ValueError as error:
        raise ValueError(f'Checkpoint directory has an invalid manifest: {path.name}.') from error
    match manifest:
        case DistributedCheckpointManifest():
            model_version = manifest.model_version
            optimizer_steps = manifest.ranks[0].state.replay_credits.completed_optimizer_steps
            run_id = manifest.run_id
            expected_name = f'distributed-{model_version:010d}'
            for artifact in (manifest.model, manifest.optimizer, manifest.gradient_scaler):
                _validate_artifact(path, artifact)
            for rank in manifest.ranks:
                rank_directory = path / f'rank-{rank.rank:05d}'
                _validate_artifact(rank_directory, rank.torch_random_state)
                _validate_artifact(rank_directory, rank.cuda_random_stream)
        case CheckpointManifest():
            model_version = manifest.state.replay_credits.model_version
            optimizer_steps = manifest.state.replay_credits.completed_optimizer_steps
            run_id = manifest.run_id
            expected_name = f'checkpoint-{model_version:010d}-{manifest.checkpoint_id.hex}'
            for artifact in (
                manifest.model,
                manifest.optimizer,
                manifest.torch_random_state,
                manifest.cuda_random_stream,
                manifest.gradient_scaler,
            ):
                _validate_artifact(path, artifact)
    if path.name != expected_name or model_version != _model_version(path):
        raise ValueError('Checkpoint directory identity does not match its manifest.')
    return ValidatedCheckpointMetadata(
        path=path,
        run_id=run_id,
        model_version=model_version,
        completed_optimizer_steps=optimizer_steps,
        manifest_sha256=hashlib.sha256(manifest_contents).hexdigest(),
    )


def _validate_artifact(directory: Path, artifact: CheckpointArtifact) -> None:
    artifact_path = (directory / artifact.filename).resolve()
    if artifact_path.parent != directory.resolve() or not artifact_path.is_file():
        raise ValueError(f'Checkpoint retention artifact is missing: {artifact.kind.value}.')
    contents = artifact_path.read_bytes()
    if len(contents) != artifact.byte_count or hashlib.sha256(contents).hexdigest() != artifact.sha256:
        raise ValueError(f'Checkpoint retention artifact is corrupt: {artifact.kind.value}.')

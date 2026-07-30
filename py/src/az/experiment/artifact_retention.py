from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.az.config.runtime import RetentionConfiguration
from src.az.training.checkpoints import CheckpointManifest, CheckpointPointer


@dataclass(frozen=True)
class RetentionResult:
    retained_checkpoint_directories: tuple[str, ...]
    deleted_checkpoint_directories: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedCheckpointMetadata:
    path: Path
    manifest: CheckpointManifest
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
            (path for path in checkpoint_repository_directory.glob('checkpoint-??????????-*') if path.is_dir()),
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
        or current.manifest.run_id != pointer.run_id
        or current.manifest.state.replay_credits.model_version != pointer.model_version
    ):
        raise ValueError('Current checkpoint pointer does not authenticate its manifest identity.')
    recent = metadata[-configuration.recent_checkpoint_count :]
    retained_names = {item.path.name for item in recent}
    retained_names.add(pointer.checkpoint_directory)
    for item in metadata:
        optimizer_steps = item.manifest.state.replay_credits.completed_optimizer_steps
        if optimizer_steps % configuration.milestone_every_optimizer_steps == 0:
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
    if len(fields) != 3 or fields[0] != 'checkpoint' or len(fields[1]) != 10 or len(fields[2]) != 32:
        raise ValueError(f'Invalid checkpoint directory name: {path.name}.')
    try:
        model_version = int(fields[1])
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
    manifest_path = path / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Checkpoint directory has no manifest: {path.name}.')
    try:
        manifest_contents = manifest_path.read_bytes()
        manifest = CheckpointManifest.model_validate_json(manifest_contents)
    except ValueError as error:
        raise ValueError(f'Checkpoint directory has an invalid manifest: {path.name}.') from error
    expected_name = f'checkpoint-{manifest.state.replay_credits.model_version:010d}-{manifest.checkpoint_id.hex}'
    if path.name != expected_name or manifest.state.replay_credits.model_version != _model_version(path):
        raise ValueError('Checkpoint directory identity does not match its manifest.')
    return ValidatedCheckpointMetadata(
        path=path,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_contents).hexdigest(),
    )

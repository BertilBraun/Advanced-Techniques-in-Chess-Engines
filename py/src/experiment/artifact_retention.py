from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.train.TrainingArgs import ArtifactRetention
from src.util.save_paths import CheckpointManifest, load_checkpoint_manifest


CHECKPOINT_PATTERN = re.compile(r'^checkpoint_(\d+)\.json$')
RAW_MODEL_PATTERN = re.compile(r'^model_(\d+)\.pt$')
INFERENCE_MODEL_PATTERN = re.compile(r'^model_(\d+)\.jit\.pt$')
OPTIMIZER_PATTERN = re.compile(r'^optimizer_(\d+)\.pt$')
REPLAY_DIRECTORY_PATTERN = re.compile(r'^memory_(\d+)$')


@dataclass(frozen=True)
class RetentionResult:
    earliest_checkpoint_iteration: int
    earliest_replay_iteration: int
    deleted_checkpoint_files: int
    deleted_replay_directories: int


def _iteration_from_name(path: Path, pattern: re.Pattern[str]) -> int | None:
    match = pattern.fullmatch(path.name)
    return int(match.group(1)) if match is not None else None


def _replay_iteration(reference_path: str) -> int | None:
    replay_directory = Path(reference_path).parts[0]
    match = REPLAY_DIRECTORY_PATTERN.fullmatch(replay_directory)
    return int(match.group(1)) if match is not None else None


def _checkpoint_artifact_iteration(path: Path) -> int | None:
    for pattern in (CHECKPOINT_PATTERN, RAW_MODEL_PATTERN, OPTIMIZER_PATTERN):
        iteration = _iteration_from_name(path, pattern)
        if iteration is not None:
            return iteration
    return None


def _retain_inference_model(
    iteration: int,
    latest_checkpoint_iteration: int,
    retention: ArtifactRetention,
) -> bool:
    earliest_recent_iteration = max(
        0,
        latest_checkpoint_iteration - retention.recent_inference_checkpoint_count + 1,
    )
    return iteration >= earliest_recent_iteration or iteration % retention.milestone_inference_interval == 0


def _write_filtered_manifest(
    save_folder: Path,
    checkpoint_iteration: int,
    replay_window_iterations: int,
) -> None:
    manifest_path = save_folder / f'checkpoint_{checkpoint_iteration}.json'
    manifest = CheckpointManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    earliest_replay_iteration = max(0, checkpoint_iteration - replay_window_iterations)
    replay_files = tuple(
        reference
        for reference in manifest.replay_files
        if (iteration := _replay_iteration(reference.path)) is None
        or earliest_replay_iteration <= iteration <= checkpoint_iteration
    )
    filtered_manifest = manifest.model_copy(update={'replay_files': replay_files})
    temporary_path = manifest_path.with_name(f'.{manifest_path.name}.tmp')
    temporary_path.write_text(filtered_manifest.model_dump_json(indent=2) + '\n', encoding='utf-8')
    temporary_path.replace(manifest_path)


def apply_artifact_retention(
    save_folder: Path,
    latest_checkpoint_iteration: int,
    retention: ArtifactRetention,
) -> RetentionResult:
    if latest_checkpoint_iteration < 0:
        raise ValueError('Latest checkpoint iteration cannot be negative.')
    if retention.checkpoint_count <= 0:
        raise ValueError('Checkpoint retention count must be positive.')
    if retention.replay_window_iterations <= 0:
        raise ValueError('Replay window retention must be positive.')
    if retention.recent_inference_checkpoint_count <= 0:
        raise ValueError('Recent inference-checkpoint retention count must be positive.')
    if retention.milestone_inference_interval <= 0:
        raise ValueError('Milestone inference-checkpoint interval must be positive.')

    earliest_checkpoint_iteration = max(0, latest_checkpoint_iteration - retention.checkpoint_count + 1)
    retained_checkpoint_iterations = tuple(
        iteration
        for iteration in range(earliest_checkpoint_iteration, latest_checkpoint_iteration + 1)
        if (save_folder / f'checkpoint_{iteration}.json').is_file()
    )
    if latest_checkpoint_iteration not in retained_checkpoint_iterations:
        raise ValueError(f'Latest checkpoint manifest does not exist for iteration {latest_checkpoint_iteration}.')

    for checkpoint_iteration in retained_checkpoint_iterations:
        _write_filtered_manifest(
            save_folder,
            checkpoint_iteration,
            retention.replay_window_iterations,
        )

    deleted_checkpoint_files = 0
    for path in save_folder.iterdir():
        iteration = _checkpoint_artifact_iteration(path)
        if iteration is not None and iteration < earliest_checkpoint_iteration:
            path.unlink()
            deleted_checkpoint_files += 1
            continue
        inference_iteration = _iteration_from_name(path, INFERENCE_MODEL_PATTERN)
        if inference_iteration is not None and not _retain_inference_model(
            inference_iteration,
            latest_checkpoint_iteration,
            retention,
        ):
            path.unlink()
            deleted_checkpoint_files += 1

    earliest_replay_iteration = max(
        0,
        earliest_checkpoint_iteration - retention.replay_window_iterations,
    )
    deleted_replay_directories = 0
    for path in save_folder.iterdir():
        if not path.is_dir():
            continue
        iteration = _iteration_from_name(path, REPLAY_DIRECTORY_PATTERN)
        if iteration is not None and iteration < earliest_replay_iteration:
            shutil.rmtree(path)
            deleted_replay_directories += 1

    for checkpoint_iteration in retained_checkpoint_iterations:
        load_checkpoint_manifest(checkpoint_iteration, save_folder)

    return RetentionResult(
        earliest_checkpoint_iteration=earliest_checkpoint_iteration,
        earliest_replay_iteration=earliest_replay_iteration,
        deleted_checkpoint_files=deleted_checkpoint_files,
        deleted_replay_directories=deleted_replay_directories,
    )

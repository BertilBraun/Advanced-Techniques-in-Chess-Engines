from __future__ import annotations

import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from src.experiment.configuration import experiment_configuration_sha256, load_experiment_configuration
from src.experiment_queue.cgroup import CgroupV2MemoryScope
from src.experiment_queue.configuration import QueueConfiguration, QueuedExperiment, ResourceSlot


@dataclass(frozen=True)
class ValidatedQueuedExperiment:
    definition: QueuedExperiment
    configuration_sha256: str


@dataclass(frozen=True)
class ValidatedQueue:
    configuration: QueueConfiguration
    experiments: tuple[ValidatedQueuedExperiment, ...]
    fingerprint: str


def validate_queue_for_launch(configuration: QueueConfiguration) -> ValidatedQueue:
    if sys.platform != 'linux':
        raise ValueError('The experiment queue launcher supports Linux only.')

    validated_experiments = tuple(_validate_experiment(experiment) for experiment in configuration.experiments)
    for slot in configuration.slots:
        _validate_slot_paths(slot)
        _validate_runner_executable(configuration.runner.command[0], slot.working_directory)

    configuration.summary_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _queue_fingerprint(configuration, validated_experiments)
    return ValidatedQueue(
        configuration=configuration,
        experiments=validated_experiments,
        fingerprint=fingerprint,
    )


def _validate_experiment(experiment: QueuedExperiment) -> ValidatedQueuedExperiment:
    if not experiment.experiment_file.is_file():
        raise ValueError(f'Experiment file does not exist: {experiment.experiment_file}')
    configuration = load_experiment_configuration(experiment.experiment_file)
    return ValidatedQueuedExperiment(
        definition=experiment,
        configuration_sha256=experiment_configuration_sha256(configuration),
    )


def _validate_slot_paths(slot: ResourceSlot) -> None:
    if not slot.working_directory.is_dir():
        raise ValueError(f'Slot working directory does not exist: {slot.working_directory}')
    slot.log_directory.mkdir(parents=True, exist_ok=True)
    if not slot.log_directory.is_dir():
        raise ValueError(f'Slot log path is not a directory: {slot.log_directory}')
    memory_scope = CgroupV2MemoryScope(slot.cgroup_directory)
    memory_scope.prepare(slot.ram_capacity_bytes)
    memory_scope.validate_process_migration()


def _validate_runner_executable(executable: str, working_directory: Path) -> None:
    executable_path = Path(executable)
    if executable_path.is_absolute() or executable_path.parent != Path('.'):
        resolved_path = executable_path if executable_path.is_absolute() else working_directory / executable_path
        if not resolved_path.is_file():
            raise ValueError(f'Runner executable does not exist: {resolved_path}')
        return
    if shutil.which(executable) is None:
        raise ValueError(f'Runner executable is not available on PATH: {executable}')


def _queue_fingerprint(
    configuration: QueueConfiguration,
    experiments: tuple[ValidatedQueuedExperiment, ...],
) -> str:
    digest = hashlib.sha256(configuration.model_dump_json().encode('utf-8'))
    for experiment in experiments:
        digest.update(experiment.definition.experiment_id.encode('utf-8'))
        digest.update(experiment.configuration_sha256.encode('ascii'))
    return digest.hexdigest()

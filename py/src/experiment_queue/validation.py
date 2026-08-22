from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.experiment.configuration import experiment_configuration_sha256, load_experiment_configuration
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

    _validate_repository(configuration)
    validated_experiments = tuple(
        _validate_experiment(experiment, configuration.repository_directory) for experiment in configuration.experiments
    )
    for slot in configuration.slots:
        if slot.log_directory.is_relative_to(configuration.repository_directory) or slot.log_directory.is_relative_to(
            configuration.worktree_root
        ):
            raise ValueError('Queue logs must be outside the control checkout and disposable worktree root.')
        _validate_slot_paths(slot)
    _validate_runner_executable(configuration.runner.command[0], configuration.repository_directory)
    for setup_command in configuration.runner.setup_commands:
        _validate_runner_executable(setup_command[0], configuration.repository_directory)

    configuration.summary_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = queue_configuration_fingerprint(configuration)
    return ValidatedQueue(
        configuration=configuration,
        experiments=validated_experiments,
        fingerprint=fingerprint,
    )


def _validate_experiment(experiment: QueuedExperiment, repository_directory: Path) -> ValidatedQueuedExperiment:
    if not experiment.experiment_file.is_file():
        raise ValueError(f'Experiment file does not exist: {experiment.experiment_file}')
    if not experiment.experiment_file.is_relative_to(repository_directory):
        raise ValueError(f'Experiment file must be inside the source repository: {experiment.experiment_file}')
    resolved_revision = _git_output(repository_directory, ('rev-parse', f'{experiment.source_revision}^{{commit}}'))
    if resolved_revision != experiment.source_revision:
        raise ValueError(f'Experiment source revision is not an exact commit: {experiment.source_revision}')
    # The resolved configuration SHA below pins the whole extends chain, so per-file blob comparison adds nothing.
    configuration = load_experiment_configuration(experiment.experiment_file)
    return ValidatedQueuedExperiment(
        definition=experiment,
        configuration_sha256=experiment_configuration_sha256(configuration),
    )


def _validate_slot_paths(slot: ResourceSlot) -> None:
    slot.log_directory.mkdir(parents=True, exist_ok=True)
    if not slot.log_directory.is_dir():
        raise ValueError(f'Slot log path is not a directory: {slot.log_directory}')


def _validate_runner_executable(executable: str, working_directory: Path) -> None:
    executable_path = Path(executable)
    if executable_path.is_absolute() or executable_path.parent != Path('.'):
        resolved_path = executable_path if executable_path.is_absolute() else working_directory / executable_path
        if not resolved_path.is_file():
            raise ValueError(f'Runner executable does not exist: {resolved_path}')
        return
    if shutil.which(executable) is None:
        raise ValueError(f'Runner executable is not available on PATH: {executable}')


def queue_configuration_fingerprint(configuration: QueueConfiguration) -> str:
    runtime_identity = configuration.model_dump(
        mode='json',
        exclude={'runner', 'experiments', 'poll_interval_seconds', 'wait_for_updates_when_empty'},
    )
    serialized = json.dumps(runtime_identity, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _validate_repository(configuration: QueueConfiguration) -> None:
    if not configuration.repository_directory.is_dir():
        raise ValueError(f'Repository directory does not exist: {configuration.repository_directory}')
    if _git_output(configuration.repository_directory, ('rev-parse', '--show-toplevel')) != str(
        configuration.repository_directory
    ):
        raise ValueError('Configured repository directory is not a Git worktree root.')
    external_roots = (
        configuration.worktree_root,
        configuration.runtime_directory,
        configuration.tensorboard_log_directory,
    )
    if any(path.is_relative_to(configuration.repository_directory) for path in external_roots):
        raise ValueError('Worktree, runtime, and TensorBoard roots must be outside the control checkout.')
    if configuration.runtime_directory.is_relative_to(
        configuration.worktree_root
    ) or configuration.worktree_root.is_relative_to(configuration.runtime_directory):
        raise ValueError('Persistent runtime and disposable worktree roots must not contain one another.')
    if configuration.tensorboard_log_directory.is_relative_to(configuration.worktree_root):
        raise ValueError('Persistent TensorBoard logs must be outside the disposable worktree root.')
    for path in external_roots:
        path.mkdir(parents=True, exist_ok=True)


def _git_output(repository_directory: Path, arguments: tuple[str, ...]) -> str:
    result = subprocess.run(
        ('git', '-C', str(repository_directory), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.experiment.configuration import (
    experiment_configuration_sha256,
    experiment_configuration_source_paths,
    load_experiment_configuration,
)
from src.experiment_queue.configuration import QueueConfiguration
from src.experiment_queue.validation import ValidatedQueuedExperiment
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class PreservedExecutionWorkspace(FrozenModel):
    experiment_id: str
    source_revision: str
    configuration_sha256: str
    source_worktree: Path
    runtime_directory: Path
    configuration_directory: Path


@dataclass(frozen=True)
class ExperimentWorkspace:
    source_worktree: Path
    runtime_directory: Path
    tensorboard_log_directory: Path
    experiment_file: Path
    preserved_configuration_directory: Path


class ExperimentWorkspaceManager:
    def __init__(self, configuration: QueueConfiguration) -> None:
        self._configuration = configuration

    def create(self, experiment: ValidatedQueuedExperiment) -> ExperimentWorkspace:
        definition = experiment.definition
        source_worktree = self._configuration.worktree_root / definition.experiment_id
        runtime_directory = self._configuration.runtime_directory
        preserved_directory = runtime_directory / '.queue-evidence' / definition.experiment_id / 'configuration'
        if source_worktree.exists():
            raise ValueError(f'Experiment worktree already exists: {source_worktree}')
        self._configuration.worktree_root.mkdir(parents=True, exist_ok=True)
        runtime_directory.mkdir(parents=True, exist_ok=True)
        self._configuration.tensorboard_log_directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            (
                'git',
                '-C',
                str(self._configuration.repository_directory),
                'worktree',
                'add',
                '--detach',
                str(source_worktree),
                definition.source_revision,
            ),
            check=True,
        )
        experiment_file = source_worktree / definition.experiment_file.relative_to(
            self._configuration.repository_directory
        )
        worktree_configuration = load_experiment_configuration(experiment_file)
        if experiment_configuration_sha256(worktree_configuration) != experiment.configuration_sha256:
            raise ValueError('Created worktree configuration does not match the validated queue entry.')
        return ExperimentWorkspace(
            source_worktree=source_worktree,
            runtime_directory=runtime_directory,
            tensorboard_log_directory=self._configuration.tensorboard_log_directory,
            experiment_file=experiment_file,
            preserved_configuration_directory=preserved_directory,
        )

    def preserve_and_remove(
        self,
        experiment: ValidatedQueuedExperiment,
        workspace: ExperimentWorkspace,
    ) -> None:
        workspace.preserved_configuration_directory.mkdir(parents=True, exist_ok=False)
        for source_path in experiment_configuration_source_paths(workspace.experiment_file):
            relative_path = source_path.relative_to(workspace.source_worktree)
            destination = workspace.preserved_configuration_directory / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
        evidence = PreservedExecutionWorkspace(
            experiment_id=experiment.definition.experiment_id,
            source_revision=experiment.definition.source_revision,
            configuration_sha256=experiment.configuration_sha256,
            source_worktree=workspace.source_worktree,
            runtime_directory=workspace.runtime_directory,
            configuration_directory=workspace.preserved_configuration_directory,
        )
        write_text_atomically(
            workspace.preserved_configuration_directory.parent / 'workspace.json',
            evidence.model_dump_json(indent=2) + '\n',
        )
        subprocess.run(
            (
                'git',
                '-C',
                str(self._configuration.repository_directory),
                'worktree',
                'remove',
                '--force',
                str(workspace.source_worktree),
            ),
            check=True,
        )

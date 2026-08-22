from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import sys

import pytest

from src.experiment.configuration import experiment_configuration_sha256, load_experiment_configuration
from src.experiment_queue.configuration import (
    QueueConfiguration,
    QueuedExperiment,
    ResourceRequest,
    ResourceSlot,
    RunnerCommand,
)
from src.experiment_queue.validation import ValidatedQueuedExperiment
from src.experiment_queue.workspace import ExperimentWorkspaceManager
from test_helpers.configuration_paths import PYTHON_ROOT, TEST_CONFIG_DIRECTORY


TEMPLATE = TEST_CONFIG_DIRECTORY / 'go-7x7-experiment.yaml'
WORKTREE_CHILD = PYTHON_ROOT / 'src' / 'experiment_queue' / 'worktree_child.py'


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ('git', '-C', str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, Path, str]:
    repository = tmp_path / 'repository'
    repository.mkdir()
    experiment_file = repository / 'experiment.yaml'
    shutil.copy2(TEMPLATE, experiment_file)
    _git(repository, 'init')
    _git(repository, 'config', 'user.name', 'Test')
    _git(repository, 'config', 'user.email', 'test@example.com')
    _git(repository, 'add', 'experiment.yaml')
    _git(repository, 'commit', '-m', 'initial')
    return repository, experiment_file, _git(repository, 'rev-parse', 'HEAD')


def _configuration(tmp_path: Path, repository: Path, experiment: QueuedExperiment) -> QueueConfiguration:
    return QueueConfiguration(
        runner=RunnerCommand(command=('python',)),
        repository_directory=repository,
        worktree_root=tmp_path / 'worktrees',
        runtime_directory=tmp_path / 'runtime',
        tensorboard_log_directory=tmp_path / 'tensorboard',
        slots=(
            ResourceSlot(
                slot_id='slot',
                cuda_devices=(),
                cpu_affinity=(0,),
                ram_capacity_bytes=1_000,
                log_directory=tmp_path / 'logs',
            ),
        ),
        experiments=(experiment,),
        summary_path=tmp_path / 'summary.json',
    )


def _validated(experiment: QueuedExperiment) -> ValidatedQueuedExperiment:
    configuration = load_experiment_configuration(experiment.experiment_file)
    return ValidatedQueuedExperiment(
        definition=experiment,
        configuration_sha256=experiment_configuration_sha256(configuration),
    )


def test_worktrees_are_exact_revision_isolated_and_outputs_are_central(tmp_path: Path) -> None:
    repository, experiment_file, first_revision = _repository(tmp_path)
    experiment = QueuedExperiment(
        experiment_id='first',
        experiment_file=experiment_file,
        source_revision=first_revision,
        resources=ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=500),
    )
    manager = ExperimentWorkspaceManager(_configuration(tmp_path, repository, experiment))
    workspace = manager.create(_validated(experiment))

    assert _git(workspace.source_worktree, 'rev-parse', 'HEAD') == first_revision
    assert workspace.source_worktree.parent == tmp_path / 'worktrees'
    assert workspace.runtime_directory == tmp_path / 'runtime'
    assert workspace.tensorboard_log_directory == tmp_path / 'tensorboard'
    assert workspace.runtime_directory.is_relative_to(workspace.source_worktree) is False


def test_success_preserves_configuration_then_removes_worktree(tmp_path: Path) -> None:
    repository, experiment_file, revision = _repository(tmp_path)
    experiment = QueuedExperiment(
        experiment_id='success',
        experiment_file=experiment_file,
        source_revision=revision,
        resources=ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=500),
    )
    manager = ExperimentWorkspaceManager(_configuration(tmp_path, repository, experiment))
    validated = _validated(experiment)
    workspace = manager.create(validated)

    manager.preserve_and_remove(validated, workspace)

    assert not workspace.source_worktree.exists()
    assert (workspace.preserved_configuration_directory / 'experiment.yaml').is_file()
    assert (workspace.preserved_configuration_directory.parent / 'workspace.json').is_file()


def test_failed_run_worktree_is_retained_for_diagnosis(tmp_path: Path) -> None:
    repository, experiment_file, revision = _repository(tmp_path)
    experiment = QueuedExperiment(
        experiment_id='failure',
        experiment_file=experiment_file,
        source_revision=revision,
        resources=ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=500),
    )
    manager = ExperimentWorkspaceManager(_configuration(tmp_path, repository, experiment))
    workspace = manager.create(_validated(experiment))

    assert workspace.source_worktree.is_dir()
    assert not workspace.preserved_configuration_directory.exists()


def test_worktree_creation_rejects_a_revision_configuration_mismatch(tmp_path: Path) -> None:
    repository, experiment_file, revision = _repository(tmp_path)
    content = experiment_file.read_text(encoding='utf-8').replace('go-7x7-template', 'changed', 1)
    experiment_file.write_text(content, encoding='utf-8')
    experiment = QueuedExperiment(
        experiment_id='mismatch',
        experiment_file=experiment_file,
        source_revision=revision,
        resources=ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=500),
    )
    manager = ExperimentWorkspaceManager(_configuration(tmp_path, repository, experiment))

    with pytest.raises(ValueError, match='does not match'):
        manager.create(_validated(experiment))


def test_worktree_child_runs_setup_in_source_and_runner_in_central_runtime(tmp_path: Path) -> None:
    source_worktree = tmp_path / 'source'
    runtime_directory = tmp_path / 'runtime'
    source_worktree.mkdir()
    runtime_directory.mkdir()
    setup_marker = source_worktree / 'setup'
    runtime_marker = runtime_directory / 'runner'
    setup_command = (
        sys.executable,
        '-c',
        f'from pathlib import Path; Path({str(setup_marker)!r}).write_text("ready")',
    )
    runner_command = (
        sys.executable,
        '-c',
        f'from pathlib import Path; Path({str(runtime_marker)!r}).write_text(str(Path.cwd()))',
    )

    subprocess.run(
        (
            sys.executable,
            str(WORKTREE_CHILD),
            '--source-worktree',
            str(source_worktree),
            '--runtime-directory',
            str(runtime_directory),
            '--setup-commands',
            json.dumps((setup_command,)),
            '--',
            *runner_command,
        ),
        check=True,
    )

    assert setup_marker.read_text(encoding='utf-8') == 'ready'
    assert Path(runtime_marker.read_text(encoding='utf-8')) == runtime_directory


def test_worktree_child_setup_failure_does_not_start_runner(tmp_path: Path) -> None:
    source_worktree = tmp_path / 'source'
    runtime_directory = tmp_path / 'runtime'
    source_worktree.mkdir()
    runtime_directory.mkdir()
    runner_marker = runtime_directory / 'runner'

    result = subprocess.run(
        (
            sys.executable,
            str(WORKTREE_CHILD),
            '--source-worktree',
            str(source_worktree),
            '--runtime-directory',
            str(runtime_directory),
            '--setup-commands',
            json.dumps(((sys.executable, '-c', 'raise SystemExit(9)'),)),
            '--',
            sys.executable,
            '-c',
            f'from pathlib import Path; Path({str(runner_marker)!r}).write_text("started")',
        ),
        check=False,
    )

    assert result.returncode != 0
    assert not runner_marker.exists()

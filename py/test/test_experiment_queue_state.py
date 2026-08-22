from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from src.experiment_queue.configuration import (
    QueueConfiguration,
    QueuedExperiment,
    ResourceRequest,
    ResourceSlot,
    RunnerCommand,
)
from src.experiment_queue.runner import ExperimentQueueRunner
from src.experiment_queue.scheduler import create_assignment
from src.experiment_queue.state import ExecutionIdentity, QueueSummary, RunningExperimentStatus, write_queue_summary
from src.experiment_queue.validation import ValidatedQueue, ValidatedQueuedExperiment
from src.experiment_queue.workspace import ExperimentWorkspaceManager


def test_restart_marks_prior_running_process_failed_without_adopting_it(tmp_path: Path) -> None:
    experiment = QueuedExperiment(
        experiment_id='experiment',
        experiment_file=tmp_path / 'experiment.yaml',
        source_revision='1' * 40,
        resources=ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_000),
    )
    slot = ResourceSlot(
        slot_id='slot',
        cuda_devices=(),
        cpu_affinity=(0,),
        ram_capacity_bytes=2_000,
        log_directory=tmp_path,
    )
    summary_path = tmp_path / 'summary.json'
    configuration = QueueConfiguration(
        runner=RunnerCommand(command=('python',)),
        repository_directory=tmp_path,
        worktree_root=tmp_path / 'worktrees',
        runtime_directory=tmp_path / 'runtime',
        tensorboard_log_directory=tmp_path / 'tensorboard',
        slots=(slot,),
        experiments=(experiment,),
        summary_path=summary_path,
    )
    fingerprint = 'a' * 64
    timestamp = datetime.now(timezone.utc)
    execution = ExecutionIdentity(
        configuration_sha256='b' * 64,
        source_revision='1' * 40,
        setup_commands=(),
        source_worktree=tmp_path / 'worktree',
        runtime_directory=tmp_path / 'runtime',
        preserved_configuration_directory=tmp_path / 'evidence',
        preserved_experiment_file=tmp_path / 'evidence' / 'experiment.yaml',
        command=('python', 'experiment.yaml'),
        assignment=create_assignment(experiment, slot),
        started_at=timestamp,
        pid=123,
        process_group_id=123,
        stdout_log=tmp_path / 'stdout.log',
        stderr_log=tmp_path / 'stderr.log',
    )
    write_queue_summary(
        summary_path,
        QueueSummary(
            queue_fingerprint=fingerprint,
            created_at=timestamp,
            updated_at=timestamp,
            experiments=(RunningExperimentStatus(experiment_id='experiment', execution=execution),),
        ),
    )
    queue = ValidatedQueue(
        configuration=configuration,
        experiments=(ValidatedQueuedExperiment(definition=experiment, configuration_sha256='b' * 64),),
        fingerprint=fingerprint,
    )

    runner = ExperimentQueueRunner(lambda: queue)

    assert runner.summary.experiments[0].status == 'failed'
    assert runner.summary.experiments[0].exit_code is None
    assert 'not recovered or adopted' in runner.summary.experiments[0].reason
    with pytest.raises(ValueError, match='Verify those process groups'):
        runner.run()


def test_pending_experiments_reload_configuration_and_desired_order(tmp_path: Path) -> None:
    slot = ResourceSlot(
        slot_id='slot',
        cuda_devices=(),
        cpu_affinity=(0,),
        ram_capacity_bytes=2_000,
        log_directory=tmp_path,
    )
    request = ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_000)
    first = QueuedExperiment(
        experiment_id='first', experiment_file=tmp_path / 'first.yaml', source_revision='1' * 40, resources=request
    )
    removed = QueuedExperiment(
        experiment_id='removed', experiment_file=tmp_path / 'removed.yaml', source_revision='2' * 40, resources=request
    )
    added = QueuedExperiment(
        experiment_id='added', experiment_file=tmp_path / 'added.yaml', source_revision='3' * 40, resources=request
    )
    initial_configuration = QueueConfiguration(
        runner=RunnerCommand(command=('python',)),
        repository_directory=tmp_path,
        worktree_root=tmp_path / 'worktrees',
        runtime_directory=tmp_path / 'runtime',
        tensorboard_log_directory=tmp_path / 'tensorboard',
        slots=(slot,),
        experiments=(first, removed),
        summary_path=tmp_path / 'summary.json',
    )
    refreshed_configuration = initial_configuration.model_copy(update={'experiments': (added, first)})
    initial = ValidatedQueue(
        configuration=initial_configuration,
        experiments=(
            ValidatedQueuedExperiment(definition=first, configuration_sha256='1' * 64),
            ValidatedQueuedExperiment(definition=removed, configuration_sha256='2' * 64),
        ),
        fingerprint='a' * 64,
    )
    refreshed = ValidatedQueue(
        configuration=refreshed_configuration,
        experiments=(
            ValidatedQueuedExperiment(definition=added, configuration_sha256='3' * 64),
            ValidatedQueuedExperiment(definition=first, configuration_sha256='4' * 64),
        ),
        fingerprint='a' * 64,
    )
    desired = [initial]
    runner = ExperimentQueueRunner(lambda: desired[0])

    desired[0] = refreshed
    runner.refresh_configuration()

    assert runner.configuration_reload_error is None
    assert tuple(status.experiment_id for status in runner.summary.experiments) == ('added', 'first')
    assert tuple(status.configuration_sha256 for status in runner.summary.experiments) == ('3' * 64, '4' * 64)


def test_terminal_experiment_configuration_change_is_rejected(tmp_path: Path) -> None:
    experiment = QueuedExperiment(
        experiment_id='running',
        experiment_file=tmp_path / 'experiment.yaml',
        source_revision='1' * 40,
        resources=ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_000),
    )
    slot = ResourceSlot(
        slot_id='slot',
        cuda_devices=(),
        cpu_affinity=(0,),
        ram_capacity_bytes=2_000,
        log_directory=tmp_path,
    )
    configuration = QueueConfiguration(
        runner=RunnerCommand(command=('python',)),
        repository_directory=tmp_path,
        worktree_root=tmp_path / 'worktrees',
        runtime_directory=tmp_path / 'runtime',
        tensorboard_log_directory=tmp_path / 'tensorboard',
        slots=(slot,),
        experiments=(experiment,),
        summary_path=tmp_path / 'summary.json',
    )
    timestamp = datetime.now(timezone.utc)
    write_queue_summary(
        configuration.summary_path,
        QueueSummary(
            queue_fingerprint='a' * 64,
            created_at=timestamp,
            updated_at=timestamp,
            experiments=(
                RunningExperimentStatus(
                    experiment_id='running',
                    execution=ExecutionIdentity(
                        configuration_sha256='1' * 64,
                        source_revision='1' * 40,
                        setup_commands=(),
                        source_worktree=tmp_path / 'worktree',
                        runtime_directory=tmp_path / 'runtime',
                        preserved_configuration_directory=tmp_path / 'evidence',
                        preserved_experiment_file=tmp_path / 'evidence' / 'experiment.yaml',
                        command=('python', 'experiment.yaml'),
                        assignment=create_assignment(experiment, slot),
                        started_at=timestamp,
                        pid=123,
                        process_group_id=123,
                        stdout_log=tmp_path / 'stdout.log',
                        stderr_log=tmp_path / 'stderr.log',
                    ),
                ),
            ),
        ),
    )
    queue = ValidatedQueue(
        configuration=configuration,
        experiments=(ValidatedQueuedExperiment(definition=experiment, configuration_sha256='2' * 64),),
        fingerprint='a' * 64,
    )
    runner = ExperimentQueueRunner(lambda: queue)

    runner.refresh_configuration()

    assert runner.configuration_reload_error is not None
    assert 'cannot change identity' in runner.configuration_reload_error
    assert runner.summary.experiments[0].status == 'failed'


def test_pending_experiment_source_revision_can_change_before_launch(tmp_path: Path) -> None:
    request = ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_000)
    initial_experiment = QueuedExperiment(
        experiment_id='pending',
        experiment_file=tmp_path / 'experiment.yaml',
        source_revision='1' * 40,
        resources=request,
    )
    updated_experiment = initial_experiment.validated_copy(update={'source_revision': '2' * 40})
    slot = ResourceSlot(
        slot_id='slot',
        cuda_devices=(),
        cpu_affinity=(0,),
        ram_capacity_bytes=2_000,
        log_directory=tmp_path / 'logs',
    )
    initial_configuration = QueueConfiguration(
        runner=RunnerCommand(command=('python',)),
        repository_directory=tmp_path,
        worktree_root=tmp_path / 'worktrees',
        runtime_directory=tmp_path / 'runtime',
        tensorboard_log_directory=tmp_path / 'tensorboard',
        slots=(slot,),
        experiments=(initial_experiment,),
        summary_path=tmp_path / 'summary.json',
    )
    updated_configuration = initial_configuration.validated_copy(update={'experiments': (updated_experiment,)})
    initial = ValidatedQueue(
        configuration=initial_configuration,
        experiments=(ValidatedQueuedExperiment(definition=initial_experiment, configuration_sha256='3' * 64),),
        fingerprint='a' * 64,
    )
    updated = ValidatedQueue(
        configuration=updated_configuration,
        experiments=(ValidatedQueuedExperiment(definition=updated_experiment, configuration_sha256='3' * 64),),
        fingerprint='a' * 64,
    )
    desired = [initial]
    runner = ExperimentQueueRunner(lambda: desired[0])

    desired[0] = updated
    runner.refresh_configuration()

    status = runner.summary.experiments[0]
    assert status.status == 'pending'
    assert status.source_revision == '2' * 40


def test_worktree_preparation_failure_is_terminal_and_does_not_crash_queue(tmp_path: Path) -> None:
    experiment = QueuedExperiment(
        experiment_id='preparation-failure',
        experiment_file=tmp_path / 'experiment.yaml',
        source_revision='1' * 40,
        resources=ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_000),
    )
    slot = ResourceSlot(
        slot_id='slot',
        cuda_devices=(),
        cpu_affinity=(0,),
        ram_capacity_bytes=2_000,
        log_directory=tmp_path / 'logs',
    )
    configuration = QueueConfiguration(
        runner=RunnerCommand(command=('python',)),
        repository_directory=tmp_path,
        worktree_root=tmp_path / 'worktrees',
        runtime_directory=tmp_path / 'runtime',
        tensorboard_log_directory=tmp_path / 'tensorboard',
        slots=(slot,),
        experiments=(experiment,),
        summary_path=tmp_path / 'summary.json',
    )
    queue = ValidatedQueue(
        configuration=configuration,
        experiments=(ValidatedQueuedExperiment(definition=experiment, configuration_sha256='2' * 64),),
        fingerprint='a' * 64,
    )
    runner = ExperimentQueueRunner(lambda: queue)

    with patch.object(ExperimentWorkspaceManager, 'create', side_effect=ValueError('cannot create worktree')):
        runner._launch_ready_experiments()

    status = runner.summary.experiments[0]
    assert status.status == 'preparation_failed'
    assert 'cannot create worktree' in status.reason
    assert runner.active_process_count == 0

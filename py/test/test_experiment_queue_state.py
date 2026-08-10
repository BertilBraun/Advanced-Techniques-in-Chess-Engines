from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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


def test_restart_marks_prior_running_process_failed_without_adopting_it(tmp_path: Path) -> None:
    experiment = QueuedExperiment(
        experiment_id='experiment',
        experiment_file=tmp_path / 'experiment.yaml',
        resources=ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_000),
    )
    slot = ResourceSlot(
        slot_id='slot',
        cuda_devices=(),
        cpu_affinity=(0,),
        ram_capacity_bytes=2_000,
        working_directory=tmp_path,
        log_directory=tmp_path,
    )
    summary_path = tmp_path / 'summary.json'
    configuration = QueueConfiguration(
        runner=RunnerCommand(command=('python',)),
        slots=(slot,),
        experiments=(experiment,),
        summary_path=summary_path,
    )
    fingerprint = 'a' * 64
    timestamp = datetime.now(timezone.utc)
    execution = ExecutionIdentity(
        configuration_sha256='b' * 64,
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
        working_directory=tmp_path,
        log_directory=tmp_path,
    )
    request = ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_000)
    first = QueuedExperiment(experiment_id='first', experiment_file=tmp_path / 'first.yaml', resources=request)
    removed = QueuedExperiment(experiment_id='removed', experiment_file=tmp_path / 'removed.yaml', resources=request)
    added = QueuedExperiment(experiment_id='added', experiment_file=tmp_path / 'added.yaml', resources=request)
    initial_configuration = QueueConfiguration(
        runner=RunnerCommand(command=('python',)),
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
        resources=ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_000),
    )
    slot = ResourceSlot(
        slot_id='slot',
        cuda_devices=(),
        cpu_affinity=(0,),
        ram_capacity_bytes=2_000,
        working_directory=tmp_path,
        log_directory=tmp_path,
    )
    configuration = QueueConfiguration(
        runner=RunnerCommand(command=('python',)),
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
    assert 'cannot change configuration' in runner.configuration_reload_error
    assert runner.summary.experiments[0].status == 'failed'

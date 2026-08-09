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
        cgroup_directory=tmp_path / 'cgroup',
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

    runner = ExperimentQueueRunner(queue)

    assert runner.summary.experiments[0].status == 'failed'
    assert runner.summary.experiments[0].exit_code is None
    assert 'not recovered or adopted' in runner.summary.experiments[0].reason
    with pytest.raises(ValueError, match='Verify those process groups'):
        runner.run()

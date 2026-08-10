from __future__ import annotations

import json
import os
import signal
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

from src.experiment_queue.configuration import (
    QueueConfiguration,
    QueuedExperiment,
    ResourceRequest,
    ResourceSlot,
    RunnerCommand,
)
from src.experiment_queue.process import launch_process, terminate_process_group
from src.experiment_queue.runner import ExperimentQueueRunner
from src.experiment_queue.scheduler import ResourceAssignment
from src.experiment_queue.state import CompletedExperimentStatus, FailedExperimentStatus, load_queue_summary
from src.experiment_queue.validation import validate_queue_for_launch


pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux process controls are required.')

EXPERIMENT_TEMPLATE = Path(__file__).parents[1] / 'configs' / 'go-7x7-experiment-template.yaml'


def _available_cpu_cores(count: int) -> tuple[int, ...]:
    available = tuple(sorted(os.sched_getaffinity(0)))
    assert len(available) >= count
    return available[:count]


def _assignment(temporary_directory: Path, ram_limit_bytes: int = 2_000_000_000) -> ResourceAssignment:
    return ResourceAssignment(
        slot_id='test-slot',
        cuda_devices=(3, 5),
        cpu_affinity=_available_cpu_cores(1),
        ram_limit_bytes=ram_limit_bytes,
        working_directory=temporary_directory,
        log_directory=temporary_directory,
    )


def test_linux_launcher_applies_cuda_affinity_process_group_and_logs(tmp_path: Path) -> None:
    assignment = _assignment(tmp_path)
    stdout_path = tmp_path / 'resources.stdout.log'
    stderr_path = tmp_path / 'resources.stderr.log'
    script = (
        'import json, os; '
        "print(json.dumps({'cuda': os.environ['CUDA_VISIBLE_DEVICES'], "
        "'affinity': sorted(os.sched_getaffinity(0))}))"
    )

    running_process = launch_process(
        experiment_id='resources',
        command=(sys.executable, '-c', script),
        assignment=assignment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    assert os.getpgid(running_process.process.pid) == running_process.process.pid
    assert running_process.process.wait(timeout=10.0) == 0
    running_process.close_logs()

    observed = json.loads(stdout_path.read_text(encoding='utf-8'))
    assert observed == {
        'cuda': '3,5',
        'affinity': list(assignment.cpu_affinity),
    }
    assert stderr_path.read_text(encoding='utf-8') == ''


def test_linux_launcher_terminates_the_complete_process_group(tmp_path: Path) -> None:
    ready_path = tmp_path / 'child-ready'
    terminated_path = tmp_path / 'child-terminated'
    child_script = (
        'import pathlib, signal, sys, time; '
        'ready=pathlib.Path(sys.argv[1]); terminated=pathlib.Path(sys.argv[2]); '
        "signal.signal(signal.SIGTERM, lambda *_: (terminated.write_text('terminated'), sys.exit(0))); "
        "ready.write_text('ready'); time.sleep(60)"
    )
    parent_script = (
        'import subprocess, sys, time; '
        f"subprocess.Popen([sys.executable, '-c', {child_script!r}, {str(ready_path)!r}, {str(terminated_path)!r}]); "
        'time.sleep(60)'
    )
    running_process = launch_process(
        experiment_id='termination',
        command=(sys.executable, '-c', parent_script),
        assignment=_assignment(tmp_path),
        stdout_path=tmp_path / 'termination.stdout.log',
        stderr_path=tmp_path / 'termination.stderr.log',
    )
    deadline = time.monotonic() + 10.0
    while not ready_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready_path.exists()

    exit_code = terminate_process_group(running_process, grace_seconds=2.0)
    running_process.close_logs()

    assert exit_code == -15
    assert terminated_path.read_text(encoding='utf-8') == 'terminated'


def test_queue_releases_slot_after_success_and_failure_and_runs_next_job(tmp_path: Path) -> None:
    experiment_paths = tuple(tmp_path / name for name in ('success-one.yaml', 'failure.yaml', 'success-two.yaml'))
    for experiment_path in experiment_paths:
        shutil.copyfile(EXPERIMENT_TEMPLATE, experiment_path)
    script = (
        'import pathlib, sys, time; '
        'experiment=pathlib.Path(sys.argv[-1]); '
        'print(experiment.name, flush=True); '
        'time.sleep(0.05); '
        "raise SystemExit(7 if 'failure' in experiment.name else 0)"
    )
    slot = ResourceSlot(
        slot_id='only-slot',
        cuda_devices=(),
        cpu_affinity=_available_cpu_cores(1),
        ram_capacity_bytes=2_000_000_000,
        working_directory=tmp_path,
        log_directory=tmp_path / 'logs',
    )
    request = ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_500_000_000)
    configuration = QueueConfiguration(
        runner=RunnerCommand(command=(sys.executable, '-c', script)),
        slots=(slot,),
        experiments=tuple(
            QueuedExperiment(
                experiment_id=f'experiment-{index}',
                experiment_file=experiment_path,
                resources=request,
            )
            for index, experiment_path in enumerate(experiment_paths)
        ),
        summary_path=tmp_path / 'queue-summary.json',
        poll_interval_seconds=0.01,
        termination_grace_seconds=1.0,
    )

    validated_queue = validate_queue_for_launch(configuration)
    summary = ExperimentQueueRunner(lambda: validated_queue).run()

    assert tuple(status.status for status in summary.experiments) == ('completed', 'failed', 'completed')
    first, second, third = summary.experiments
    assert isinstance(first, CompletedExperimentStatus)
    assert isinstance(second, FailedExperimentStatus)
    assert isinstance(third, CompletedExperimentStatus)
    assert second.exit_code == 7
    assert second.execution.started_at >= first.finished_at
    assert third.execution.started_at >= second.finished_at
    assert tuple(status.execution.assignment.slot_id for status in summary.experiments) == ('only-slot',) * 3
    assert load_queue_summary(configuration.summary_path) == summary
    assert tuple(
        status.execution.stdout_log.read_text(encoding='utf-8').strip() for status in summary.experiments
    ) == tuple(path.name for path in experiment_paths)


def test_queue_reloads_changed_pending_experiment_before_launch(tmp_path: Path) -> None:
    first_path = tmp_path / 'first.yaml'
    pending_path = tmp_path / 'pending.yaml'
    shutil.copyfile(EXPERIMENT_TEMPLATE, first_path)
    shutil.copyfile(EXPERIMENT_TEMPLATE, pending_path)
    script = (
        'import pathlib, sys, time; '
        'path=pathlib.Path(sys.argv[-1]); '
        "time.sleep(0.3) if path.name == 'first.yaml' else None; "
        "print('updated' if 'updated-pending' in path.read_text() else 'original', flush=True)"
    )
    slot = ResourceSlot(
        slot_id='only-slot',
        cuda_devices=(),
        cpu_affinity=_available_cpu_cores(1),
        ram_capacity_bytes=2_000_000_000,
        working_directory=tmp_path,
        log_directory=tmp_path / 'logs',
    )
    request = ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_500_000_000)
    configuration = QueueConfiguration(
        runner=RunnerCommand(command=(sys.executable, '-c', script)),
        slots=(slot,),
        experiments=(
            QueuedExperiment(experiment_id='first', experiment_file=first_path, resources=request),
            QueuedExperiment(experiment_id='pending', experiment_file=pending_path, resources=request),
        ),
        summary_path=tmp_path / 'queue-summary.json',
        poll_interval_seconds=0.01,
        termination_grace_seconds=1.0,
    )

    def update_pending_experiment() -> None:
        content = pending_path.read_text(encoding='utf-8')
        pending_path.write_text(content.replace('go-7x7-template', 'updated-pending', 1), encoding='utf-8')

    update = threading.Timer(0.1, update_pending_experiment)
    update.start()
    try:
        runner = ExperimentQueueRunner(lambda: validate_queue_for_launch(configuration))
        summary = runner.run()
    finally:
        update.cancel()

    assert tuple(status.status for status in summary.experiments) == ('completed', 'completed')
    pending_status = next(status for status in summary.experiments if status.experiment_id == 'pending')
    assert pending_status.execution.stdout_log.read_text(encoding='utf-8').strip() == 'updated'
    assert (
        pending_status.execution.configuration_sha256
        == validate_queue_for_launch(configuration).experiments[1].configuration_sha256
    )


def test_empty_queue_waits_for_new_desired_experiment(tmp_path: Path) -> None:
    first_path = tmp_path / 'first.yaml'
    added_path = tmp_path / 'added.yaml'
    shutil.copyfile(EXPERIMENT_TEMPLATE, first_path)
    shutil.copyfile(EXPERIMENT_TEMPLATE, added_path)
    slot = ResourceSlot(
        slot_id='only-slot',
        cuda_devices=(),
        cpu_affinity=_available_cpu_cores(1),
        ram_capacity_bytes=2_000_000_000,
        working_directory=tmp_path,
        log_directory=tmp_path / 'logs',
    )
    request = ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_500_000_000)
    first = QueuedExperiment(experiment_id='first', experiment_file=first_path, resources=request)
    added = QueuedExperiment(experiment_id='added', experiment_file=added_path, resources=request)
    initial_configuration = QueueConfiguration(
        runner=RunnerCommand(command=(sys.executable, '-c', "print('completed')")),
        slots=(slot,),
        experiments=(first,),
        summary_path=tmp_path / 'queue-summary.json',
        poll_interval_seconds=0.01,
        termination_grace_seconds=1.0,
        wait_for_updates_when_empty=True,
    )
    final_configuration = initial_configuration.model_copy(
        update={
            'runner': RunnerCommand(command=(sys.executable, '-c', "print('updated-command')")),
            'experiments': (first, added),
            'wait_for_updates_when_empty': False,
        }
    )
    desired = [validate_queue_for_launch(initial_configuration)]
    update = threading.Timer(0.2, lambda: desired.__setitem__(0, validate_queue_for_launch(final_configuration)))
    update.start()
    try:
        summary = ExperimentQueueRunner(lambda: desired[0]).run()
    finally:
        update.cancel()

    assert tuple(status.experiment_id for status in summary.experiments) == ('first', 'added')
    assert tuple(status.status for status in summary.experiments) == ('completed', 'completed')
    added_status = next(status for status in summary.experiments if status.experiment_id == 'added')
    assert added_status.execution.stdout_log.read_text(encoding='utf-8').strip() == 'updated-command'


def test_queue_terminates_a_process_tree_over_its_rss_limit(tmp_path: Path) -> None:
    experiment_path = tmp_path / 'memory-limit.yaml'
    shutil.copyfile(EXPERIMENT_TEMPLATE, experiment_path)
    child_script = 'import time; allocation = bytearray(30_000_000); time.sleep(60)'
    parent_script = (
        'import subprocess, sys, time; allocation = bytearray(30_000_000); '
        f"subprocess.Popen([sys.executable, '-c', {child_script!r}]); time.sleep(60)"
    )
    slot = ResourceSlot(
        slot_id='memory-slot',
        cuda_devices=(),
        cpu_affinity=_available_cpu_cores(1),
        ram_capacity_bytes=100_000_000,
        working_directory=tmp_path,
        log_directory=tmp_path / 'logs',
    )
    configuration = QueueConfiguration(
        runner=RunnerCommand(command=(sys.executable, '-c', parent_script)),
        slots=(slot,),
        experiments=(
            QueuedExperiment(
                experiment_id='memory-limit',
                experiment_file=experiment_path,
                resources=ResourceRequest(
                    cuda_device_count=0,
                    cpu_core_count=1,
                    ram_limit_bytes=50_000_000,
                ),
            ),
        ),
        summary_path=tmp_path / 'queue-summary.json',
        poll_interval_seconds=0.01,
        termination_grace_seconds=0.1,
    )

    validated_queue = validate_queue_for_launch(configuration)
    summary = ExperimentQueueRunner(lambda: validated_queue).run()

    status = summary.experiments[0]
    assert isinstance(status, FailedExperimentStatus)
    assert 'Process-tree RSS' in status.reason
    assert status.exit_code != 0


def test_queue_termination_records_failure_and_releases_the_running_slot(tmp_path: Path) -> None:
    experiment_path = tmp_path / 'long-running.yaml'
    shutil.copyfile(EXPERIMENT_TEMPLATE, experiment_path)
    slot = ResourceSlot(
        slot_id='slot',
        cuda_devices=(),
        cpu_affinity=_available_cpu_cores(1),
        ram_capacity_bytes=2_000_000_000,
        working_directory=tmp_path,
        log_directory=tmp_path / 'logs',
    )
    configuration = QueueConfiguration(
        runner=RunnerCommand(command=(sys.executable, '-c', 'import time; time.sleep(60)')),
        slots=(slot,),
        experiments=(
            QueuedExperiment(
                experiment_id='long-running',
                experiment_file=experiment_path,
                resources=ResourceRequest(
                    cuda_device_count=0,
                    cpu_core_count=1,
                    ram_limit_bytes=1_500_000_000,
                ),
            ),
        ),
        summary_path=tmp_path / 'queue-summary.json',
        poll_interval_seconds=0.01,
        termination_grace_seconds=1.0,
    )
    validated_queue = validate_queue_for_launch(configuration)
    runner = ExperimentQueueRunner(lambda: validated_queue)
    termination = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM))
    termination.start()
    try:
        summary = runner.run()
    finally:
        termination.cancel()

    status = summary.experiments[0]
    assert isinstance(status, FailedExperimentStatus)
    assert status.exit_code == -15
    assert 'complete process group was terminated' in status.reason
    assert runner.active_process_count == 0

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

from src.experiment_queue.cgroup import CgroupV2MemoryScope
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


def _fake_cgroup(directory: Path) -> Path:
    directory.mkdir()
    for filename, content in (
        ('cgroup.events', 'populated 0\nfrozen 0\n'),
        ('cgroup.kill', ''),
        ('cgroup.procs', ''),
        ('memory.max', 'max\n'),
        ('memory.oom.group', '0\n'),
        ('memory.swap.max', 'max\n'),
    ):
        (directory / filename).write_text(content, encoding='ascii')
    return directory


def _assignment(temporary_directory: Path, ram_limit_bytes: int = 2_000_000_000) -> ResourceAssignment:
    return ResourceAssignment(
        slot_id='test-slot',
        cuda_devices=(3, 5),
        cpu_affinity=_available_cpu_cores(1),
        ram_limit_bytes=ram_limit_bytes,
        cgroup_directory=_fake_cgroup(temporary_directory / 'cgroup'),
        working_directory=temporary_directory,
        log_directory=temporary_directory,
    )


def test_linux_launcher_applies_cuda_affinity_cgroup_memory_process_group_and_logs(tmp_path: Path) -> None:
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
    assert (assignment.cgroup_directory / 'memory.max').read_text(encoding='ascii') == (
        f'{assignment.ram_limit_bytes}\n'
    )
    assert (assignment.cgroup_directory / 'memory.swap.max').read_text(encoding='ascii') == '0\n'
    assert (assignment.cgroup_directory / 'memory.oom.group').read_text(encoding='ascii') == '1\n'


@pytest.mark.integration
def test_linux_cgroup_enforces_one_aggregate_budget_across_descendants(tmp_path: Path) -> None:
    cgroup_directory = _configured_integration_cgroup()
    memory_scope = CgroupV2MemoryScope(cgroup_directory)
    memory_scope.validate()
    memory_scope.validate_process_migration()
    assignment = ResourceAssignment(
        slot_id='real-cgroup',
        cuda_devices=(),
        cpu_affinity=_available_cpu_cores(1),
        ram_limit_bytes=160_000_000,
        cgroup_directory=cgroup_directory,
        working_directory=tmp_path,
        log_directory=tmp_path,
    )
    child_script = 'import time; allocation = bytearray(100_000_000); time.sleep(30)'
    parent_script = (
        'import subprocess, sys; '
        f"children = [subprocess.Popen([sys.executable, '-c', {child_script!r}]) for _ in range(2)]; "
        '[child.wait() for child in children]'
    )
    oom_kills_before = _memory_event_count(cgroup_directory, 'oom_kill')

    running_process = launch_process(
        experiment_id='aggregate-memory',
        command=(sys.executable, '-c', parent_script),
        assignment=assignment,
        stdout_path=tmp_path / 'aggregate-memory.stdout.log',
        stderr_path=tmp_path / 'aggregate-memory.stderr.log',
    )
    exit_code = running_process.process.wait(timeout=20.0)
    running_process.close_logs()
    _wait_for_empty_cgroup(running_process.memory_scope)

    assert exit_code != 0
    assert _memory_event_count(cgroup_directory, 'oom_kill') > oom_kills_before
    assert not running_process.memory_scope.populated


@pytest.mark.integration
def test_linux_cgroup_remains_populated_after_the_runner_leader_exits(tmp_path: Path) -> None:
    cgroup_directory = _configured_integration_cgroup()
    assignment = ResourceAssignment(
        slot_id='real-cgroup',
        cuda_devices=(),
        cpu_affinity=_available_cpu_cores(1),
        ram_limit_bytes=500_000_000,
        cgroup_directory=cgroup_directory,
        working_directory=tmp_path,
        log_directory=tmp_path,
    )
    child_script = 'import time; time.sleep(1.0)'
    parent_script = (
        f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {child_script!r}], start_new_session=True)"
    )

    running_process = launch_process(
        experiment_id='surviving-descendant',
        command=(sys.executable, '-c', parent_script),
        assignment=assignment,
        stdout_path=tmp_path / 'surviving-descendant.stdout.log',
        stderr_path=tmp_path / 'surviving-descendant.stderr.log',
    )
    assert running_process.process.wait(timeout=10.0) == 0
    assert running_process.memory_scope.populated
    _wait_for_empty_cgroup(running_process.memory_scope)
    running_process.close_logs()

    assert not running_process.memory_scope.populated


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
        cgroup_directory=_fake_cgroup(tmp_path / 'cgroup'),
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

    summary = ExperimentQueueRunner(validate_queue_for_launch(configuration)).run()

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


def test_queue_termination_records_failure_and_releases_the_running_slot(tmp_path: Path) -> None:
    experiment_path = tmp_path / 'long-running.yaml'
    shutil.copyfile(EXPERIMENT_TEMPLATE, experiment_path)
    slot = ResourceSlot(
        slot_id='slot',
        cuda_devices=(),
        cpu_affinity=_available_cpu_cores(1),
        ram_capacity_bytes=2_000_000_000,
        cgroup_directory=_fake_cgroup(tmp_path / 'cgroup'),
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
    runner = ExperimentQueueRunner(validate_queue_for_launch(configuration))
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


def _memory_event_count(cgroup_directory: Path, event_name: str) -> int:
    for line in (cgroup_directory / 'memory.events').read_text(encoding='ascii').splitlines():
        name, value = line.split(maxsplit=1)
        if name == event_name:
            return int(value)
    raise ValueError(f'Memory event {event_name!r} is missing from {cgroup_directory}.')


def _configured_integration_cgroup() -> Path:
    cgroup_directory_value = os.environ.get('EXPERIMENT_QUEUE_TEST_CGROUP')
    if cgroup_directory_value is None:
        pytest.skip('EXPERIMENT_QUEUE_TEST_CGROUP is not configured.')
    return Path(cgroup_directory_value)


def _wait_for_empty_cgroup(memory_scope: CgroupV2MemoryScope) -> None:
    deadline = time.monotonic() + 5.0
    while memory_scope.populated and time.monotonic() < deadline:
        time.sleep(0.01)

from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from src.az.replay.envelope import ReplayRecord
from src.az.replay.storage import ShardMetadata
from src.az.runtime.ipc import ByteQueue, StopSignal
from src.az.runtime.messages import (
    IpcReplayRecord,
    WorkerFailure,
    WorkerReady,
    WorkerRecords,
    WorkerStopped,
)
from src.az.runtime.orchestrator import ReplayPublisher, RuntimeOrchestrator, WorkerEntrypoint
from src.az.runtime.telemetry_journal import TelemetryJournal
from src.az.runtime.topology import RuntimeTopology, WorkerAssignment
from test.unit.go_stage5_helpers import envelope


def _send(queue: ByteQueue, message: WorkerReady | WorkerRecords | WorkerStopped | WorkerFailure) -> None:
    queue.put(message.model_dump_json().encode())


def _clean_worker(
    specification: bytes,
    stop_signal: StopSignal,
    queue: ByteQueue,
) -> None:
    del specification, stop_signal
    _send(queue, WorkerReady(kind='worker_ready', worker_index=0, process_id=os.getpid(), model_version=0))
    _send(queue, WorkerStopped(kind='worker_stopped', worker_index=0, completed_games=0, emitted_positions=0))


def _failing_worker(
    specification: bytes,
    stop_signal: StopSignal,
    queue: ByteQueue,
) -> None:
    del specification, stop_signal
    _send(
        queue,
        WorkerFailure(
            kind='worker_failure',
            worker_index=0,
            error_type='RuntimeError',
            message='injected worker failure',
        ),
    )
    raise RuntimeError('injected worker failure')


def _slow_start_worker(
    specification: bytes,
    stop_signal: StopSignal,
    queue: ByteQueue,
) -> None:
    del specification, stop_signal
    time.sleep(0.25)
    _send(queue, WorkerReady(kind='worker_ready', worker_index=0, process_id=os.getpid(), model_version=0))
    _send(queue, WorkerStopped(kind='worker_stopped', worker_index=0, completed_games=0, emitted_positions=0))


def _wrong_pid_worker(
    specification: bytes,
    stop_signal: StopSignal,
    queue: ByteQueue,
) -> None:
    del specification, stop_signal
    _send(
        queue,
        WorkerReady(
            kind='worker_ready',
            worker_index=0,
            process_id=os.getpid() + 1,
            model_version=0,
        ),
    )
    _send(queue, WorkerStopped(kind='worker_stopped', worker_index=0, completed_games=0, emitted_positions=0))


def _late_duplicate_ready_worker(
    specification: bytes,
    stop_signal: StopSignal,
    queue: ByteQueue,
) -> None:
    del specification, stop_signal
    ready = WorkerReady(
        kind='worker_ready',
        worker_index=0,
        process_id=os.getpid(),
        model_version=0,
    )
    _send(queue, ready)
    time.sleep(0.25)
    _send(queue, ready)
    _send(queue, WorkerStopped(kind='worker_stopped', worker_index=0, completed_games=0, emitted_positions=0))


def _publishing_worker(
    specification: bytes,
    stop_signal: StopSignal,
    queue: ByteQueue,
) -> None:
    del specification, stop_signal
    _send(queue, WorkerReady(kind='worker_ready', worker_index=0, process_id=os.getpid(), model_version=0))
    record = ReplayRecord(envelope=envelope(1), payload=b'payload')
    _send(
        queue,
        WorkerRecords(
            kind='worker_records',
            worker_index=0,
            records=(IpcReplayRecord.from_record(record),),
        ),
    )
    _send(queue, WorkerStopped(kind='worker_stopped', worker_index=0, completed_games=1, emitted_positions=1))


def _orchestrator(
    tmp_path: Path,
    worker: WorkerEntrypoint,
    publisher: ReplayPublisher,
    wall_clock_seconds: float = 10,
    startup_timeout_seconds: float = 60,
) -> RuntimeOrchestrator:
    start_method = 'spawn'
    return RuntimeOrchestrator(
        worker_entrypoint=worker,
        worker_specifications=(b'worker-0',),
        wall_clock_seconds=wall_clock_seconds,
        startup_timeout_seconds=startup_timeout_seconds,
        shutdown_grace_seconds=3,
        start_method=start_method,
        replay_publisher=publisher,
        topology=RuntimeTopology(
            workers=(WorkerAssignment(worker_index=0, device_index=None, maximum_active_searches=1),),
            trainer_device_indices=(),
        ),
        games_per_shard=1,
        telemetry_path=(tmp_path / 'telemetry.azt').resolve(),
        telemetry_write_every_seconds=1,
    )


def _discard(records: tuple[ReplayRecord, ...]) -> ShardMetadata:
    return ShardMetadata(Path('unused'), 0, len(records), 1)


@pytest.mark.integration
def test_nontimeout_completion_records_clean_lifecycle(tmp_path: Path) -> None:
    result = _orchestrator(tmp_path, _clean_worker, _discard).run()

    assert not result.timed_out
    assert result.orphan_process_ids == ()
    assert not any(child.name.startswith('az-self-play-') for child in multiprocessing.active_children())


@pytest.mark.integration
def test_experiment_clock_starts_only_after_slow_worker_is_ready(tmp_path: Path) -> None:
    result = _orchestrator(
        tmp_path,
        _slow_start_worker,
        _discard,
        wall_clock_seconds=1,
        startup_timeout_seconds=60,
    ).run()

    assert result.startup_elapsed_seconds >= 0.2
    assert result.elapsed_seconds < 2
    assert not result.timed_out
    assert any(isinstance(message, WorkerStopped) for message in result.messages)
    assert not any(child.name.startswith('az-self-play-') for child in multiprocessing.active_children())


@pytest.mark.integration
def test_wrong_readiness_pid_is_fatal_and_leaves_no_orphan(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match='Runtime orchestration failed'):
        _orchestrator(tmp_path, _wrong_pid_worker, _discard).run()

    assert not any(child.name.startswith('az-self-play-') for child in multiprocessing.active_children())


@pytest.mark.integration
def test_late_duplicate_readiness_is_fatal_and_leaves_no_orphan(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match='Runtime orchestration failed'):
        _orchestrator(tmp_path, _late_duplicate_ready_worker, _discard).run()

    assert not any(child.name.startswith('az-self-play-') for child in multiprocessing.active_children())


@pytest.mark.integration
def test_worker_failure_leaves_no_orphan(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match='Self-play worker 0 failed'):
        _orchestrator(tmp_path, _failing_worker, _discard).run()

    assert not any(child.name.startswith('az-self-play-') for child in multiprocessing.active_children())


@pytest.mark.integration
def test_publisher_failure_leaves_no_orphan(tmp_path: Path) -> None:
    def fail_publication(records: tuple[ReplayRecord, ...]) -> ShardMetadata:
        del records
        raise RuntimeError('injected publisher failure')

    with pytest.raises(RuntimeError, match='Runtime orchestration failed'):
        _orchestrator(tmp_path, _publishing_worker, fail_publication).run()

    payloads = TelemetryJournal((tmp_path / 'telemetry.azt').resolve()).read_payloads()
    assert all(b'worker_published' not in payload for payload in payloads)
    assert any(b'worker_publication_aborted' in payload for payload in payloads)
    assert not any(child.name.startswith('az-self-play-') for child in multiprocessing.active_children())

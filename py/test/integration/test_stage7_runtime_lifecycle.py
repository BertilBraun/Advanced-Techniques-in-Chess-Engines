from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

import pytest
from pydantic import Field

from src.az.config.base import FrozenModel
from src.az.replay.envelope import ReplayRecord
from src.az.replay.storage import ShardMetadata
from src.az.runtime.ipc import ByteQueue, RuntimeControlSignal
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


class GateWorkerSpecification(FrozenModel):
    worker_index: int = Field(ge=0)
    evidence_path: str = Field(min_length=1)
    delayed_readiness: bool
    fail_before_readiness: bool


class GateEvidence(FrozenModel):
    worker_index: int = Field(ge=0)
    experiment_start_monotonic_ns: int
    stop_was_set: bool


def _send(
    queue: ByteQueue,
    message: WorkerReady | WorkerRecords | WorkerStopped | WorkerFailure,
) -> None:
    queue.put(message.model_dump_json().encode())


def _clean_worker(
    specification: bytes,
    stop_signal: RuntimeControlSignal,
    queue: ByteQueue,
) -> None:
    del specification, stop_signal
    _send(queue, WorkerReady(kind='worker_ready', worker_index=0, process_id=os.getpid(), model_version=0))
    _send(queue, WorkerStopped(kind='worker_stopped', worker_index=0, completed_games=0, emitted_positions=0))


def _failing_worker(
    specification: bytes,
    stop_signal: RuntimeControlSignal,
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
    stop_signal: RuntimeControlSignal,
    queue: ByteQueue,
) -> None:
    del specification, stop_signal
    time.sleep(0.25)
    _send(queue, WorkerReady(kind='worker_ready', worker_index=0, process_id=os.getpid(), model_version=0))
    _send(queue, WorkerStopped(kind='worker_stopped', worker_index=0, completed_games=0, emitted_positions=0))


def _wrong_pid_worker(
    specification: bytes,
    stop_signal: RuntimeControlSignal,
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
    stop_signal: RuntimeControlSignal,
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
    stop_signal: RuntimeControlSignal,
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


def _gated_worker(
    specification: bytes,
    stop_signal: RuntimeControlSignal,
    queue: ByteQueue,
) -> None:
    parsed = GateWorkerSpecification.model_validate_json(specification)
    if parsed.delayed_readiness:
        time.sleep(0.2)
    _send(
        queue,
        WorkerReady(
            kind='worker_ready',
            worker_index=parsed.worker_index,
            process_id=os.getpid(),
            model_version=0,
        ),
    )
    epoch = stop_signal.wait_for_experiment_start()
    Path(parsed.evidence_path).write_text(
        GateEvidence(
            worker_index=parsed.worker_index,
            experiment_start_monotonic_ns=epoch,
            stop_was_set=stop_signal.is_set(),
        ).model_dump_json(),
        encoding='utf-8',
    )
    _send(
        queue,
        WorkerStopped(
            kind='worker_stopped',
            worker_index=parsed.worker_index,
            completed_games=0,
            emitted_positions=0,
        ),
    )


def _startup_failure_gated_worker(
    specification: bytes,
    stop_signal: RuntimeControlSignal,
    queue: ByteQueue,
) -> None:
    parsed = GateWorkerSpecification.model_validate_json(specification)
    if parsed.fail_before_readiness:
        _send(
            queue,
            WorkerFailure(
                kind='worker_failure',
                worker_index=parsed.worker_index,
                error_type='RuntimeError',
                message='startup failure before experiment gate',
            ),
        )
        raise RuntimeError('startup failure before experiment gate')
    _send(
        queue,
        WorkerReady(
            kind='worker_ready',
            worker_index=parsed.worker_index,
            process_id=os.getpid(),
            model_version=0,
        ),
    )
    epoch = stop_signal.wait_for_experiment_start()
    Path(parsed.evidence_path).write_text(
        GateEvidence(
            worker_index=parsed.worker_index,
            experiment_start_monotonic_ns=epoch,
            stop_was_set=stop_signal.is_set(),
        ).model_dump_json(),
        encoding='utf-8',
    )
    _send(
        queue,
        WorkerStopped(
            kind='worker_stopped',
            worker_index=parsed.worker_index,
            completed_games=0,
            emitted_positions=0,
        ),
    )


def _orchestrator(
    tmp_path: Path,
    worker: WorkerEntrypoint,
    publisher: ReplayPublisher,
    wall_clock_seconds: float = 10,
    startup_timeout_seconds: float = 60,
    worker_count: int = 1,
    worker_specifications: tuple[bytes, ...] | None = None,
    elapsed_offset_seconds: float = 0,
) -> RuntimeOrchestrator:
    start_method = 'spawn'
    serialized_specifications = (
        tuple(str(index).encode('ascii') for index in range(worker_count))
        if worker_specifications is None
        else worker_specifications
    )
    return RuntimeOrchestrator(
        worker_entrypoint=worker,
        worker_specifications=serialized_specifications,
        wall_clock_seconds=wall_clock_seconds,
        startup_timeout_seconds=startup_timeout_seconds,
        shutdown_grace_seconds=3,
        start_method=start_method,
        replay_publisher=publisher,
        topology=RuntimeTopology(
            workers=tuple(
                WorkerAssignment(worker_index=index, device_index=None, maximum_active_searches=1)
                for index in range(len(serialized_specifications))
            ),
            trainer_device_indices=(),
        ),
        games_per_shard=1,
        telemetry_path=(tmp_path / 'telemetry.azt').resolve(),
        telemetry_write_every_seconds=1,
        elapsed_offset_seconds=elapsed_offset_seconds,
    )


def _discard(records: tuple[ReplayRecord, ...]) -> ShardMetadata:
    return ShardMetadata(Path('unused'), 0, len(records), 1)


def _gate_specifications(
    temporary_directory: Path,
    fail_first_worker: bool,
) -> tuple[tuple[bytes, ...], tuple[Path, ...]]:
    evidence_paths = tuple(
        (temporary_directory / f'gate-evidence-{worker_index}.json').resolve() for worker_index in range(2)
    )
    specifications = tuple(
        GateWorkerSpecification(
            worker_index=worker_index,
            evidence_path=str(evidence_paths[worker_index]),
            delayed_readiness=worker_index == 1 and not fail_first_worker,
            fail_before_readiness=worker_index == 0 and fail_first_worker,
        )
        .model_dump_json()
        .encode()
        for worker_index in range(2)
    )
    return specifications, evidence_paths


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
def test_workers_observe_one_shared_epoch_only_after_every_worker_is_ready(tmp_path: Path) -> None:
    specifications, evidence_paths = _gate_specifications(tmp_path, fail_first_worker=False)
    result = _orchestrator(
        tmp_path,
        _gated_worker,
        _discard,
        worker_specifications=specifications,
    ).run()

    observations = tuple(GateEvidence.model_validate_json(path.read_text(encoding='utf-8')) for path in evidence_paths)
    assert result.startup_elapsed_seconds >= 0.15
    assert len(observations) == 2
    assert {observation.worker_index for observation in observations} == {0, 1}
    assert len({observation.experiment_start_monotonic_ns for observation in observations}) == 1
    assert observations[0].experiment_start_monotonic_ns > 0
    assert not any(observation.stop_was_set for observation in observations)
    assert result.orphan_process_ids == ()


@pytest.mark.integration
def test_startup_failure_releases_gate_without_beginning_experiment(tmp_path: Path) -> None:
    specifications, evidence_paths = _gate_specifications(tmp_path, fail_first_worker=True)

    with pytest.raises(RuntimeError, match='Self-play worker 0 failed'):
        _orchestrator(
            tmp_path,
            _startup_failure_gated_worker,
            _discard,
            worker_specifications=specifications,
        ).run()

    assert not evidence_paths[0].exists()
    observation = GateEvidence.model_validate_json(evidence_paths[1].read_text(encoding='utf-8'))
    assert observation.worker_index == 1
    assert observation.experiment_start_monotonic_ns == 0
    assert observation.stop_was_set
    assert not any(child.name.startswith('az-self-play-') for child in multiprocessing.active_children())


@pytest.mark.integration
def test_resumed_workers_observe_cumulative_schedule_epoch(tmp_path: Path) -> None:
    specifications, evidence_paths = _gate_specifications(tmp_path, fail_first_worker=False)
    offset_seconds = 123

    _orchestrator(
        tmp_path,
        _gated_worker,
        _discard,
        worker_specifications=specifications,
        elapsed_offset_seconds=offset_seconds,
    ).run()

    evidence = tuple(GateEvidence.model_validate_json(path.read_bytes()) for path in evidence_paths)
    observed_elapsed = tuple(
        (time.monotonic_ns() - item.experiment_start_monotonic_ns) / 1_000_000_000 for item in evidence
    )
    assert all(elapsed >= offset_seconds for elapsed in observed_elapsed)


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

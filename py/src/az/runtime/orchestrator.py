from __future__ import annotations

import multiprocessing
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter
import torch

from src.az.replay.envelope import ReplayRecord
from src.az.replay.storage import ShardMetadata
from src.az.runtime.ipc import ByteQueue, RuntimeControlSignal
from src.az.runtime.messages import (
    RuntimeFailure,
    RuntimeMessage,
    WorkerFailure,
    WorkerPublicationAborted,
    WorkerPublished,
    WorkerRecords,
    WorkerReady,
    WorkerStopped,
)
from src.az.runtime.publication import CompletedGamePublicationBuffer, PublishedReplayShard
from src.az.runtime.telemetry_journal import TelemetryJournal
from src.az.runtime.topology import RuntimeTopology


WorkerEntrypoint = Callable[
    [bytes, RuntimeControlSignal, ByteQueue],
    None,
]
ReplayPublisher = Callable[[tuple[ReplayRecord, ...]], ShardMetadata]
MESSAGE_ADAPTER = TypeAdapter(RuntimeMessage)


class _RuntimeControl:
    def __init__(self, context: multiprocessing.context.BaseContext) -> None:
        self._stop = context.Event()
        self._start = context.Event()
        self._epoch = context.Value('q', 0)

    def is_set(self) -> bool:
        return self._stop.is_set()

    def set(self) -> None:
        self._stop.set()
        self._start.set()

    def begin_experiment(self, monotonic_ns: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = monotonic_ns
        self._start.set()

    def wait_for_experiment_start(self) -> int:
        self._start.wait()
        with self._epoch.get_lock():
            return int(self._epoch.value)


@dataclass(frozen=True)
class RuntimeResult:
    startup_elapsed_seconds: float
    elapsed_seconds: float
    messages: tuple[RuntimeMessage, ...]
    orphan_process_ids: tuple[int, ...]
    timed_out: bool


class RuntimeOrchestrator:
    def __init__(
        self,
        worker_entrypoint: WorkerEntrypoint,
        worker_specifications: tuple[bytes, ...],
        wall_clock_seconds: float,
        startup_timeout_seconds: float,
        shutdown_grace_seconds: float,
        start_method: str,
        replay_publisher: ReplayPublisher,
        topology: RuntimeTopology,
        games_per_shard: int,
        telemetry_path: Path,
        telemetry_write_every_seconds: float,
    ) -> None:
        if not worker_specifications:
            raise ValueError('At least one worker specification is required.')
        if wall_clock_seconds <= 0 or startup_timeout_seconds <= 0 or shutdown_grace_seconds <= 0:
            raise ValueError('Runtime duration, startup timeout, and shutdown grace must be positive.')
        if start_method not in multiprocessing.get_all_start_methods():
            raise ValueError('Requested multiprocessing start method is unavailable.')
        if len(topology.workers) != len(worker_specifications):
            raise ValueError('Runtime topology must assign every worker specification exactly once.')
        topology.validate_visible_cuda_devices(torch.cuda.device_count())
        if games_per_shard <= 0:
            raise ValueError('Games per replay shard must be positive.')
        if not telemetry_path.is_absolute() or telemetry_write_every_seconds <= 0:
            raise ValueError('Telemetry path must be absolute and cadence must be positive.')
        self._worker_entrypoint = worker_entrypoint
        self._worker_specifications = worker_specifications
        self._wall_clock_seconds = wall_clock_seconds
        self._startup_timeout_seconds = startup_timeout_seconds
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._context = multiprocessing.get_context(start_method)
        self._topology = topology
        self._games_per_shard = games_per_shard
        self._publication_buffer = CompletedGamePublicationBuffer(
            games_per_shard,
            replay_publisher,
        )
        self._pending_publications: list[tuple[int, int, int]] = []
        self._publication_scanned_count = 0
        self._telemetry_journal = TelemetryJournal(telemetry_path)
        self._telemetry_write_every_seconds = telemetry_write_every_seconds
        self._telemetry_written_count = 0
        self._last_telemetry_write = 0.0

    def run(self) -> RuntimeResult:
        startup_started = time.monotonic()
        experiment_started_ns: int | None = None
        stop_event = _RuntimeControl(self._context)
        message_queue: ByteQueue = self._context.Queue()
        processes = tuple(
            self._context.Process(
                target=self._worker_entrypoint,
                args=(specification, stop_event, message_queue),
                name=f'az-self-play-{index}',
            )
            for index, specification in enumerate(self._worker_specifications)
        )
        messages: list[RuntimeMessage] = []
        for process in processes:
            process.start()
        failure: WorkerFailure | None = None
        timed_out = False
        runtime_failure: Exception | None = None
        try:
            failure = self._await_worker_startup(
                message_queue,
                messages,
                processes,
                startup_started + self._startup_timeout_seconds,
            )
            if failure is None:
                experiment_started_ns = time.monotonic_ns()
                stop_event.begin_experiment(experiment_started_ns)
                deadline_ns = experiment_started_ns + int(self._wall_clock_seconds * 1_000_000_000)
                while time.monotonic_ns() < deadline_ns and failure is None:
                    self._drain(message_queue, messages, timeout=0.05)
                    self._publish(messages)
                    failure = next(
                        (message for message in reversed(messages) if isinstance(message, WorkerFailure)),
                        None,
                    )
                    if self._stopped_worker_indices(messages) == set(range(len(processes))) or all(
                        not process.is_alive() for process in processes
                    ):
                        break
                timed_out = (
                    time.monotonic_ns() >= deadline_ns
                    and any(process.is_alive() for process in processes)
                    and self._stopped_worker_indices(messages) != set(range(len(processes)))
                )
        except Exception as error:
            runtime_failure = error
            self._abort_publications(messages)
            messages.append(
                RuntimeFailure(
                    kind='runtime_failure',
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
        finally:
            stop_event.set()
            grace_deadline = time.monotonic() + self._shutdown_grace_seconds
            while time.monotonic() < grace_deadline and any(process.is_alive() for process in processes):
                self._drain(message_queue, messages, timeout=0.02)
                if runtime_failure is None:
                    try:
                        self._publish(messages)
                    except Exception as error:
                        runtime_failure = error
                        self._abort_publications(messages)
                        messages.append(
                            RuntimeFailure(
                                kind='runtime_failure',
                                error_type=type(error).__name__,
                                message=str(error),
                            )
                        )
                for process in processes:
                    process.join(timeout=0)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=1)
            self._drain_final_lifecycle_messages(
                message_queue,
                messages,
                process_count=len(processes),
                timeout_seconds=1.0,
            )
            if runtime_failure is None and failure is None:
                try:
                    self._validate_worker_readiness(
                        messages,
                        processes,
                        require_complete=True,
                    )
                except Exception as error:
                    runtime_failure = error
                    self._abort_publications(messages)
                    messages.append(
                        RuntimeFailure(
                            kind='runtime_failure',
                            error_type=type(error).__name__,
                            message=str(error),
                        )
                    )
            if runtime_failure is None:
                try:
                    self._publish(messages)
                    self._flush_publication(messages)
                except Exception as error:
                    runtime_failure = error
                    self._abort_publications(messages)
                    messages.append(
                        RuntimeFailure(
                            kind='runtime_failure',
                            error_type=type(error).__name__,
                            message=str(error),
                        )
                    )
            self._write_telemetry(messages, force=True)
            message_queue.close()
            message_queue.join_thread()
        orphan_ids = tuple(process.pid for process in processes if process.is_alive() and process.pid is not None)
        nonzero = tuple(process.exitcode for process in processes if process.exitcode not in (0, None))
        if runtime_failure is not None:
            raise RuntimeError('Runtime orchestration failed.') from runtime_failure
        if failure is not None:
            raise RuntimeError(f'Self-play worker {failure.worker_index} failed: {failure.message}')
        if nonzero:
            raise RuntimeError(f'Self-play worker processes exited unsuccessfully: {nonzero}.')
        if orphan_ids:
            raise RuntimeError(f'Self-play worker processes did not terminate: {orphan_ids}.')
        stopped = {message.worker_index for message in messages if isinstance(message, WorkerStopped)}
        if stopped != set(range(len(processes))):
            raise RuntimeError('Not every self-play worker reported a clean stop.')
        experiment_elapsed = (
            0.0 if experiment_started_ns is None else (time.monotonic_ns() - experiment_started_ns) / 1_000_000_000
        )
        return RuntimeResult(
            startup_elapsed_seconds=(
                (experiment_started_ns / 1_000_000_000 if experiment_started_ns is not None else time.monotonic())
                - startup_started
            ),
            elapsed_seconds=experiment_elapsed,
            messages=tuple(messages),
            orphan_process_ids=orphan_ids,
            timed_out=timed_out,
        )

    def _await_worker_startup(
        self,
        message_queue: ByteQueue,
        messages: list[RuntimeMessage],
        processes: tuple[multiprocessing.Process, ...],
        deadline: float,
    ) -> WorkerFailure | None:
        expected_workers = set(range(len(processes)))
        all_exited_at: float | None = None
        while time.monotonic() < deadline:
            self._drain(message_queue, messages, timeout=0.05)
            self._publish(messages)
            failure = next(
                (message for message in reversed(messages) if isinstance(message, WorkerFailure)),
                None,
            )
            if failure is not None:
                return failure
            ready_workers = self._validate_worker_readiness(
                messages,
                processes,
                require_complete=False,
            )
            if ready_workers == expected_workers:
                return None
            if all(not process.is_alive() for process in processes):
                if all_exited_at is None:
                    all_exited_at = time.monotonic()
                elif time.monotonic() - all_exited_at >= 1:
                    raise RuntimeError('Self-play workers exited before reporting readiness.')
            else:
                all_exited_at = None
        raise TimeoutError('Self-play workers did not become ready before the startup timeout.')

    @staticmethod
    def _validate_worker_readiness(
        messages: list[RuntimeMessage],
        processes: tuple[multiprocessing.Process, ...],
        require_complete: bool,
    ) -> set[int]:
        expected_workers = set(range(len(processes)))
        ready_messages = tuple(message for message in messages if isinstance(message, WorkerReady))
        ready_workers = {message.worker_index for message in ready_messages}
        if len(ready_messages) != len(ready_workers):
            raise RuntimeError('A self-play worker reported readiness more than once.')
        if not ready_workers.issubset(expected_workers):
            raise RuntimeError('A self-play worker reported an unexpected readiness identity.')
        for message in ready_messages:
            expected_process_id = processes[message.worker_index].pid
            if expected_process_id is None or message.process_id != expected_process_id:
                raise RuntimeError('Self-play worker readiness PID does not match its assigned process.')
        if require_complete and ready_workers != expected_workers:
            raise RuntimeError('Not every self-play worker reported readiness.')
        return ready_workers

    def _drain_final_lifecycle_messages(
        self,
        message_queue: ByteQueue,
        messages: list[RuntimeMessage],
        process_count: int,
        timeout_seconds: float,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        expected_workers = set(range(process_count))
        while time.monotonic() < deadline:
            self._drain(message_queue, messages, timeout=0.02)
            stopped_workers = self._stopped_worker_indices(messages)
            if stopped_workers == expected_workers or any(isinstance(message, WorkerFailure) for message in messages):
                return

    @staticmethod
    def _stopped_worker_indices(messages: list[RuntimeMessage]) -> set[int]:
        return {message.worker_index for message in messages if isinstance(message, WorkerStopped)}

    def _drain(
        self,
        message_queue: ByteQueue,
        messages: list[RuntimeMessage],
        timeout: float,
    ) -> None:
        first = True
        while True:
            try:
                contents = message_queue.get(timeout=timeout if first else 0)
            except queue.Empty:
                return
            messages.append(MESSAGE_ADAPTER.validate_json(contents))
            first = False

    def _publish(self, messages: list[RuntimeMessage]) -> None:
        for index in range(self._publication_scanned_count, len(messages)):
            message = messages[index]
            if isinstance(message, WorkerRecords):
                records = tuple(record.to_record() for record in message.records)
                self._pending_publications.append((index, message.worker_index, len(records)))
                published = self._publication_buffer.add_completed_game(records)
                if published is not None:
                    self._commit_publication_messages(messages, published)
        self._publication_scanned_count = len(messages)
        self._write_telemetry(messages, force=False)

    def _flush_publication(self, messages: list[RuntimeMessage]) -> None:
        published = self._publication_buffer.flush()
        if published is None:
            return
        self._commit_publication_messages(messages, published)

    def _commit_publication_messages(
        self,
        messages: list[RuntimeMessage],
        published: PublishedReplayShard,
    ) -> None:
        if len(self._pending_publications) != published.game_count:
            raise AssertionError('Committed replay shard does not match pending completed games.')
        for index, worker_index, position_count in self._pending_publications:
            messages[index] = WorkerPublished(
                kind='worker_published',
                worker_index=worker_index,
                committed_games=1,
                committed_positions=position_count,
                partial_shard=published.partial,
                shard_sequence=published.shard_sequence,
            )
        self._pending_publications.clear()

    def _abort_publications(self, messages: list[RuntimeMessage]) -> None:
        self._publication_buffer.discard()
        for index, worker_index, position_count in self._pending_publications:
            messages[index] = WorkerPublicationAborted(
                kind='worker_publication_aborted',
                worker_index=worker_index,
                discarded_positions=position_count,
            )
        self._pending_publications.clear()

    def _write_telemetry(
        self,
        messages: list[RuntimeMessage],
        force: bool,
    ) -> None:
        now = time.monotonic()
        if not force and now - self._last_telemetry_write < self._telemetry_write_every_seconds:
            return
        if self._pending_publications:
            return
        unwritten = messages[self._telemetry_written_count :]
        if not unwritten:
            return
        self._telemetry_journal.append(tuple(message.model_dump_json().encode() for message in unwritten))
        self._telemetry_written_count = len(messages)
        self._last_telemetry_write = now

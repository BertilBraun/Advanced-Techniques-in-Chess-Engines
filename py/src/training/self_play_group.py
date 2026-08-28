from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

from src.games.implementation import GameImplementation
from src.search_budget.calibration import BlendPublication
from src.self_play.process_runtime import self_play_worker_main
from src.self_play.protocol import (
    PausedSelfPlayState,
    RunningSelfPlayState,
    SelfPlayDesiredState,
    SelfPlayStateApplied,
    StoppedSelfPlayState,
)
from src.self_play.resignation import PublishedResignationPolicy
from src.training.checkpoint import CheckpointReference
from src.util.log import log, warn

RESPONSE_TIMEOUT_SECONDS = 120.0
RESTART_HANDSHAKE_TIMEOUT_SECONDS = 600.0
RESTART_BACKOFF_SECONDS = 120.0
TERMINATION_JOIN_SECONDS = 5.0
SHUTDOWN_JOIN_SECONDS = 60.0


@dataclass
class SelfPlayWorkerSlot:
    worker_id: int
    device_id: int
    connection: Connection | None
    process: BaseProcess | None
    awaiting_handshake: bool = False
    handshake_deadline: float = 0.0
    next_restart_allowed_at: float = 0.0

    @property
    def is_live(self) -> bool:
        return self.connection is not None and not self.awaiting_handshake


@dataclass(frozen=True)
class SelfPlaySupervision:
    restarted_worker_ids: tuple[int, ...]
    failed_worker_ids: tuple[int, ...]


class SelfPlayGroup:
    def __init__(self, game: GameImplementation) -> None:
        self.game = game
        self._closed = False
        self._context = multiprocessing.get_context('spawn')
        topology = game.training.topology.self_play
        self._slots: list[SelfPlayWorkerSlot] = []
        for worker_id, device_id in enumerate(topology.device_ids):
            connection, process = self._start_worker(worker_id, device_id)
            self._slots.append(SelfPlayWorkerSlot(worker_id, device_id, connection, process))

    @property
    def worker_count(self) -> int:
        return len(self._slots)

    @property
    def live_worker_count(self) -> int:
        return sum(1 for slot in self._slots if slot.is_live)

    def apply(
        self,
        desired_states: tuple[RunningSelfPlayState | StoppedSelfPlayState, ...],
    ) -> tuple[SelfPlayStateApplied, ...]:
        if self._closed:
            raise RuntimeError('Self-play group is closed.')
        if len(desired_states) != self.worker_count:
            raise ValueError('Every self-play worker requires exactly one desired state.')
        live_slots = [slot for slot in self._slots if slot.is_live]
        for slot in live_slots:
            self._send(slot, desired_states[slot.worker_id])
        return self._collect(live_slots)

    def apply_to_workers(
        self,
        worker_ids: tuple[int, ...],
        desired_state: RunningSelfPlayState,
    ) -> tuple[SelfPlayStateApplied, ...]:
        selected_slots = self._slots_for_workers(worker_ids)
        for slot in selected_slots:
            self._send(slot, desired_state)
        return self._collect(selected_slots)

    def request_pause(self, worker_ids: tuple[int, ...]) -> None:
        for slot in self._slots_for_workers(worker_ids):
            self._send(slot, PausedSelfPlayState())

    def supervise(
        self,
        checkpoint: CheckpointReference,
        search_budget: BlendPublication,
        resignation_policy: PublishedResignationPolicy,
    ) -> SelfPlaySupervision:
        """Reap dead workers and drive restarts forward without ever waiting on a worker."""
        if self._closed:
            raise RuntimeError('Self-play group is closed.')
        now = time.monotonic()
        restarted_worker_ids: list[int] = []
        failed_worker_ids: list[int] = []
        for slot in self._slots:
            if slot.connection is None:
                if now >= slot.next_restart_allowed_at:
                    self._begin_restart(slot, checkpoint, search_budget, resignation_policy, now)
                continue
            if slot.awaiting_handshake:
                self._advance_restart(slot, now, restarted_worker_ids, failed_worker_ids)
                continue
            assert slot.process is not None
            if not slot.process.is_alive():
                self._retire(slot, f'process exited with code {slot.process.exitcode}', now, backoff_seconds=0.0)
        return SelfPlaySupervision(tuple(restarted_worker_ids), tuple(failed_worker_ids))

    def close(self) -> None:
        if self._closed:
            return
        self.apply(tuple(StoppedSelfPlayState() for _ in self._slots))
        for slot in self._slots:
            if slot.process is not None:
                slot.process.join(SHUTDOWN_JOIN_SECONDS)
                self._terminate(slot.process, slot.worker_id)
                slot.process = None
            if slot.connection is not None:
                slot.connection.close()
                slot.connection = None
        self._closed = True

    def _begin_restart(
        self,
        slot: SelfPlayWorkerSlot,
        checkpoint: CheckpointReference,
        search_budget: BlendPublication,
        resignation_policy: PublishedResignationPolicy,
        now: float,
    ) -> None:
        connection, process = self._start_worker(slot.worker_id, slot.device_id)
        slot.connection = connection
        slot.process = process
        slot.awaiting_handshake = True
        slot.handshake_deadline = now + RESTART_HANDSHAKE_TIMEOUT_SECONDS
        connection.send(
            RunningSelfPlayState(
                checkpoint=checkpoint,
                search_budget=search_budget,
                resignation_policy=resignation_policy,
            )
        )

    def _advance_restart(
        self,
        slot: SelfPlayWorkerSlot,
        now: float,
        restarted_worker_ids: list[int],
        failed_worker_ids: list[int],
    ) -> None:
        assert slot.connection is not None and slot.process is not None
        if slot.connection.poll():
            response = self._receive(slot.connection, timeout_seconds=0.0)
            if response is not None and response.kind == 'running':
                slot.awaiting_handshake = False
                restarted_worker_ids.append(slot.worker_id)
                return
            self._retire(slot, 'restart handshake returned an unusable state', now)
            failed_worker_ids.append(slot.worker_id)
            return
        if not slot.process.is_alive():
            self._retire(slot, f'restarted process exited with code {slot.process.exitcode}', now)
            failed_worker_ids.append(slot.worker_id)
            return
        if now >= slot.handshake_deadline:
            self._retire(slot, f'restart handshake timed out after {RESTART_HANDSHAKE_TIMEOUT_SECONDS:.0f}s', now)
            failed_worker_ids.append(slot.worker_id)

    def _retire(
        self,
        slot: SelfPlayWorkerSlot,
        reason: str,
        now: float,
        backoff_seconds: float = RESTART_BACKOFF_SECONDS,
    ) -> None:
        warn(f'Self-play worker {slot.worker_id} retired: {reason}.')
        if slot.process is not None:
            self._terminate(slot.process, slot.worker_id)
            slot.process = None
        if slot.connection is not None:
            slot.connection.close()
            slot.connection = None
        slot.awaiting_handshake = False
        slot.next_restart_allowed_at = now + backoff_seconds

    @staticmethod
    def _terminate(process: BaseProcess, worker_id: int) -> None:
        if process.is_alive():
            process.terminate()
            process.join(TERMINATION_JOIN_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(TERMINATION_JOIN_SECONDS)
        if process.is_alive():
            warn(f'Self-play worker {worker_id} did not die after SIGKILL; abandoning the process.')
        elif process.exitcode:
            log(f'Self-play worker {worker_id} exited with code {process.exitcode}.')

    def _send(self, slot: SelfPlayWorkerSlot, desired_state: SelfPlayDesiredState) -> None:
        assert slot.connection is not None
        slot.connection.send(desired_state)

    def _collect(self, slots: list[SelfPlayWorkerSlot]) -> tuple[SelfPlayStateApplied, ...]:
        responses: list[SelfPlayStateApplied] = []
        now = time.monotonic()
        for slot in slots:
            assert slot.connection is not None
            response = self._receive(slot.connection, RESPONSE_TIMEOUT_SECONDS)
            if response is None:
                self._retire(slot, f'no response within {RESPONSE_TIMEOUT_SECONDS:.0f}s', now)
                continue
            responses.append(response)
        return tuple(responses)

    @staticmethod
    def _receive(connection: Connection, timeout_seconds: float) -> SelfPlayStateApplied | None:
        if not connection.poll(timeout_seconds):
            return None
        try:
            response: SelfPlayStateApplied = connection.recv()
        except (EOFError, OSError):
            return None
        return response

    def _slots_for_workers(self, worker_ids: tuple[int, ...]) -> list[SelfPlayWorkerSlot]:
        if self._closed:
            raise RuntimeError('Self-play group is closed.')
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError('Selected self-play worker IDs must be unique.')
        if any(worker_id < 0 or worker_id >= self.worker_count for worker_id in worker_ids):
            raise ValueError('Selected self-play worker ID is outside the worker range.')
        return [self._slots[worker_id] for worker_id in worker_ids if self._slots[worker_id].is_live]

    def _start_worker(self, worker_id: int, device_id: int) -> tuple[Connection, BaseProcess]:
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=self_play_worker_main,
            args=(child, worker_id, device_id, self.game.configuration.model_dump_json()),
            name=f'self-play-worker-{worker_id}',
        )
        process.start()
        child.close()
        return parent, process

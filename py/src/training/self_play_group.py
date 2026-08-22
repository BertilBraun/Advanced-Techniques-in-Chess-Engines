from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

from src.games.implementation import GameImplementation
from src.self_play.process_runtime import self_play_worker_main
from src.self_play.protocol import (
    PausedSelfPlayState,
    RunningSelfPlayState,
    SelfPlayStateApplied,
    StoppedSelfPlayState,
)
from src.self_play.resignation import PublishedResignationPolicy
from src.training.checkpoint import CheckpointReference


class SelfPlayGroup:
    def __init__(self, game: GameImplementation) -> None:
        self.game = game
        self._closed = False
        self._context = multiprocessing.get_context('spawn')
        topology = game.training.topology.self_play
        self._device_ids = topology.device_ids
        self._connections: list[Connection] = []
        self._processes: list[BaseProcess] = []
        for worker_id, device_id in enumerate(self._device_ids):
            parent, process = self._start_worker(worker_id, device_id)
            self._connections.append(parent)
            self._processes.append(process)

    @property
    def worker_count(self) -> int:
        return len(self._connections)

    def apply(
        self,
        desired_states: tuple[RunningSelfPlayState | StoppedSelfPlayState, ...],
    ) -> tuple[SelfPlayStateApplied, ...]:
        if self._closed:
            raise RuntimeError('Self-play group is closed.')
        if len(desired_states) != self.worker_count:
            raise ValueError('Every self-play worker requires exactly one desired state.')
        for connection, desired_state in zip(self._connections, desired_states):
            connection.send(desired_state)
        return tuple(self._receive(connection) for connection in self._connections)

    def apply_to_workers(
        self,
        worker_ids: tuple[int, ...],
        desired_state: RunningSelfPlayState,
    ) -> tuple[SelfPlayStateApplied, ...]:
        selected_connections = self._connections_for_workers(worker_ids)
        for connection in selected_connections:
            connection.send(desired_state)
        return tuple(self._receive(connection) for connection in selected_connections)

    def request_pause(self, worker_ids: tuple[int, ...]) -> None:
        for connection in self._connections_for_workers(worker_ids):
            connection.send(PausedSelfPlayState())

    def restart_exited_workers(
        self,
        checkpoint: CheckpointReference,
        resignation_policy: PublishedResignationPolicy,
    ) -> tuple[int, ...]:
        if self._closed:
            raise RuntimeError('Self-play group is closed.')
        restarted_worker_ids: list[int] = []
        for worker_id, process in enumerate(self._processes):
            if process.is_alive():
                continue
            process.join()
            self._connections[worker_id].close()
            connection, replacement = self._start_worker(worker_id, self._device_ids[worker_id])
            self._connections[worker_id] = connection
            self._processes[worker_id] = replacement
            connection.send(RunningSelfPlayState(checkpoint=checkpoint, resignation_policy=resignation_policy))
            response = self._receive(connection)
            if response.kind != 'running':
                raise RuntimeError(f'Restarted self-play worker {worker_id} did not enter the running state.')
            restarted_worker_ids.append(worker_id)
        return tuple(restarted_worker_ids)

    def close(self) -> None:
        if self._closed:
            return
        responses = self.apply(tuple(StoppedSelfPlayState() for _ in self._connections))
        if any(response.kind != 'stopped' for response in responses):
            raise RuntimeError('Self-play worker did not acknowledge shutdown.')
        for process in self._processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f'Self-play worker exited with code {process.exitcode}.')
        for connection in self._connections:
            connection.close()
        self._closed = True

    @staticmethod
    def _receive(connection: Connection) -> SelfPlayStateApplied:
        try:
            response: SelfPlayStateApplied = connection.recv()
        except EOFError as error:
            raise RuntimeError('Self-play worker connection closed unexpectedly.') from error
        return response

    def _connections_for_workers(self, worker_ids: tuple[int, ...]) -> tuple[Connection, ...]:
        if self._closed:
            raise RuntimeError('Self-play group is closed.')
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError('Selected self-play worker IDs must be unique.')
        if any(worker_id < 0 or worker_id >= self.worker_count for worker_id in worker_ids):
            raise ValueError('Selected self-play worker ID is outside the worker range.')
        return tuple(self._connections[worker_id] for worker_id in worker_ids)

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

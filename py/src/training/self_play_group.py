from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
import random

import numpy as np
import torch

from src.games.implementation import GameImplementation
from src.games.composition import create_game_implementation
from src.experiment.configuration import load_experiment_configuration_json
from src.self_play.protocol import (
    PausedSelfPlayState,
    RunningSelfPlayState,
    RunningSelfPlayStateApplied,
    SelfPlayDesiredState,
    SelfPlayStateApplied,
    SelfPlayStatistics,
    StatisticsLevel,
    StoppedSelfPlayState,
    StoppedSelfPlayStateApplied,
)
from src.self_play.worker import SelfPlayWorker
from src.self_play.resignation import PublishedResignationPolicy
from src.training.checkpoint import CheckpointReference
from src.util.tensorboard import TensorboardWriter


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
            target=_self_play_worker_main,
            args=(child, worker_id, device_id, self.game.configuration.model_dump_json()),
            name=f'self-play-worker-{worker_id}',
        )
        process.start()
        child.close()
        return parent, process


class _SelfPlayProcessRuntime:
    def __init__(self, worker: SelfPlayWorker, worker_id: int) -> None:
        self.worker = worker
        self.worker_id = worker_id
        self.loaded_generation: int | None = None
        self.loaded_sha256: str | None = None
        self.completed_search_batches = 0
        self.running = False

    def run_batch(self) -> None:
        self.worker.run_batch()
        self.completed_search_batches += 1

    def apply(self, desired_state: SelfPlayDesiredState) -> SelfPlayStateApplied | None:
        match desired_state:
            case PausedSelfPlayState():
                self.running = False
                return None
            case StoppedSelfPlayState():
                return StoppedSelfPlayStateApplied(worker_id=self.worker_id)
            case RunningSelfPlayState():
                return self._apply_running_state(desired_state)

    def _apply_running_state(self, desired_state: RunningSelfPlayState) -> RunningSelfPlayStateApplied:
        checkpoint = desired_state.checkpoint
        self._validate_checkpoint_transition(checkpoint, desired_state.completed_generation_statistics)
        statistics = self._completed_generation_statistics(desired_state.completed_generation_statistics)
        self.worker.update_resignation_policy(desired_state.resignation_policy)
        self._load_checkpoint(checkpoint)
        self.running = True
        return RunningSelfPlayStateApplied(
            worker_id=self.worker_id,
            loaded_generation=checkpoint.generation,
            loaded_inference_model_sha256=checkpoint.inference_model_sha256,
            completed_generation_statistics=statistics,
        )

    def _validate_checkpoint_transition(
        self,
        checkpoint: CheckpointReference,
        statistics_level: StatisticsLevel | None,
    ) -> None:
        if self.loaded_generation is not None and checkpoint.generation < self.loaded_generation:
            raise ValueError('Self-play model generation cannot move backwards.')
        if statistics_level is not None:
            if self.loaded_generation is None:
                raise ValueError('Cannot collect generation statistics before loading a model.')
            if checkpoint.generation <= self.loaded_generation:
                raise ValueError('Completed-generation statistics require a newer checkpoint.')
        if checkpoint.generation == self.loaded_generation and checkpoint.inference_model_sha256 != self.loaded_sha256:
            raise ValueError('Loaded model generation changed immutable inference identity.')

    def _completed_generation_statistics(self, statistics_level: StatisticsLevel | None) -> SelfPlayStatistics | None:
        if statistics_level is None:
            return None
        assert self.loaded_generation is not None
        if statistics_level is StatisticsLevel.DETAILED:
            self.worker.snapshot_statistics()
        statistics = SelfPlayStatistics(
            completed_generation=self.loaded_generation,
            level=statistics_level,
            completed_search_batches=self.completed_search_batches,
        )
        self.completed_search_batches = 0
        return statistics

    def _load_checkpoint(self, checkpoint: CheckpointReference) -> None:
        if checkpoint.generation == self.loaded_generation:
            return
        checkpoint.validate_inference_model()
        self.worker.refresh_published_model(checkpoint)
        self.loaded_generation = checkpoint.generation
        self.loaded_sha256 = checkpoint.inference_model_sha256


def _self_play_worker_main(
    connection: Connection,
    worker_id: int,
    device_id: int,
    configuration_json: str,
) -> None:
    configuration = load_experiment_configuration_json(configuration_json)
    game = create_game_implementation(configuration)
    seed = game.training.random_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if game.training.topology.trainer.device_type == 'cuda':
        torch.cuda.set_device(device_id)
    worker = SelfPlayWorker(
        game,
        game.training.topology.self_play.parallel_games_per_process,
        worker_id,
        device_id,
        Path(game.training.save_path) / 'completed-games' / 'inbox',
    )
    runtime = _SelfPlayProcessRuntime(worker, worker_id)
    try:
        tensorboard_enabled = worker_id < game.training.topology.self_play.tensorboard_processes
        with TensorboardWriter(run=0, suffix='self_play', enabled=tensorboard_enabled):
            while True:
                if runtime.running and not connection.poll():
                    runtime.run_batch()
                    continue
                desired_state: SelfPlayDesiredState = connection.recv()
                applied_state = runtime.apply(desired_state)
                if applied_state is None:
                    continue
                connection.send(applied_state)
                match applied_state:
                    case StoppedSelfPlayStateApplied():
                        return
                    case _:
                        pass
    finally:
        worker.close()
        connection.close()

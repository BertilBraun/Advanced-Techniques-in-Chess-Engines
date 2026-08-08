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
from src.experiment.configuration import ExperimentConfiguration
from src.self_play.protocol import (
    PausedSelfPlayState,
    PausedSelfPlayStateApplied,
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

    def apply(self, desired_states: tuple[SelfPlayDesiredState, ...]) -> tuple[SelfPlayStateApplied, ...]:
        if self._closed:
            raise RuntimeError('Self-play group is closed.')
        if len(desired_states) != self.worker_count:
            raise ValueError('Every self-play worker requires exactly one desired state.')
        for connection, desired_state in zip(self._connections, desired_states):
            connection.send(desired_state)
        return tuple(self._receive(connection) for connection in self._connections)

    def restart_exited_workers(self, checkpoint: CheckpointReference) -> tuple[int, ...]:
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
            connection.send(RunningSelfPlayState(checkpoint=checkpoint))
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

    def _start_worker(self, worker_id: int, device_id: int) -> tuple[Connection, BaseProcess]:
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_self_play_worker_main,
            args=(child, worker_id, device_id, self.game.configuration),
            name=f'self-play-worker-{worker_id}',
        )
        process.start()
        child.close()
        return parent, process


def _self_play_worker_main(
    connection: Connection,
    worker_id: int,
    device_id: int,
    configuration: ExperimentConfiguration,
) -> None:
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
    loaded_generation: int | None = None
    loaded_sha256: str | None = None
    completed_search_batches = 0
    running = False
    try:
        while True:
            if running and not connection.poll():
                worker.run_batch()
                completed_search_batches += 1
                continue
            desired_state: SelfPlayDesiredState = connection.recv()
            match desired_state:
                case PausedSelfPlayState():
                    running = False
                    connection.send(PausedSelfPlayStateApplied(worker_id=worker_id))
                case StoppedSelfPlayState():
                    connection.send(StoppedSelfPlayStateApplied(worker_id=worker_id))
                    return
                case RunningSelfPlayState():
                    statistics = None
                    checkpoint = desired_state.checkpoint
                    if loaded_generation is not None and checkpoint.generation < loaded_generation:
                        raise ValueError('Self-play model generation cannot move backwards.')
                    if desired_state.completed_generation_statistics is not None:
                        if loaded_generation is None:
                            raise ValueError('Cannot collect generation statistics before loading a model.')
                        if checkpoint.generation <= loaded_generation:
                            raise ValueError('Completed-generation statistics require a newer checkpoint.')
                        if desired_state.completed_generation_statistics is StatisticsLevel.DETAILED:
                            worker.snapshot_statistics()
                        statistics = SelfPlayStatistics(
                            completed_generation=loaded_generation,
                            level=desired_state.completed_generation_statistics,
                            completed_search_batches=completed_search_batches,
                        )
                        completed_search_batches = 0
                    if loaded_generation != checkpoint.generation:
                        checkpoint.validate_inference_model()
                        worker.refresh_published_model(checkpoint)
                        loaded_generation = checkpoint.generation
                        loaded_sha256 = checkpoint.inference_model_sha256
                    elif loaded_sha256 != checkpoint.inference_model_sha256:
                        raise ValueError('Loaded model generation changed immutable inference identity.')
                    running = True
                    connection.send(
                        RunningSelfPlayStateApplied(
                            worker_id=worker_id,
                            loaded_generation=checkpoint.generation,
                            loaded_inference_model_sha256=checkpoint.inference_model_sha256,
                            completed_generation_statistics=statistics,
                        )
                    )
    finally:
        connection.close()

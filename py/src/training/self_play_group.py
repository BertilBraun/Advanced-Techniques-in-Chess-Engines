from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
import random

import numpy as np
import torch

from src.games.implementation import GameImplementation
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


class SelfPlayGroup:
    def __init__(self, run_id: int, game: GameImplementation) -> None:
        self.game = game
        self._closed = False
        context = multiprocessing.get_context('spawn')
        topology = game.training.topology.self_play
        self._connections: list[Connection] = []
        self._processes: list[BaseProcess] = []
        worker_id = 0
        for device_id in topology.device_ids:
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=_self_play_worker_main,
                args=(child, run_id, worker_id, device_id, game),
                name=f'self-play-worker-{worker_id}',
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
            worker_id += 1

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


def _self_play_worker_main(
    connection: Connection,
    run_id: int,
    worker_id: int,
    device_id: int,
    game: GameImplementation,
) -> None:
    seed = game.training.random_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if game.training.topology.trainer.device_type == 'cuda':
        torch.cuda.set_device(device_id)
    del run_id
    policy = game.create_self_play_policy(device_id, worker_id)
    worker = SelfPlayWorker(
        policy,
        game.training.topology.self_play.parallel_games_per_process,
        worker_id,
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
                            worker.snapshot_statistics(loaded_generation)
                        statistics = SelfPlayStatistics(
                            completed_generation=loaded_generation,
                            level=desired_state.completed_generation_statistics,
                            completed_search_batches=completed_search_batches,
                        )
                        completed_search_batches = 0
                    if loaded_generation != checkpoint.generation:
                        checkpoint.validate_inference_model()
                        worker.refresh_published_model(checkpoint.generation, checkpoint.inference_model_path)
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

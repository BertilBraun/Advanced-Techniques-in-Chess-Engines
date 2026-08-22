from __future__ import annotations

import random
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import torch
from src.experiment.configuration import load_experiment_configuration_json
from src.games.composition import create_game_implementation
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
from src.training.checkpoint import CheckpointReference
from src.util.tensorboard import TensorboardWriter


class SelfPlayProcessRuntime:
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


def self_play_worker_main(
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
    runtime = SelfPlayProcessRuntime(worker, worker_id)
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

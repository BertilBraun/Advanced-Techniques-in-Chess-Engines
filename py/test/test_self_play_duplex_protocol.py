from dataclasses import dataclass
import hashlib
import multiprocessing
from pathlib import Path
from threading import Thread
from typing import cast

import pytest

from src.games.implementation import GameImplementation
from src.self_play.protocol import (
    PausedSelfPlayState,
    PausedSelfPlayStateApplied,
    RunningSelfPlayState,
    RunningSelfPlayStateApplied,
    StatisticsLevel,
    StoppedSelfPlayState,
    StoppedSelfPlayStateApplied,
)
from src.training.checkpoint import CheckpointReference
from src.training.self_play_group import _self_play_worker_main
import src.training.self_play_group as self_play_group_module


@dataclass(frozen=True)
class _TrainerTopology:
    device_type: str = 'cpu'


@dataclass(frozen=True)
class _SelfPlayTopology:
    parallel_games_per_process: int = 2


@dataclass(frozen=True)
class _Topology:
    trainer: _TrainerTopology = _TrainerTopology()
    self_play: _SelfPlayTopology = _SelfPlayTopology()


@dataclass(frozen=True)
class _Training:
    save_path: str
    random_seed: int = 5
    topology: _Topology = _Topology()


class _Game:
    def __init__(self, run_path: Path) -> None:
        self.training = _Training(save_path=str(run_path))


class _Worker:
    def __init__(
        self,
        game: GameImplementation,
        parallel_game_count: int,
        worker_id: int,
        device_id: int,
        inbox_path: Path,
    ) -> None:
        del game, parallel_game_count, worker_id, device_id, inbox_path
        self.generation: int | None = None

    def run_batch(self) -> None:
        pass

    def refresh_published_model(self, checkpoint: CheckpointReference) -> None:
        self.generation = checkpoint.generation

    def snapshot_statistics(self) -> None:
        assert self.generation == 0


def _checkpoint(tmp_path: Path, generation: int) -> CheckpointReference:
    inference_path = tmp_path / f'model_{generation}.jit.pt'
    inference_path.write_bytes(f'model {generation}'.encode('ascii'))
    return CheckpointReference(
        generation=generation,
        manifest_path=tmp_path / f'checkpoint_{generation}.json',
        model_path=tmp_path / f'model_{generation}.pt',
        optimizer_path=tmp_path / f'optimizer_{generation}.pt',
        inference_model_path=inference_path,
        inference_model_sha256=hashlib.sha256(inference_path.read_bytes()).hexdigest(),
    )


def test_worker_applies_duplex_desired_states_and_reports_transition_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(self_play_group_module, 'SelfPlayWorker', _Worker)
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe(duplex=True)
    process = Thread(
        target=_self_play_worker_main,
        args=(child, 4, 0, cast(GameImplementation, _Game(tmp_path))),
    )
    process.start()

    parent.send(RunningSelfPlayState(checkpoint=_checkpoint(tmp_path, 0)))
    first = parent.recv()
    assert type(first) is RunningSelfPlayStateApplied
    assert first.loaded_generation == 0

    parent.send(PausedSelfPlayState())
    paused = parent.recv()
    assert type(paused) is PausedSelfPlayStateApplied

    parent.send(
        RunningSelfPlayState(
            checkpoint=_checkpoint(tmp_path, 1),
            completed_generation_statistics=StatisticsLevel.DETAILED,
        )
    )
    transitioned = parent.recv()
    assert type(transitioned) is RunningSelfPlayStateApplied
    assert transitioned.loaded_generation == 1
    assert transitioned.completed_generation_statistics is not None
    assert transitioned.completed_generation_statistics.completed_generation == 0

    parent.send(StoppedSelfPlayState())
    stopped = parent.recv()
    assert type(stopped) is StoppedSelfPlayStateApplied
    process.join(timeout=10)
    parent.close()

    assert not process.is_alive()

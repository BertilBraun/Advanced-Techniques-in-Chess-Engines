from dataclasses import dataclass
import hashlib
import multiprocessing
from pathlib import Path
from threading import Thread
from typing import cast

from src.games.implementation import GameImplementation
from src.self_play.active_game import ContinuingGame
from src.self_play.completed_game import GameIdentity
from src.self_play.protocol import (
    PausedSelfPlayState,
    PausedSelfPlayStateApplied,
    RunningSelfPlayState,
    RunningSelfPlayStateApplied,
    StatisticsLevel,
    StoppedSelfPlayState,
    StoppedSelfPlayStateApplied,
)
from src.self_play.worker import GameSelfPlayPolicy
from src.training.checkpoint import CheckpointReference
from src.training.self_play_group import _self_play_worker_main


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


class _Policy(GameSelfPlayPolicy[int, int, int, None]):
    def __init__(self) -> None:
        self.generation: int | None = None

    def refresh_model(self, model_generation: int, model_path: Path, active_games: tuple[int, ...]) -> None:
        del model_path, active_games
        self.generation = model_generation

    def snapshot_statistics(self, tensorboard_step: int) -> None:
        assert tensorboard_step == self.generation

    def new_game(self, identity: GameIdentity) -> int:
        del identity
        return 0

    def build_search_request(self, game: int) -> int:
        return game

    def search_active_games(self, requests: tuple[int, ...]) -> tuple[int, ...]:
        return requests

    def advance_game(self, game: int, request: int, result: int) -> ContinuingGame[int]:
        del request, result
        return ContinuingGame(game + 1)


class _Game:
    def __init__(self, run_path: Path) -> None:
        self.training = _Training(save_path=str(run_path))

    def create_self_play_policy(self, device_id: int, worker_id: int) -> _Policy:
        del device_id, worker_id
        return _Policy()


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


def test_worker_applies_duplex_desired_states_and_reports_transition_statistics(tmp_path: Path) -> None:
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe(duplex=True)
    process = Thread(
        target=_self_play_worker_main,
        args=(child, 1, 4, 0, cast(GameImplementation, _Game(tmp_path))),
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

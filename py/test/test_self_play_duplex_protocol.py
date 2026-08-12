from dataclasses import dataclass
import hashlib
import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from threading import Thread
from types import TracebackType
from typing import cast

import pytest

from src.games.implementation import GameImplementation
from src.experiment.configuration import ExperimentConfiguration
from src.self_play.protocol import (
    PausedSelfPlayState,
    RunningSelfPlayState,
    RunningSelfPlayStateApplied,
    StatisticsLevel,
    StoppedSelfPlayState,
    StoppedSelfPlayStateApplied,
)
from src.training.checkpoint import CheckpointReference
from src.training.configuration import SelfPlayTopologyParams
from src.training.self_play_group import SelfPlayGroup, _self_play_worker_main
import src.training.self_play_group as self_play_group_module


@dataclass(frozen=True)
class _TrainerTopology:
    device_type: str = 'cpu'


@dataclass(frozen=True)
class _SelfPlayTopology:
    parallel_games_per_process: int = 2
    tensorboard_processes: int = 1


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

    def close(self) -> None:
        pass


class _TensorboardWriter:
    def __init__(self, enabled: bool, observed: list[bool]) -> None:
        self.enabled = enabled
        self.observed = observed

    def __enter__(self) -> None:
        self.observed.append(self.enabled)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception_value, traceback


class _Connection:
    def __init__(self, response: RunningSelfPlayStateApplied | None = None) -> None:
        self.response = response
        self.sent: list[RunningSelfPlayState] = []
        self.closed = False

    def send(self, desired_state: RunningSelfPlayState) -> None:
        self.sent.append(desired_state)

    def recv(self) -> RunningSelfPlayStateApplied:
        assert self.response is not None
        return self.response

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self, alive: bool) -> None:
        self.alive = alive
        self.joined = False

    def is_alive(self) -> bool:
        return self.alive

    def join(self) -> None:
        self.joined = True


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
    observed_tensorboard_states: list[bool] = []

    def create_tensorboard_writer(
        run: int,
        suffix: str,
        enabled: bool,
    ) -> _TensorboardWriter:
        assert run == 0
        assert suffix == 'self_play'
        return _TensorboardWriter(enabled, observed_tensorboard_states)

    monkeypatch.setattr(self_play_group_module, 'TensorboardWriter', create_tensorboard_writer)
    fake_game = _Game(tmp_path)

    def create_game(configuration: ExperimentConfiguration) -> GameImplementation:
        del configuration
        return cast(GameImplementation, fake_game)

    monkeypatch.setattr(self_play_group_module, 'create_game_implementation', create_game)
    monkeypatch.setattr(
        self_play_group_module,
        'load_experiment_configuration_json',
        lambda payload: cast(ExperimentConfiguration, fake_game),
    )
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe(duplex=True)
    process = Thread(
        target=_self_play_worker_main,
        args=(child, 0, 0, '{}'),
    )
    process.start()

    parent.send(RunningSelfPlayState(checkpoint=_checkpoint(tmp_path, 0)))
    first = parent.recv()
    assert type(first) is RunningSelfPlayStateApplied
    assert first.loaded_generation == 0

    parent.send(PausedSelfPlayState())
    assert not parent.poll(0.1)
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
    assert observed_tensorboard_states == [True]


def test_group_restarts_only_exited_workers_at_active_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(tmp_path, 3)
    healthy_connection = _Connection()
    exited_connection = _Connection()
    replacement_connection = _Connection(
        RunningSelfPlayStateApplied(
            worker_id=1,
            loaded_generation=checkpoint.generation,
            loaded_inference_model_sha256=checkpoint.inference_model_sha256,
            completed_generation_statistics=None,
        )
    )
    healthy_process = _Process(alive=True)
    exited_process = _Process(alive=False)
    replacement_process = _Process(alive=True)
    group = SelfPlayGroup.__new__(SelfPlayGroup)
    group._closed = False
    group._device_ids = (0, 1)
    group._connections = [cast(Connection, healthy_connection), cast(Connection, exited_connection)]
    group._processes = [cast(BaseProcess, healthy_process), cast(BaseProcess, exited_process)]

    def start_worker(worker_id: int, device_id: int) -> tuple[Connection, BaseProcess]:
        assert (worker_id, device_id) == (1, 1)
        return cast(Connection, replacement_connection), cast(BaseProcess, replacement_process)

    monkeypatch.setattr(group, '_start_worker', start_worker)

    assert group.restart_exited_workers(checkpoint) == (1,)
    assert exited_process.joined
    assert exited_connection.closed
    assert replacement_connection.sent == [RunningSelfPlayState(checkpoint=checkpoint)]


def test_group_applies_state_only_to_selected_workers(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 3)
    connections = [
        _Connection(
            RunningSelfPlayStateApplied(
                worker_id=worker_id,
                loaded_generation=checkpoint.generation,
                loaded_inference_model_sha256=checkpoint.inference_model_sha256,
                completed_generation_statistics=None,
            )
        )
        for worker_id in range(4)
    ]
    group = SelfPlayGroup.__new__(SelfPlayGroup)
    group._closed = False
    group._connections = [cast(Connection, connection) for connection in connections]

    desired_state = RunningSelfPlayState(checkpoint=checkpoint)
    responses = group.apply_to_workers((1, 3), desired_state)

    assert [response.worker_id for response in responses] == [1, 3]
    assert connections[0].sent == []
    assert connections[1].sent == [desired_state]
    assert connections[2].sent == []
    assert connections[3].sent == [desired_state]


@pytest.mark.parametrize(
    'tensorboard_processes,paused_worker_ids',
    [
        (3, ()),
        (1, (0, 0)),
        (1, (-1,)),
        (1, (2,)),
    ],
)
def test_self_play_topology_rejects_invalid_worker_assignments(
    tensorboard_processes: int,
    paused_worker_ids: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        SelfPlayTopologyParams(
            device_ids=(0, 1),
            parallel_games_per_process=8,
            tensorboard_processes=tensorboard_processes,
            node_ids_to_pause_during_training=paused_worker_ids,
        )

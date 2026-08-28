from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from threading import Thread
from types import TracebackType
from typing import cast

import pytest
import src.self_play.process_runtime as process_runtime_module
from src.experiment.configuration import ExperimentConfiguration
from src.games.implementation import GameImplementation
from src.search_budget.calibration import CurveDecisionReason, CurvePublication
from src.search_budget.curve import SearchBudgetCurve, analytic_initial_curve, flat_curve
from src.self_play.process_runtime import self_play_worker_main
from src.self_play.protocol import (
    PausedSelfPlayState,
    RunningSelfPlayState,
    RunningSelfPlayStateApplied,
    StatisticsLevel,
    StoppedSelfPlayState,
    StoppedSelfPlayStateApplied,
)
from src.self_play.resignation import PublishedResignationPolicy
from src.training.checkpoint import CheckpointReference
from src.training.configuration import SelfPlayTopologyParams
from src.training.self_play_group import SelfPlayGroup, SelfPlaySupervision, SelfPlayWorkerSlot
from test_helpers.checkpoints import checkpoint_reference


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

    def refresh_published_model(self, checkpoint: CheckpointReference, search_budget_curve: SearchBudgetCurve) -> None:
        assert search_budget_curve in {flat_curve(), analytic_initial_curve()}
        self.generation = checkpoint.generation

    def update_resignation_policy(self, policy: PublishedResignationPolicy) -> None:
        assert policy == PublishedResignationPolicy()

    def snapshot_statistics(self) -> None:
        assert self.generation == 0

    def search_budget_spend_residual(self) -> int:
        return -1

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

    def poll(self, timeout: float = 0.0) -> bool:
        del timeout
        return self.response is not None

    def recv(self) -> RunningSelfPlayStateApplied:
        assert self.response is not None
        return self.response

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self, alive: bool) -> None:
        self.alive = alive
        self.joined = False
        self.terminated = False
        self.exitcode = None if alive else 1

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.joined = True

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def kill(self) -> None:
        self.terminate()


def _checkpoint(tmp_path: Path, generation: int) -> CheckpointReference:
    return checkpoint_reference(tmp_path, generation, write_inference_model=True)


def _publication(generation: int, adaptive: bool = False) -> CurvePublication:
    return CurvePublication(
        curve=analytic_initial_curve() if adaptive else flat_curve(),
        application_generation=generation,
        decision_reason=CurveDecisionReason.VALIDATED_PENDING if adaptive else CurveDecisionReason.INITIAL,
    )


def test_worker_applies_duplex_desired_states_and_reports_transition_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process_runtime_module, 'SelfPlayWorker', _Worker)
    observed_tensorboard_states: list[bool] = []

    def create_tensorboard_writer(
        run: int,
        suffix: str,
        enabled: bool,
    ) -> _TensorboardWriter:
        assert run == 0
        assert suffix == 'self_play'
        return _TensorboardWriter(enabled, observed_tensorboard_states)

    monkeypatch.setattr(process_runtime_module, 'TensorboardWriter', create_tensorboard_writer)
    fake_game = _Game(tmp_path)

    def create_game(configuration: ExperimentConfiguration) -> GameImplementation:
        del configuration
        return cast(GameImplementation, fake_game)

    monkeypatch.setattr(process_runtime_module, 'create_game_implementation', create_game)
    monkeypatch.setattr(
        process_runtime_module,
        'load_experiment_configuration_json',
        lambda payload: cast(ExperimentConfiguration, fake_game),
    )
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe(duplex=True)
    process = Thread(
        target=self_play_worker_main,
        args=(child, 0, 0, '{}'),
    )
    process.start()

    parent.send(RunningSelfPlayState(checkpoint=_checkpoint(tmp_path, 0), search_budget=_publication(0)))
    first = parent.recv()
    assert type(first) is RunningSelfPlayStateApplied
    assert first.loaded_generation == 0

    parent.send(PausedSelfPlayState())
    assert not parent.poll(0.1)
    parent.send(
        RunningSelfPlayState(
            checkpoint=_checkpoint(tmp_path, 1),
            search_budget=_publication(1, True),
            completed_generation_statistics=StatisticsLevel.DETAILED,
        )
    )
    transitioned = parent.recv()
    assert type(transitioned) is RunningSelfPlayStateApplied
    assert transitioned.loaded_generation == 1
    assert transitioned.completed_generation_statistics is not None
    assert transitioned.completed_generation_statistics.completed_generation == 0
    assert transitioned.completed_generation_statistics.search_budget_spend_residual == -1

    parent.send(StoppedSelfPlayState())
    stopped = parent.recv()
    assert type(stopped) is StoppedSelfPlayStateApplied
    process.join(timeout=10)
    parent.close()

    assert not process.is_alive()
    assert observed_tensorboard_states == [True]


def _applied(
    worker_id: int,
    checkpoint: CheckpointReference,
    search_budget: CurvePublication,
) -> RunningSelfPlayStateApplied:
    return RunningSelfPlayStateApplied(
        worker_id=worker_id,
        loaded_generation=checkpoint.generation,
        loaded_inference_model_sha256=checkpoint.inference_model_sha256,
        search_budget=search_budget,
        completed_generation_statistics=None,
    )


def _group(connections: list[_Connection], processes: list[_Process]) -> SelfPlayGroup:
    group = SelfPlayGroup.__new__(SelfPlayGroup)
    group._closed = False
    group._slots = [
        SelfPlayWorkerSlot(worker_id, worker_id, cast(Connection, connection), cast(BaseProcess, process))
        for worker_id, (connection, process) in enumerate(zip(connections, processes))
    ]
    return group


def test_group_restarts_only_exited_workers_at_active_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(tmp_path, 3)
    search_budget = _publication(3)
    exited_connection = _Connection()
    replacement_connection = _Connection(_applied(1, checkpoint, search_budget))
    exited_process = _Process(alive=False)
    group = _group([_Connection(), exited_connection], [_Process(alive=True), exited_process])

    def start_worker(worker_id: int, device_id: int) -> tuple[Connection, BaseProcess]:
        assert (worker_id, device_id) == (1, 1)
        return cast(Connection, replacement_connection), cast(BaseProcess, _Process(alive=True))

    monkeypatch.setattr(group, '_start_worker', start_worker)
    policy = PublishedResignationPolicy()

    assert group.supervise(checkpoint, search_budget, policy) == SelfPlaySupervision((), ())
    assert group.supervise(checkpoint, search_budget, policy) == SelfPlaySupervision((), ())
    assert group.supervise(checkpoint, search_budget, policy) == SelfPlaySupervision((1,), ())
    assert exited_connection.closed
    assert replacement_connection.sent == [RunningSelfPlayState(checkpoint=checkpoint, search_budget=search_budget)]


def test_group_abandons_a_restart_whose_handshake_never_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(tmp_path, 3)
    search_budget = _publication(3)
    group = _group([_Connection(), _Connection()], [_Process(alive=True), _Process(alive=False)])
    monkeypatch.setattr(
        group,
        '_start_worker',
        lambda worker_id, device_id: (cast(Connection, _Connection()), cast(BaseProcess, _Process(alive=True))),
    )
    policy = PublishedResignationPolicy()
    group.supervise(checkpoint, search_budget, policy)
    group.supervise(checkpoint, search_budget, policy)
    group._slots[1].handshake_deadline = 0.0

    assert group.supervise(checkpoint, search_budget, policy) == SelfPlaySupervision((), (1,))
    assert group.live_worker_count == 1


def test_group_backs_off_before_retrying_a_failed_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(tmp_path, 3)
    search_budget = _publication(3)
    group = _group([_Connection()], [_Process(alive=False)])
    started_worker_ids: list[int] = []

    def start_worker(worker_id: int, device_id: int) -> tuple[Connection, BaseProcess]:
        del device_id
        started_worker_ids.append(worker_id)
        return cast(Connection, _Connection()), cast(BaseProcess, _Process(alive=False))

    monkeypatch.setattr(group, '_start_worker', start_worker)
    policy = PublishedResignationPolicy()
    for _ in range(4):
        group.supervise(checkpoint, search_budget, policy)

    assert started_worker_ids == [0]


def test_group_retires_a_worker_that_does_not_answer_an_applied_state(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 3)
    search_budget = _publication(3)
    connections = [_Connection(_applied(0, checkpoint, search_budget)), _Connection()]
    group = _group(connections, [_Process(alive=True), _Process(alive=True)])

    responses = group.apply((RunningSelfPlayState(checkpoint=checkpoint, search_budget=search_budget),) * 2)

    assert [response.worker_id for response in responses] == [0]
    assert group.live_worker_count == 1


def test_group_applies_state_only_to_selected_workers(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 3)
    search_budget = _publication(3)
    connections = [_Connection(_applied(worker_id, checkpoint, search_budget)) for worker_id in range(4)]
    group = _group(connections, [_Process(alive=True) for _ in range(4)])

    desired_state = RunningSelfPlayState(checkpoint=checkpoint, search_budget=search_budget)
    responses = group.apply_to_workers((1, 3), desired_state)

    assert [response.worker_id for response in responses] == [1, 3]
    assert connections[0].sent == []
    assert connections[1].sent == [desired_state]
    assert connections[2].sent == []
    assert connections[3].sent == [desired_state]


def test_receive_gives_up_when_a_worker_never_answers() -> None:
    parent, child = multiprocessing.get_context('spawn').Pipe(duplex=True)
    try:
        started_at = time.monotonic()
        assert SelfPlayGroup._receive(parent, timeout_seconds=0.05) is None
        assert time.monotonic() - started_at < 5.0
    finally:
        parent.close()
        child.close()


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

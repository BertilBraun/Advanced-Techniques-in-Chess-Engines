from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest


pytest.importorskip('GPUtil')

import src.cluster.CommanderProcess as commander_module
from src.cluster.CommanderProcess import CommanderProcess
from src.train.TrainingArgs import TrainingArgs
from src.train.TrainingStats import TrainingStats


@dataclass(frozen=True)
class _ClusterSettings:
    self_play_node_ids_to_pause_during_training: tuple[int, ...] = ()


@dataclass(frozen=True)
class _CommanderArguments:
    num_iterations: int
    save_path: str
    cluster: _ClusterSettings


@dataclass(frozen=True)
class _ReplayStatistics:
    num_samples: int = 0
    num_games: int = 0


class _Trainer:
    replay_stats = _ReplayStatistics()

    def wait_for_enough_training_samples(
        self,
        iteration: int,
        stop_reason: Callable[[], str | None],
    ) -> bool:
        assert iteration == 0
        assert stop_reason() is None
        return True

    def load_all_memories_to_train_on_for_iteration(self, iteration: int) -> None:
        assert iteration == 0

    def train(self, iteration: int) -> TrainingStats:
        assert iteration == 0
        return cast(TrainingStats, object())


class _Gating:
    def run(self, iteration: int, current_best_iteration: int) -> int:
        return current_best_iteration


class _Communication:
    def boardcast(self, message: str) -> None:
        assert message == 'START AT ITERATION: 0'


def test_training_reaps_evaluations_without_waiting_for_active_ones(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commander = CommanderProcess.__new__(CommanderProcess)
    commander.args = cast(
        TrainingArgs,
        _CommanderArguments(
            num_iterations=1,
            save_path=str(tmp_path),
            cluster=_ClusterSettings(),
        ),
    )
    commander.communication = cast(commander_module.Communication, _Communication())
    commander.run_id = 0
    commander.latest_completed_iteration = 0
    commander.final_stop_reason = None
    reap_calls = 0

    def no_stop_reason() -> None:
        return None

    def do_nothing() -> None:
        return None

    def reap_evaluations() -> None:
        nonlocal reap_calls
        reap_calls += 1

    def fail_if_waited() -> bool:
        raise AssertionError('Training must not wait for active evaluations.')

    def no_games(_: int, __: str) -> int:
        return 0

    def ignore_iteration_telemetry(
        iteration: int,
        games_at_wait_start: int,
        games_at_wait_end: int,
        wait_seconds: float,
        replay_samples_loaded: int,
        replay_games_loaded: int,
        training_seconds: float,
    ) -> None:
        assert iteration == 0
        assert games_at_wait_start == games_at_wait_end == 0
        assert wait_seconds >= 0
        assert replay_samples_loaded == replay_games_loaded == 0
        assert training_seconds >= 0

    monkeypatch.setattr(commander, '_stop_reason', no_stop_reason)
    monkeypatch.setattr(commander, '_ensure_processes_are_running', do_nothing)
    monkeypatch.setattr(commander, '_reap_evaluation_processes', reap_evaluations)
    monkeypatch.setattr(commander, '_wait_for_all_evaluations', fail_if_waited)
    monkeypatch.setattr(commander, '_write_iteration_telemetry', ignore_iteration_telemetry)
    monkeypatch.setattr(commander_module, 'number_of_games_in_iteration', no_games)
    iterations = commander._run_iterations(_Trainer(), _Gating(), 0, 0)
    next(iterations)

    assert reap_calls == 1

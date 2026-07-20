from dataclasses import dataclass
from typing import cast

import pytest


pytest.importorskip('GPUtil')

import src.cluster.CommanderProcess as commander_module
from src.cluster.CommanderProcess import CommanderProcess
from src.cluster.TrainerProcess import TrainerProcess
from src.train.TrainingArgs import TrainingArgs
from src.train.TrainingStats import TrainingStats


@dataclass(frozen=True)
class _ClusterArguments:
    self_play_node_ids_to_pause_during_training: tuple[int, ...]


@dataclass(frozen=True)
class _TrainingArguments:
    cluster: _ClusterArguments


class _Trainer:
    def __init__(self, result: TrainingStats | BaseException) -> None:
        self.result = result

    def train(self, iteration: int) -> TrainingStats:
        assert iteration == 3
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def commander() -> CommanderProcess:
    result = CommanderProcess.__new__(CommanderProcess)
    result.args = cast(
        TrainingArgs,
        _TrainingArguments(cluster=_ClusterArguments((1, 4))),
    )
    result.communication = cast(commander_module.Communication, object())
    return result


def test_commander_resumes_paused_workers_after_distributed_training_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    expected_stats = cast(TrainingStats, object())

    def pause(*_: object, **__: object) -> None:
        calls.append('pause')

    def resume(*_: object, **__: object) -> None:
        calls.append('resume')

    monkeypatch.setattr(commander_module, 'pause_self_play_workers', pause)
    monkeypatch.setattr(commander_module, 'resume_self_play_workers', resume)

    result = commander()._train_with_self_play_cleanup(
        cast(TrainerProcess, _Trainer(expected_stats)),
        iteration=3,
    )

    assert result is expected_stats
    assert calls == ['pause', 'resume']


def test_commander_resumes_workers_after_distributed_training_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    training_error = RuntimeError('rank failed')

    def pause(*_: object, **__: object) -> None:
        calls.append('pause')

    def resume(*_: object, **__: object) -> None:
        calls.append('resume')

    monkeypatch.setattr(commander_module, 'pause_self_play_workers', pause)
    monkeypatch.setattr(commander_module, 'resume_self_play_workers', resume)

    with pytest.raises(RuntimeError, match='rank failed') as raised:
        commander()._train_with_self_play_cleanup(
            cast(TrainerProcess, _Trainer(training_error)),
            iteration=3,
        )

    assert raised.value is training_error
    assert calls == ['pause', 'resume']


def test_commander_attempts_resume_when_pause_acknowledgement_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    pause_error = RuntimeError('pause timeout')

    def pause(*_: object, **__: object) -> None:
        calls.append('pause')
        raise pause_error

    def resume(*_: object, **__: object) -> None:
        calls.append('resume')

    monkeypatch.setattr(commander_module, 'pause_self_play_workers', pause)
    monkeypatch.setattr(commander_module, 'resume_self_play_workers', resume)

    with pytest.raises(RuntimeError, match='pause timeout') as raised:
        commander()._train_with_self_play_cleanup(
            cast(TrainerProcess, _Trainer(cast(TrainingStats, object()))),
            iteration=3,
        )

    assert raised.value is pause_error
    assert calls == ['pause', 'resume']

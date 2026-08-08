from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import sys
from types import ModuleType
from typing import cast

import pytest

sys.modules.setdefault('GPUtil', ModuleType('GPUtil'))

import src.training.commander as commander_module
from src.training.commander import CommanderProcess, TrainingLifecycle
from src.games.chess.evaluation.scheduler import CreditEvaluationScheduler
from src.training.trainer_process import QuantumResult, ReplayState, TrainerProcess
from src.training.ledger import CreditTrainingLedger
from src.training.configuration import CreditTrainingParams, ReplayConfiguration, TrainingArgs


@dataclass(frozen=True)
class _ClusterArguments:
    node_ids_to_pause_during_training: tuple[int, ...]


@dataclass(frozen=True)
class _TrainingArguments:
    topology: _TopologyArguments
    trainer: _TrainingConfiguration
    lifecycle: _LifecycleArguments


@dataclass(frozen=True)
class _LifecycleArguments:
    replay: ReplayConfiguration


@dataclass(frozen=True)
class _TopologyArguments:
    self_play: _ClusterArguments


@dataclass(frozen=True)
class _TrainingConfiguration:
    global_batch_size: int


class _Trainer:
    def __init__(self, result: QuantumResult | BaseException) -> None:
        self.result = result

    def train_quantum(self, global_step: int, model_version: int) -> QuantumResult:
        assert global_step == 1_000
        assert model_version == 11
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _ReplayTrainer:
    def __init__(self, credited_unique_samples: int) -> None:
        self.credited_unique_samples = credited_unique_samples
        self.calls: list[int] = []

    def maintain_replay(self, capacity: int) -> ReplayState:
        self.calls.append(capacity)
        return ReplayState(
            credited_unique_samples=self.credited_unique_samples,
            credited_completed_searches=17,
            live_unique_samples=self.credited_unique_samples,
            compacted_container=False,
            oldest_source_model_version=None,
            newest_source_model_version=None,
            weighted_mean_source_model_version_midpoint=None,
            oldest_position_age_seconds=None,
            weighted_mean_position_age_seconds=None,
            evicted_unique_samples=0,
            replay_memory_bytes=0,
        )


def commander() -> CommanderProcess:
    result = CommanderProcess.__new__(CommanderProcess)
    result.args = cast(
        TrainingArgs,
        _TrainingArguments(
            topology=_TopologyArguments(self_play=_ClusterArguments((1, 4))),
            trainer=_TrainingConfiguration(global_batch_size=1),
            lifecycle=_LifecycleArguments(
                replay=ReplayConfiguration(
                    capacity={'kind': 'constant', 'value': 10},
                    maximum_capacity=10,
                    maximum_policy_entries=10,
                )
            ),
        ),
    )
    result.communication = cast(commander_module.Communication, object())
    return result


def _parameters() -> CreditTrainingParams:
    return CreditTrainingParams(
        replay_ratio=Decimal(4),
        optimizer_steps_per_quantum=1,
        maximum_optimizer_steps=10,
        retained_checkpoint_interval_generations=1,
    )


def _lifecycle(tmp_path: Path, trainer: _ReplayTrainer) -> TrainingLifecycle:
    ledger = CreditTrainingLedger(tmp_path, _parameters(), global_batch_size=1)
    return TrainingLifecycle(
        ledger=ledger,
        trainer=cast(TrainerProcess, trainer),
        evaluation_scheduler=cast(CreditEvaluationScheduler, object()),
        previous_progress=ledger.progress,
        previous_credited_completed_searches=0,
        credit_wait_started_at=0,
        credit_observation_started_at=0,
    )


def test_commander_waits_until_enough_credits(tmp_path: Path) -> None:
    replay_trainer = _ReplayTrainer(credited_unique_samples=0)

    observation = commander()._observe_replay_credits(
        _lifecycle(tmp_path, replay_trainer),
        _parameters(),
    )

    assert observation is None
    assert replay_trainer.calls == [10]


def test_commander_observes_enough_credits_for_exactly_one_quantum(tmp_path: Path) -> None:
    replay_trainer = _ReplayTrainer(credited_unique_samples=1)

    observation = commander()._observe_replay_credits(
        _lifecycle(tmp_path, replay_trainer),
        _parameters(),
    )

    assert observation is not None
    assert observation.progress.can_train(4)
    assert replay_trainer.calls == [10]


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
        commander()._train_quantum_with_self_play_cleanup(
            cast(TrainerProcess, _Trainer(cast(QuantumResult, object()))),
            global_step=1_000,
            model_version=11,
        )

    assert raised.value is pause_error
    assert calls == ['pause', 'resume']


@pytest.mark.parametrize('fails', (False, True))
def test_training_quantum_resumes_paused_workers(
    monkeypatch: pytest.MonkeyPatch,
    fails: bool,
) -> None:
    calls: list[str] = []
    expected_result = cast(QuantumResult, object())
    training_error = RuntimeError('credit rank failed')

    def pause(*_: object, **__: object) -> None:
        calls.append('pause')

    def resume(*_: object, **__: object) -> None:
        calls.append('resume')

    monkeypatch.setattr(commander_module, 'pause_self_play_workers', pause)
    monkeypatch.setattr(commander_module, 'resume_self_play_workers', resume)
    trainer = cast(
        TrainerProcess,
        _Trainer(training_error if fails else expected_result),
    )

    if fails:
        with pytest.raises(RuntimeError, match='credit rank failed'):
            commander()._train_quantum_with_self_play_cleanup(
                trainer,
                global_step=1_000,
                model_version=11,
            )
    else:
        result = commander()._train_quantum_with_self_play_cleanup(
            trainer,
            global_step=1_000,
            model_version=11,
        )
        assert result is expected_result

    assert calls == ['pause', 'resume']

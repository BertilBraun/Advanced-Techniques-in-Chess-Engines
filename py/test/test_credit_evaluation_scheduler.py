from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

import src.cluster.CreditEvaluationScheduler as scheduler_module
from src.cluster.CreditEvaluationScheduler import (
    CreditEvaluationScheduler,
    CreditEvaluationStatus,
    credit_evaluation_arguments,
)
from src.settings import TRAINING_ARGS
from src.train.CreditPublication import (
    CreditPublicationManifest,
    PublishedArtifact,
    file_sha256,
    publication_manifest_path,
)
from src.train.TrainingArgs import CreditTrainingParams, EvaluationParams, TrainingArgs, UncachedInferenceParams


EvaluationTarget = Callable[[int, TrainingArgs, EvaluationParams, int, int | None], None]


class _FakeProcess:
    created: list[_FakeProcess] = []

    def __init__(
        self,
        target: EvaluationTarget,
        args: tuple[int, TrainingArgs, EvaluationParams, int, int],
        name: str,
    ) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.pid: int | None = None
        self.exitcode: int | None = None
        self.alive = False
        self.terminated = False
        self.__class__.created.append(self)

    def start(self) -> None:
        self.pid = 10_000 + len(self.__class__.created)
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False
        self.exitcode = -15

    def complete(self, exit_code: int) -> None:
        self.alive = False
        self.exitcode = exit_code


def _arguments(
    run_path: Path,
    maximum_attempts: int = 3,
    retry_backoff_seconds: float = 0,
) -> TrainingArgs:
    parameters = CreditTrainingParams(
        replay_ratio=Decimal(4),
        optimizer_steps_per_quantum=50,
        maximum_optimizer_steps=500_000,
        initial_replay_capacity_unique_positions=100_000,
        maximum_replay_capacity_unique_positions=2_500_000,
        replay_capacity_ramp_model_versions=1_000,
        retained_checkpoint_interval_steps=1_000,
    )
    trainer = TRAINING_ARGS.trainer.validated_copy(update={'global_batch_size': 1_024, 'local_batch_size': 1_024})
    schedule = TRAINING_ARGS.lifecycle.evaluation.validated_copy(
        update={
            'interval_optimizer_steps': 1_000,
            'full_interval_optimizer_steps': 2_000,
            'timeout_seconds': 60,
            'maximum_attempts': maximum_attempts,
            'retry_backoff_seconds': retry_backoff_seconds,
        }
    )
    lifecycle = TRAINING_ARGS.lifecycle.validated_copy(
        update={
            'credit': parameters.model_dump(mode='json'),
            'evaluation': schedule.model_dump(mode='json'),
        }
    )
    return TRAINING_ARGS.validated_copy(
        update={
            'save_path': str(run_path),
            'trainer': trainer.model_dump(mode='json'),
            'lifecycle': lifecycle.model_dump(mode='json'),
        }
    )


def _evaluation() -> EvaluationParams:
    return EvaluationParams(
        num_searches_per_turn=64,
        num_games=2,
        every_n_model_versions=1,
        max_concurrent_tasks=1,
        inference=UncachedInferenceParams(mode='uncached'),
        dataset_path=None,
        reference_model_path=None,
        opening_suite_path=None,
        raw_results_path=None,
        maximum_game_plies=None,
        bootstrap_seed=0,
        bootstrap_samples=10,
        mcts_threads=1,
        previous_model_offsets=(1, 2),
        historical_model_versions=(1, 2, 3),
        historical_model_rotation_period=1,
        stockfish_skill_levels=(),
        stockfish_binary_path=None,
        stockfish_nodes_per_move=1_000,
        stockfish_threads=1,
        stockfish_hash_mib=128,
        evaluate_random=False,
    )


def _publication(run_path: Path, completed_optimizer_steps: int) -> CreditPublicationManifest:
    model_version = completed_optimizer_steps // 50
    presentations = completed_optimizer_steps * 1_024
    jit_path = run_path / f'model_{model_version}.jit.pt'
    jit_path.write_bytes(f'jit-model-{model_version}'.encode())
    digest = file_sha256(jit_path)
    publication = CreditPublicationManifest(
        model_version=model_version,
        completed_optimizer_steps=completed_optimizer_steps,
        trained_position_presentations=presentations,
        global_batch_size=1_024,
        credited_unique_positions=presentations // 4,
        earned_position_credits=Decimal(presentations),
        consumed_position_credits=Decimal(presentations),
        available_position_credits=Decimal(0),
        model=PublishedArtifact(path=f'model_{model_version}.pt', sha256=digest),
        optimizer=PublishedArtifact(path=f'optimizer_{model_version}.pt', sha256=digest),
        jit_model=PublishedArtifact(path=f'model_{model_version}.jit.pt', sha256=digest),
        checkpoint_manifest_path=f'checkpoint_{model_version}.json',
        checkpoint_manifest_sha256=digest,
        source_revision='a' * 40,
        run_configuration_sha256='b' * 64,
    )
    path = publication_manifest_path(run_path, model_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(publication.model_dump_json(indent=2) + '\n', encoding='utf-8')
    return publication


@pytest.fixture(autouse=True)
def fake_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeProcess.created.clear()
    monkeypatch.setattr(scheduler_module, 'Process', _FakeProcess)


def test_slow_evaluation_is_out_of_band_and_coalesces_newest_boundary(tmp_path: Path) -> None:
    scheduler = CreditEvaluationScheduler(1, _arguments(tmp_path), _evaluation())
    first = _publication(tmp_path, 1_000)
    second = _publication(tmp_path, 2_000)

    scheduler.offer(first)
    scheduler.poll()
    assert scheduler.state.active is not None
    assert scheduler.state.active.source.model_version == 20
    assert _FakeProcess.created[0].args[3] == 20
    assert _FakeProcess.created[0].args[4] == 20
    scheduler.offer(second)
    scheduler.poll()
    assert scheduler.state.pending is not None
    assert scheduler.state.pending.source.model_version == 40

    first_process = _FakeProcess.created[0]
    first_process.complete(1)
    scheduler.poll()
    assert scheduler.state.results[-1].status is CreditEvaluationStatus.COALESCED
    assert scheduler.state.pending is not None
    assert scheduler.state.pending.source.model_version == 40

    scheduler.poll()
    assert scheduler.state.active is not None
    assert scheduler.state.active.source.model_version == 40
    assert scheduler.pinned_model_versions == frozenset({40})


def test_initial_publication_starts_model_zero_evaluation(tmp_path: Path) -> None:
    scheduler = CreditEvaluationScheduler(1, _arguments(tmp_path), _evaluation())

    scheduler.offer(_publication(tmp_path, 0))
    scheduler.poll()

    assert scheduler.state.active is not None
    assert scheduler.state.active.source.model_version == 0
    assert scheduler.state.active.source.completed_optimizer_steps == 0


def test_crash_retry_is_bounded_and_later_boundary_still_runs(tmp_path: Path) -> None:
    scheduler = CreditEvaluationScheduler(1, _arguments(tmp_path, maximum_attempts=2), _evaluation())
    scheduler.offer(_publication(tmp_path, 1_000))
    scheduler.poll()
    _FakeProcess.created[-1].complete(1)
    scheduler.poll()
    assert scheduler.state.results[-1].status is CreditEvaluationStatus.RETRY_PENDING

    scheduler.poll()
    _FakeProcess.created[-1].complete(1)
    scheduler.poll()
    assert scheduler.state.results[-1].status is CreditEvaluationStatus.PERMANENT_FAILURE
    assert scheduler.state.pending is None

    scheduler.offer(_publication(tmp_path, 2_000))
    scheduler.poll()
    assert scheduler.state.active is not None
    assert scheduler.state.active.source.completed_optimizer_steps == 2_000


def test_timed_out_evaluation_is_terminated_and_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = CreditEvaluationScheduler(1, _arguments(tmp_path), _evaluation())
    scheduler.offer(_publication(tmp_path, 1_000))
    scheduler.poll()
    active = scheduler.state.active
    assert active is not None
    monkeypatch.setattr(
        scheduler_module.time,
        'time',
        lambda: active.started_at_seconds + scheduler.timeout_seconds + 1,
    )

    scheduler.poll()

    assert _FakeProcess.created[-1].terminated
    assert scheduler.state.active is None
    assert scheduler.state.pending is not None
    assert scheduler.state.results[-1].status is CreditEvaluationStatus.RETRY_PENDING


def test_scheduler_restart_recovers_active_job_as_bounded_retry(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    scheduler = CreditEvaluationScheduler(1, arguments, _evaluation())
    scheduler.offer(_publication(tmp_path, 1_000))
    scheduler.poll()
    assert scheduler.pinned_model_versions == frozenset({20})

    restarted = CreditEvaluationScheduler(1, arguments, _evaluation())
    assert restarted.state.active is None
    assert restarted.state.pending is not None
    assert restarted.state.pending.next_attempt == 2
    assert restarted.state.results[-1].failure is not None
    assert restarted.pinned_model_versions == frozenset({20})


def test_close_records_an_already_completed_evaluation_as_success(tmp_path: Path) -> None:
    scheduler = CreditEvaluationScheduler(1, _arguments(tmp_path), _evaluation())
    scheduler.offer(_publication(tmp_path, 1_000))
    scheduler.poll()
    _FakeProcess.created[-1].complete(0)

    scheduler.close()

    assert scheduler.state.active is None
    assert scheduler.state.results[-1].status is CreditEvaluationStatus.SUCCEEDED


def test_pending_start_failure_advances_attempt_instead_of_spinning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = CreditEvaluationScheduler(1, _arguments(tmp_path, maximum_attempts=2), _evaluation())
    scheduler.offer(_publication(tmp_path, 1_000))

    def reject_arguments(_args: TrainingArgs, _optimizer_steps: int) -> TrainingArgs:
        raise ValueError('invalid evaluation mapping')

    monkeypatch.setattr(scheduler_module, 'credit_evaluation_arguments', reject_arguments)
    scheduler.poll()

    assert scheduler.state.active is None
    assert scheduler.state.pending is not None
    assert scheduler.state.pending.next_attempt == 2
    assert scheduler.state.results[-1].status is CreditEvaluationStatus.RETRY_PENDING


def test_scheduler_rejects_tampered_evaluation_artifact_before_launch(tmp_path: Path) -> None:
    scheduler = CreditEvaluationScheduler(1, _arguments(tmp_path, maximum_attempts=2), _evaluation())
    publication = _publication(tmp_path, 1_000)
    scheduler.offer(publication)
    (tmp_path / publication.jit_model.path).write_bytes(b'tampered')

    scheduler.poll()

    assert not _FakeProcess.created
    assert scheduler.state.pending is not None
    assert scheduler.state.pending.next_attempt == 2
    assert scheduler.state.results[-1].status is CreditEvaluationStatus.RETRY_PENDING


def test_credit_evaluation_offsets_are_evaluation_checkpoint_ordinals(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    translated, evaluation = credit_evaluation_arguments(arguments, _evaluation(), 2_000)

    assert evaluation.previous_model_offsets == (20, 40)
    assert evaluation.historical_model_versions == (20, 40, 60)
    assert evaluation.every_n_model_versions == 20
    assert translated.lifecycle.inference_retention.milestone_interval == 20

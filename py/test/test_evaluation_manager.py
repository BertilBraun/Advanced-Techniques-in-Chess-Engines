from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from src.evaluation.contracts import (
    CheckpointOpponent,
    EvaluationFailurePhase,
    EvaluationResult,
    FailedEvaluationResult,
    FixedDatasetEvaluationJob,
    FixedDatasetEvaluationResult,
    MatchEvaluationJob,
)
from src.evaluation.manager import EvaluationManager
from src.evaluation.process import write_evaluation_result
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.training.checkpoint import CheckpointReference


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeProcess:
    def __init__(
        self,
        target: Callable[..., None],
        args: tuple[object, ...],
        name: str,
        events: list[str],
    ) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.exitcode: int | None = None
        self.started = False
        self.events = events

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started and self.exitcode is None

    def terminate(self) -> None:
        self.events.append(f'terminate:{self.name}')
        self.exitcode = -15

    def join(self) -> None:
        self.events.append(f'join:{self.name}')
        if self.exitcode is None:
            self.exitcode = 0


class FakeProcessContext:
    def __init__(self) -> None:
        self.processes: list[FakeProcess] = []
        self.events: list[str] = []

    def Process(
        self,
        target: Callable[..., None],
        args: tuple[object, ...],
        name: str,
    ) -> FakeProcess:
        process = FakeProcess(target, args, name, self.events)
        self.processes.append(process)
        return process


def checkpoint(run_path: Path, generation: int) -> CheckpointReference:
    return CheckpointReference(
        generation=generation,
        manifest_path=run_path / f'checkpoint-{generation}.json',
        model_path=run_path / f'model-{generation}.pt',
        optimizer_path=run_path / f'optimizer-{generation}.pt',
        inference_model_path=run_path / f'inference-{generation}.pt',
        inference_model_sha256='0' * 64,
    )


def experiment_configuration(
    run_path: Path,
    job_timeout_seconds: float = 10.0,
    shutdown_grace_seconds: float = 0.1,
) -> ChessExperimentConfiguration:
    loaded = load_experiment_configuration(Path('configs/chess-experiment-template.yaml'))
    assert isinstance(loaded, ChessExperimentConfiguration)
    training = loaded.training.model_copy(
        update={
            'save_path': str(run_path),
            'topology': loaded.training.topology.model_copy(
                update={'evaluation': loaded.training.topology.evaluation.model_copy(update={'device_cycle': (2, 5)})}
            ),
        }
    )
    return loaded.model_copy(
        update={
            'training': training,
            'evaluation': loaded.evaluation.model_copy(
                update={
                    'cadence_seconds': 20,
                    'job_timeout_seconds': job_timeout_seconds,
                    'shutdown_grace_seconds': shutdown_grace_seconds,
                }
            ),
        }
    )


def test_manager_schedules_boundary_checkpoint_and_cycles_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = experiment_configuration(tmp_path)
    clock = FakeClock()
    context = FakeProcessContext()
    manager = EvaluationManager(experiment, checkpoint(tmp_path, 0), clock, context)

    clock.now = 19.0
    assert manager.schedule_due_jobs(checkpoint(tmp_path, 1)) == ()

    clock.now = 21.0
    first_jobs = manager.schedule_due_jobs(checkpoint(tmp_path, 2))
    assert {job.candidate.generation for job in first_jobs} == {1}
    assert tuple(job.device_id for job in first_jobs) == (2, 5, 2, 5, 2, 5, 2)
    assert all(process.started for process in context.processes)

    scalar_steps: list[int | None] = []
    monkeypatch.setattr(
        'src.evaluation.manager.log_scalar',
        lambda name, value, step: scalar_steps.append(step),
    )
    dataset_job = next(job for job in first_jobs if isinstance(job, FixedDatasetEvaluationJob))
    write_evaluation_result(
        FixedDatasetEvaluationResult(
            kind='fixed_dataset',
            job=dataset_job,
            position_count=500,
            source_game_count=25,
            top_action_accuracy=0.25,
            policy_cross_entropy=2.0,
            duration_seconds=1.0,
        ),
        dataset_job.result_path,
    )
    dataset_process = next(process for process in context.processes if process.args[1] == dataset_job)
    dataset_process.exitcode = 0
    assert manager.collect_completed_jobs()[0].job == dataset_job
    assert scalar_steps == [20, 20]

    clock.now = 41.0
    second_jobs = manager.schedule_due_jobs(checkpoint(tmp_path, 3))
    assert {job.candidate.generation for job in second_jobs} == {2}
    assert tuple(job.device_id for job in second_jobs) == (5, 2, 5, 2, 5, 2, 5, 2)
    previous_jobs = tuple(
        job
        for job in second_jobs
        if isinstance(job, MatchEvaluationJob) and isinstance(job.opponent, CheckpointOpponent)
    )
    assert len(previous_jobs) == 1
    assert previous_jobs[0].opponent.checkpoint.generation == 1

    clock.now = 61.0
    manager.schedule_due_jobs(checkpoint(tmp_path, 4))
    clock.now = 81.0
    fourth_jobs = manager.schedule_due_jobs(checkpoint(tmp_path, 5))
    previous_jobs = tuple(
        job
        for job in fourth_jobs
        if isinstance(job, MatchEvaluationJob) and isinstance(job.opponent, CheckpointOpponent)
    )
    assert tuple(job.opponent.checkpoint.generation for job in previous_jobs) == tuple(
        4 - offset for offset in (1, 2, 3)
    )
    assert {1, 2, 3, 4, *range(10, 101, 10)} <= set(manager.required_checkpoint_generations)


def test_manager_schedules_only_available_older_fixed_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    context = FakeProcessContext()
    manager = EvaluationManager(experiment_configuration(tmp_path), checkpoint(tmp_path, 0), clock, context)
    (tmp_path / 'checkpoint_10.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(
        CheckpointReference,
        'load_for_inference',
        classmethod(lambda _cls, run_path, generation: checkpoint(run_path, generation)),
    )

    clock.now = 19.0
    manager.schedule_due_jobs(checkpoint(tmp_path, 20))
    clock.now = 21.0
    jobs = manager.schedule_due_jobs(checkpoint(tmp_path, 21))

    fixed_opponents = tuple(
        job.opponent.checkpoint.generation
        for job in jobs
        if isinstance(job, MatchEvaluationJob)
        and isinstance(job.opponent, CheckpointOpponent)
        and job.definition.kind == 'fixed_checkpoint'
    )
    assert fixed_opponents == (10,)


def test_manager_publishes_missing_artifact_and_deadline_failures(tmp_path: Path) -> None:
    clock = FakeClock()
    context = FakeProcessContext()
    manager = EvaluationManager(
        experiment_configuration(tmp_path),
        checkpoint(tmp_path, 0),
        clock,
        context,
    )
    clock.now = 21.0
    jobs = manager.schedule_due_jobs(checkpoint(tmp_path, 1))
    context.processes[0].exitcode = 3

    missing = manager.collect_completed_jobs()

    assert len(missing) == 1
    assert isinstance(missing[0], FailedEvaluationResult)
    assert missing[0].phase is EvaluationFailurePhase.MISSING_ARTIFACT

    clock.now = 32.0
    deadline_failures = manager.collect_completed_jobs()
    assert len(deadline_failures) == len(jobs) - 1
    assert all(
        isinstance(result, FailedEvaluationResult) and result.phase is EvaluationFailurePhase.DEADLINE
        for result in deadline_failures
    )


def test_manager_cancels_running_jobs_on_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    context = FakeProcessContext()
    manager = EvaluationManager(
        experiment_configuration(tmp_path),
        checkpoint(tmp_path, 0),
        clock,
        context,
    )
    clock.now = 21.0
    jobs = manager.schedule_due_jobs(checkpoint(tmp_path, 1))
    monkeypatch.setattr('src.evaluation.manager.time.sleep', lambda seconds: setattr(clock, 'now', clock.now + seconds))

    manager.close()

    assert all(job.result_path.exists() for job in jobs)
    assert all(process.exitcode == -15 for process in context.processes)
    first_join = next(index for index, event in enumerate(context.events) if event.startswith('join:'))
    last_terminate = max(index for index, event in enumerate(context.events) if event.startswith('terminate:'))
    assert last_terminate < first_join
    results = tuple(
        TypeAdapter(EvaluationResult).validate_json(job.result_path.read_text(encoding='utf-8')) for job in jobs
    )
    assert all(
        isinstance(result, FailedEvaluationResult) and result.phase is EvaluationFailurePhase.CANCELLED
        for result in results
    )


def test_manager_relaunches_unfinished_jobs_after_restart(tmp_path: Path) -> None:
    first_clock = FakeClock()
    first_context = FakeProcessContext()
    first_manager = EvaluationManager(
        experiment_configuration(tmp_path),
        checkpoint(tmp_path, 0),
        first_clock,
        first_context,
    )
    first_clock.now = 21.0
    jobs = first_manager.schedule_due_jobs(checkpoint(tmp_path, 1))
    candidate = checkpoint(tmp_path, 0)
    candidate.manifest_path.write_text('{}', encoding='utf-8')
    candidate.inference_model_path.write_bytes(b'model')

    restarted_context = FakeProcessContext()
    EvaluationManager(
        experiment_configuration(tmp_path),
        candidate,
        FakeClock(),
        restarted_context,
    )

    assert len(restarted_context.processes) == len(jobs)
    assert all(process.started for process in restarted_context.processes)

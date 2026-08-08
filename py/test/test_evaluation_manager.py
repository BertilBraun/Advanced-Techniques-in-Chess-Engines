from collections.abc import Callable
from pathlib import Path

import pytest

from src.evaluation.contracts import (
    CheckpointOpponent,
    FixedDatasetEvaluationJob,
    FixedDatasetEvaluationResult,
    MatchEvaluationJob,
)
from src.evaluation.manager import EvaluationManager
from src.evaluation.process import write_evaluation_result
from src.experiment.configuration import load_experiment_configuration
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
    ) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.exitcode: int | None = None
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started and self.exitcode is None

    def terminate(self) -> None:
        self.exitcode = -15

    def join(self) -> None:
        if self.exitcode is None:
            self.exitcode = 0


class FakeProcessContext:
    def __init__(self) -> None:
        self.processes: list[FakeProcess] = []

    def Process(
        self,
        target: Callable[..., None],
        args: tuple[object, ...],
        name: str,
    ) -> FakeProcess:
        process = FakeProcess(target, args, name)
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


def test_manager_schedules_boundary_checkpoint_and_cycles_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_experiment_configuration(Path('configs/chess-experiment-template.yaml'))
    training = loaded.training.model_copy(
        update={
            'save_path': str(tmp_path),
            'topology': loaded.training.topology.model_copy(
                update={'evaluation': loaded.training.topology.evaluation.model_copy(update={'device_cycle': (2, 5)})}
            ),
        }
    )
    experiment = loaded.model_copy(
        update={
            'training': training,
            'evaluation': loaded.evaluation.model_copy(update={'cadence_seconds': 20}),
        }
    )
    clock = FakeClock()
    context = FakeProcessContext()
    manager = EvaluationManager(experiment, checkpoint(tmp_path, 0), clock, context)

    clock.now = 19.0
    assert manager.schedule_due_jobs(checkpoint(tmp_path, 1)) == ()

    clock.now = 21.0
    first_jobs = manager.schedule_due_jobs(checkpoint(tmp_path, 2))
    assert {job.candidate.generation for job in first_jobs} == {1}
    assert tuple(job.device_id for job in first_jobs) == (2, 5, 2)
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
    assert tuple(job.device_id for job in second_jobs) == (5, 2, 5, 2)
    previous_jobs = tuple(
        job
        for job in second_jobs
        if isinstance(job, MatchEvaluationJob) and isinstance(job.opponent, CheckpointOpponent)
    )
    assert len(previous_jobs) == 1
    assert previous_jobs[0].opponent.checkpoint.generation == 1

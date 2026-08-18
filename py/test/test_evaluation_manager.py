from collections.abc import Callable
import hashlib
from pathlib import Path
import pickle

import pytest
from pydantic import TypeAdapter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.custom_scalar import layout_pb2

from src.evaluation.contracts import (
    CheckpointOpponent,
    EvaluationFailurePhase,
    EVALUATION_JOB_ADAPTER,
    EvaluationReferenceManifest,
    EvaluationResult,
    FailedEvaluationResult,
    FixedDatasetEvaluationJob,
    FixedDatasetEvaluationResult,
    MatchEvaluationJob,
    ElapsedCheckpointReference,
)
from src.evaluation.configuration import ReferenceCheckpointEvaluationDefinition
from src.evaluation.manager import EvaluationManager
from src.evaluation.process import write_evaluation_result
from src.evaluation.scheduling import ScheduledEvaluationSuite, jobs_for_suite
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.training.checkpoint import CheckpointReference
from src.util.tensorboard import TensorboardWriter


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
        self.pid: int | None = None
        self.started = False
        self.events = events

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started and self.exitcode is None

    def terminate(self) -> None:
        self.events.append(f'terminate:{self.name}')
        self.exitcode = -15

    def kill(self) -> None:
        self.events.append(f'kill:{self.name}')
        self.exitcode = -9

    def join(self, timeout: float | None = None) -> None:
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


def test_manager_writes_compact_evaluation_layout_to_raw_coordinator_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensorboard_root = tmp_path / 'tensorboard'
    monkeypatch.setenv('TRAINING_TENSORBOARD_LOG_PATH', str(tensorboard_root))
    experiment = experiment_configuration(tmp_path / 'run')

    with TensorboardWriter(run=0, suffix='coordinator', postfix_pid=False):
        manager = EvaluationManager(
            experiment,
            checkpoint(tmp_path / 'run', 0),
            FakeClock(),
            FakeProcessContext(),
        )
        manager.close()

    accumulator = EventAccumulator(str(tensorboard_root / 'run_0' / 'coordinator'))
    accumulator.Reload()
    layout_event = accumulator.Tensors('custom_scalars__config__')[0]
    layout = layout_pb2.Layout.FromString(layout_event.tensor_proto.string_val[0])
    match_category = next(category for category in layout.category if category.title == 'Evaluation matches')
    search_random_charts = tuple(chart for chart in match_category.chart if chart.title.startswith('search-random '))
    assert tuple(chart.title for chart in search_random_charts) == (
        'search-random W/D/L',
        'search-random score',
    )
    assert tuple(search_random_charts[0].multiline.tag) == (
        'evaluation/search-random/wins',
        'evaluation/search-random/draws',
        'evaluation/search-random/losses',
    )
    dataset_category = next(category for category in layout.category if category.title == 'Evaluation datasets')
    assert tuple(chart.title for chart in dataset_category.chart) == (
        'Top-action accuracy',
        'Policy cross-entropy',
    )
    metadata_category = next(category for category in layout.category if category.title == 'Evaluation metadata')
    player_score_chart = next(
        chart for chart in metadata_category.chart if chart.title == 'search-random player scores'
    )
    assert tuple(player_score_chart.multiline.tag) == (
        'evaluation_metadata/search-random/first_player_score',
        'evaluation_metadata/search-random/second_player_score',
    )
    duration_chart = next(chart for chart in metadata_category.chart if chart.title == 'Duration (seconds)')
    assert len(duration_chart.multiline.tag) == len(experiment.evaluation.definitions)


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

    scalar_events: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        'src.evaluation.manager.log_scalar',
        lambda name, value, step: scalar_events.append((name, step)),
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
    dataset_process = next(
        process for process in context.processes if EVALUATION_JOB_ADAPTER.validate_json(process.args[1]) == dataset_job
    )
    assert isinstance(dataset_process.args[0], str)
    assert isinstance(dataset_process.args[1], str)
    pickle.dumps(dataset_process.args)
    dataset_process.exitcode = 0
    assert manager.collect_completed_jobs()[0].job == dataset_job
    assert scalar_events == [
        ('evaluation_metadata/progress/model_generation', 20),
        ('evaluation_metadata/progress/optimizer_steps', 20),
        ('evaluation_metadata/fixed-dataset/duration_seconds', 20),
        ('evaluation/fixed-dataset/top_action_accuracy', 20),
        ('evaluation/fixed-dataset/policy_cross_entropy', 20),
    ]

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
    assert {1, 2, 3, 4} <= set(manager.required_checkpoint_generations)


def test_manager_limits_active_evaluation_processes(tmp_path: Path) -> None:
    clock = FakeClock()
    context = FakeProcessContext()
    experiment = experiment_configuration(tmp_path)
    experiment = experiment.model_copy(
        update={
            'evaluation': experiment.evaluation.model_copy(update={'maximum_concurrent_jobs': 2}),
        }
    )
    manager = EvaluationManager(experiment, checkpoint(tmp_path, 0), clock, context)

    clock.now = 21.0
    jobs = manager.schedule_due_jobs(checkpoint(tmp_path, 1))

    assert len(jobs) == 7
    assert len(context.processes) == 2
    assert len(manager._processes) == 2

    for process in context.processes:
        process.exitcode = 1
    manager.collect_completed_jobs()

    assert len(context.processes) == 4
    assert len(manager._processes) == 2


def test_baseline_schedule_keeps_only_three_recent_checkpoint_offsets(tmp_path: Path) -> None:
    experiment = load_experiment_configuration(Path('configs/baselines/vast-go-7x7-2gpu-4h.yaml'))
    experiment = experiment.model_copy(
        update={'evaluation': experiment.evaluation.model_copy(update={'cadence_seconds': 20})}
    )
    suites = tuple(
        ScheduledEvaluationSuite(boundary_seconds=boundary, checkpoint=checkpoint(tmp_path, boundary // 20))
        for boundary in range(20, 121, 20)
    )

    jobs, _ = jobs_for_suite(
        experiment,
        tmp_path,
        tmp_path / 'results',
        suites[-1],
        suites[:-1],
        0,
    )
    previous_jobs = tuple(job for job in jobs if job.definition.kind == 'previous_checkpoint')
    assert tuple(job.definition.boundary_offset for job in previous_jobs) == (1, 2, 3)
    assert all(job.definition.opening_pair_count == 200 for job in previous_jobs)


def test_same_time_reference_checkpoint_is_resolved_from_manifest(tmp_path: Path) -> None:
    experiment = experiment_configuration(tmp_path)
    search = next(
        definition.search
        for definition in experiment.evaluation.definitions
        if definition.kind == 'previous_checkpoint'
    )
    reference_inference_path = tmp_path / 'reference-inference.pt'
    reference_inference_path.write_bytes(b'reference')
    reference = checkpoint(tmp_path, 9).model_copy(
        update={
            'inference_model_path': reference_inference_path,
            'inference_model_sha256': hashlib.sha256(b'reference').hexdigest(),
        }
    )
    manifest_path = tmp_path / 'reference-checkpoints.json'
    manifest_path.write_text(
        EvaluationReferenceManifest(
            checkpoints=(ElapsedCheckpointReference(boundary_seconds=20, checkpoint=reference),)
        ).model_dump_json(),
        encoding='utf-8',
    )
    definition = ReferenceCheckpointEvaluationDefinition(
        kind='reference_checkpoint',
        definition_id='same-time-baseline',
        manifest_path=str(manifest_path),
        search=search,
        maximum_game_plies=200,
    )
    experiment = experiment.model_copy(
        update={
            'evaluation': experiment.evaluation.model_copy(
                update={'definitions': (*experiment.evaluation.definitions, definition)}
            )
        }
    )
    clock = FakeClock()
    manager = EvaluationManager(experiment, checkpoint(tmp_path, 0), clock, FakeProcessContext())

    clock.now = 21.0
    jobs = manager.schedule_due_jobs(checkpoint(tmp_path, 1))

    reference_job = next(job for job in jobs if job.definition.kind == 'reference_checkpoint')
    assert isinstance(reference_job, MatchEvaluationJob)
    assert isinstance(reference_job.opponent, CheckpointOpponent)
    assert reference_job.opponent.checkpoint == reference


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
    restarted_clock = FakeClock()
    restarted_manager = EvaluationManager(
        experiment_configuration(tmp_path),
        candidate,
        restarted_clock,
        restarted_context,
    )

    assert restarted_context.processes == []

    restarted_manager.start()

    assert len(restarted_context.processes) == len(jobs)
    assert all(process.started for process in restarted_context.processes)

    restarted_clock.now = 20.0
    resumed_jobs = restarted_manager.schedule_due_jobs(checkpoint(tmp_path, 2))

    assert {job.boundary_seconds for job in resumed_jobs} == {40}
    assert {job.candidate.generation for job in resumed_jobs} == {1}

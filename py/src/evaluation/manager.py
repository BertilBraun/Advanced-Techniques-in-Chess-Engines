from __future__ import annotations

from collections.abc import Callable
import hashlib
import multiprocessing as mp
from pathlib import Path
import time
from typing import Literal

from pydantic import Field, TypeAdapter

from src.evaluation.configuration import (
    FixedCheckpointEvaluationDefinition,
    FixedDatasetEvaluationDefinition,
    KataGoEvaluationDefinition,
    PreviousCheckpointEvaluationDefinition,
    RandomOpponentEvaluationDefinition,
    StockfishEvaluationDefinition,
)
from src.evaluation.contracts import (
    CheckpointOpponent,
    EvaluationFailurePhase,
    EvaluationJob,
    EvaluationResult,
    FailedEvaluationResult,
    FixedDatasetEvaluationJob,
    KataGoOpponent,
    MatchEvaluationJob,
    RandomOpponent,
    StockfishOpponent,
)
from src.evaluation.process import run_evaluation_job, write_evaluation_result
from src.experiment.configuration import ExperimentConfiguration
from src.training.checkpoint import CheckpointReference
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.log import log
from src.util.tensorboard import log_scalar


class CheckpointPublication(FrozenModel):
    elapsed_seconds: float = Field(ge=0.0)
    checkpoint: CheckpointReference


class ScheduledEvaluationSuite(FrozenModel):
    boundary_seconds: int = Field(gt=0)
    checkpoint: CheckpointReference


class EvaluationManagerState(FrozenModel):
    schema_version: Literal[1] = 1
    accumulated_elapsed_seconds: float = Field(ge=0.0)
    next_boundary_seconds: int = Field(gt=0)
    next_device_index: int = Field(ge=0)
    checkpoint_publications: tuple[CheckpointPublication, ...]
    scheduled_suites: tuple[ScheduledEvaluationSuite, ...]
    pending_jobs: tuple[EvaluationJob, ...]


class EvaluationManager:
    def __init__(
        self,
        experiment: ExperimentConfiguration,
        starting_checkpoint: CheckpointReference,
        clock: Callable[[], float] = time.monotonic,
        process_context: mp.context.BaseContext | None = None,
    ) -> None:
        self.experiment = experiment
        self.configuration = experiment.evaluation
        self.run_path = Path(experiment.training.save_path)
        self.result_directory = self.run_path / 'evaluations'
        self.result_directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.result_directory / 'manager-state.json'
        self.clock = clock
        self.process_context = mp.get_context('spawn') if process_context is None else process_context
        self.session_started_at = clock()
        if self.state_path.exists():
            self._state = EvaluationManagerState.model_validate_json(self.state_path.read_text(encoding='utf-8'))
            self._elapsed_at_session_start = self._state.accumulated_elapsed_seconds
        else:
            self._elapsed_at_session_start = 0.0
            self._state = EvaluationManagerState(
                accumulated_elapsed_seconds=0.0,
                next_boundary_seconds=self.configuration.cadence_seconds,
                next_device_index=0,
                checkpoint_publications=(CheckpointPublication(elapsed_seconds=0.0, checkpoint=starting_checkpoint),),
                scheduled_suites=(),
                pending_jobs=(),
            )
            self._save_state()
        self._processes: dict[str, tuple[mp.Process, float]] = {}
        self._resume_pending_jobs()

    @property
    def elapsed_seconds(self) -> float:
        return self._elapsed_at_session_start + self.clock() - self.session_started_at

    def collect_completed_jobs(self) -> tuple[EvaluationResult, ...]:
        completed: list[EvaluationResult] = []
        for job_id, (process, started_at) in tuple(self._processes.items()):
            elapsed = self.clock() - started_at
            job = self._pending_job(job_id)
            if process.is_alive() and elapsed < job.deadline_seconds:
                continue
            if process.is_alive():
                process.terminate()
                process.join()
                result = self._failure_result(
                    job,
                    EvaluationFailurePhase.DEADLINE,
                    'Evaluation job exceeded its deadline.',
                    process.exitcode,
                    elapsed,
                )
                write_evaluation_result(result, job.result_path)
            else:
                process.join()
                if job.result_path.exists():
                    result = self._read_result(job)
                else:
                    result = self._failure_result(
                        job,
                        EvaluationFailurePhase.MISSING_ARTIFACT,
                        'Evaluation child exited without a result artifact.',
                        process.exitcode,
                        elapsed,
                    )
                    write_evaluation_result(result, job.result_path)
            completed.append(result)
            del self._processes[job_id]
            self._state = self._state.model_copy(
                update={
                    'pending_jobs': tuple(pending for pending in self._state.pending_jobs if pending.job_id != job_id)
                }
            )
            self._report(result)
        if completed:
            self._save_state()
        return tuple(completed)

    def schedule_due_jobs(self, checkpoint: CheckpointReference) -> tuple[EvaluationJob, ...]:
        elapsed_seconds = self.elapsed_seconds
        self._record_checkpoint_publication(checkpoint, elapsed_seconds)
        scheduled: list[EvaluationJob] = []
        next_boundary = self._state.next_boundary_seconds
        while next_boundary <= elapsed_seconds:
            candidate = self._checkpoint_at(next_boundary)
            suite = ScheduledEvaluationSuite(boundary_seconds=next_boundary, checkpoint=candidate)
            jobs = self._jobs_for_suite(suite)
            self._state = self._state.model_copy(
                update={
                    'next_boundary_seconds': next_boundary + self.configuration.cadence_seconds,
                    'scheduled_suites': (*self._state.scheduled_suites, suite),
                    'pending_jobs': (*self._state.pending_jobs, *jobs),
                }
            )
            self._save_state()
            for job in jobs:
                self._launch(job)
            scheduled.extend(jobs)
            next_boundary += self.configuration.cadence_seconds
        return tuple(scheduled)

    def seconds_until_next_boundary(self) -> float:
        return max(0.0, self._state.next_boundary_seconds - self.elapsed_seconds)

    def close(self) -> None:
        deadline = self.clock() + self.configuration.shutdown_grace_seconds
        while self._processes and self.clock() < deadline:
            self.collect_completed_jobs()
            if self._processes:
                time.sleep(0.05)
        for job_id, (process, started_at) in tuple(self._processes.items()):
            job = self._pending_job(job_id)
            if process.is_alive():
                process.terminate()
            process.join()
            result = self._failure_result(
                job,
                EvaluationFailurePhase.CANCELLED,
                'Evaluation job was cancelled during coordinator shutdown.',
                process.exitcode,
                self.clock() - started_at,
            )
            write_evaluation_result(result, job.result_path)
            self._report(result)
            del self._processes[job_id]
        self._state = self._state.model_copy(
            update={
                'accumulated_elapsed_seconds': self.elapsed_seconds,
                'pending_jobs': (),
            }
        )
        self._save_state()

    def _record_checkpoint_publication(
        self,
        checkpoint: CheckpointReference,
        elapsed_seconds: float,
    ) -> None:
        publications = self._state.checkpoint_publications
        if publications[-1].checkpoint == checkpoint:
            return
        if checkpoint.generation <= publications[-1].checkpoint.generation:
            raise ValueError('Evaluation checkpoint publications must advance generations.')
        self._state = self._state.model_copy(
            update={
                'checkpoint_publications': (
                    *publications,
                    CheckpointPublication(elapsed_seconds=elapsed_seconds, checkpoint=checkpoint),
                )
            }
        )
        self._save_state()

    def _checkpoint_at(self, boundary_seconds: int) -> CheckpointReference:
        available = tuple(
            publication
            for publication in self._state.checkpoint_publications
            if publication.elapsed_seconds <= boundary_seconds
        )
        if not available:
            raise RuntimeError('No complete checkpoint was available at the evaluation boundary.')
        return available[-1].checkpoint

    def _jobs_for_suite(self, suite: ScheduledEvaluationSuite) -> tuple[EvaluationJob, ...]:
        jobs: list[EvaluationJob] = []
        device_cycle = self.experiment.training.topology.evaluation.device_cycle
        device_index = self._state.next_device_index
        for definition in self.configuration.definitions:
            opponent = None
            match definition:
                case FixedDatasetEvaluationDefinition():
                    kind = 'fixed_dataset'
                case RandomOpponentEvaluationDefinition():
                    kind = 'match'
                    opponent = RandomOpponent(kind='random')
                case PreviousCheckpointEvaluationDefinition(boundary_offset=offset):
                    opponent_boundary = suite.boundary_seconds - offset * self.configuration.cadence_seconds
                    previous = next(
                        (
                            scheduled.checkpoint
                            for scheduled in self._state.scheduled_suites
                            if scheduled.boundary_seconds == opponent_boundary
                        ),
                        None,
                    )
                    if previous is None:
                        continue
                    kind = 'match'
                    opponent = CheckpointOpponent(kind='checkpoint', checkpoint=previous)
                case FixedCheckpointEvaluationDefinition(generation=generation):
                    kind = 'match'
                    opponent = CheckpointOpponent(
                        kind='checkpoint',
                        checkpoint=CheckpointReference.load(self.run_path, generation),
                    )
                case StockfishEvaluationDefinition(skill_level=skill_level):
                    kind = 'match'
                    opponent = StockfishOpponent(kind='stockfish', skill_level=skill_level)
                case KataGoEvaluationDefinition():
                    kind = 'match'
                    opponent = KataGoOpponent(kind='katago')
            device_id = device_cycle[device_index % len(device_cycle)]
            device_index += 1
            job_id = f'{suite.boundary_seconds:010d}-{definition.definition_id}-g{suite.checkpoint.generation}'
            common = {
                'job_id': job_id,
                'definition': definition,
                'boundary_seconds': suite.boundary_seconds,
                'candidate': suite.checkpoint,
                'device_id': device_id,
                'deadline_seconds': self.configuration.job_timeout_seconds,
                'random_seed': self._job_seed(suite.boundary_seconds, definition.definition_id),
                'result_path': self.result_directory / f'{job_id}.json',
            }
            if kind == 'fixed_dataset':
                jobs.append(FixedDatasetEvaluationJob(kind='fixed_dataset', **common))
            else:
                assert opponent is not None
                jobs.append(MatchEvaluationJob(kind='match', opponent=opponent, **common))
        self._state = self._state.model_copy(update={'next_device_index': device_index})
        return tuple(jobs)

    def _job_seed(self, boundary_seconds: int, definition_id: str) -> int:
        payload = f'{self.experiment.training.random_seed}:{boundary_seconds}:{definition_id}'.encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')

    def _launch(self, job: EvaluationJob) -> bool:
        process = self.process_context.Process(
            target=run_evaluation_job,
            args=(self.experiment, job),
            name=f'evaluation-{job.job_id}',
        )
        try:
            process.start()
        except Exception as error:
            result = self._failure_result(
                job,
                EvaluationFailurePhase.SETUP,
                f'Evaluation child could not start: {error}',
                None,
                0.0,
            )
            write_evaluation_result(result, job.result_path)
            self._state = self._state.model_copy(
                update={
                    'pending_jobs': tuple(
                        pending for pending in self._state.pending_jobs if pending.job_id != job.job_id
                    )
                }
            )
            self._report(result)
            self._save_state()
            return False
        self._processes[job.job_id] = (process, self.clock())
        return True

    def _resume_pending_jobs(self) -> None:
        retained: list[EvaluationJob] = []
        for job in self._state.pending_jobs:
            if job.result_path.exists():
                result = self._read_result(job)
                self._report(result)
                continue
            required_paths = (job.candidate.inference_model_path, job.candidate.manifest_path)
            match job:
                case MatchEvaluationJob(opponent=CheckpointOpponent(checkpoint=checkpoint)):
                    required_paths = (*required_paths, checkpoint.inference_model_path)
                case FixedDatasetEvaluationJob() | MatchEvaluationJob():
                    pass
            if all(path.exists() for path in required_paths):
                if self._launch(job):
                    retained.append(job)
                continue
            result = self._failure_result(
                job,
                EvaluationFailurePhase.MISSING_ARTIFACT,
                'Evaluation restart could not find every referenced checkpoint artifact.',
                None,
                0.0,
            )
            write_evaluation_result(result, job.result_path)
            self._report(result)
        self._state = self._state.model_copy(update={'pending_jobs': tuple(retained)})
        self._save_state()

    def _pending_job(self, job_id: str) -> EvaluationJob:
        return next(job for job in self._state.pending_jobs if job.job_id == job_id)

    def _read_result(self, job: EvaluationJob) -> EvaluationResult:
        try:
            result = TypeAdapter(EvaluationResult).validate_json(job.result_path.read_text(encoding='utf-8'))
            if result.job.job_id != job.job_id:
                raise ValueError('Evaluation result job ID does not match its scheduled job.')
            return result
        except Exception as error:
            result = self._failure_result(
                job,
                EvaluationFailurePhase.VALIDATION,
                f'Evaluation result artifact is invalid: {error}',
                None,
                0.0,
            )
            write_evaluation_result(result, job.result_path)
            return result

    def _failure_result(
        self,
        job: EvaluationJob,
        phase: EvaluationFailurePhase,
        message: str,
        exit_code: int | None,
        duration_seconds: float,
    ) -> FailedEvaluationResult:
        return FailedEvaluationResult(
            kind='failed',
            job=job,
            phase=phase,
            message=message,
            exit_code=exit_code,
            traceback_path=None,
            duration_seconds=max(0.0, duration_seconds),
        )

    def _report(self, result: EvaluationResult) -> None:
        step = result.job.boundary_seconds
        definition_id = result.job.definition.definition_id
        match result.kind:
            case 'fixed_dataset':
                log_scalar(
                    f'evaluation/{definition_id}/top_action_accuracy',
                    result.top_action_accuracy,
                    step,
                )
                log_scalar(
                    f'evaluation/{definition_id}/policy_cross_entropy',
                    result.policy_cross_entropy,
                    step,
                )
                log(
                    f'Evaluation {definition_id} at {step}s: '
                    f'accuracy={result.top_action_accuracy:.3f}, '
                    f'cross-entropy={result.policy_cross_entropy:.3f}'
                )
            case 'match':
                log_scalar(f'evaluation/{definition_id}/score', result.aggregate.score, step)
                log_scalar(f'evaluation/{definition_id}/wins', result.aggregate.wins, step)
                log_scalar(f'evaluation/{definition_id}/draws', result.aggregate.draws, step)
                log_scalar(f'evaluation/{definition_id}/losses', result.aggregate.losses, step)
                log(
                    f'Evaluation {definition_id} at {step}s: '
                    f'{result.aggregate.wins}/{result.aggregate.draws}/{result.aggregate.losses}'
                )
            case 'failed':
                log(f'Evaluation {definition_id} at {step}s failed: {result.message}')

    def _save_state(self) -> None:
        snapshot = self._state.model_copy(update={'accumulated_elapsed_seconds': self.elapsed_seconds})
        write_text_atomically(self.state_path, snapshot.model_dump_json(indent=2) + '\n')

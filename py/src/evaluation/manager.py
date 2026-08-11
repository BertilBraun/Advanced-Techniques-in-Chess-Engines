from __future__ import annotations

from collections.abc import Callable
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
from typing import Literal

from pydantic import Field, TypeAdapter

from src.evaluation.contracts import (
    CheckpointOpponent,
    EvaluationFailurePhase,
    EvaluationJob,
    EvaluationReferenceManifest,
    EvaluationResult,
    FailedEvaluationResult,
    FixedDatasetEvaluationJob,
    FixedDatasetEvaluationResult,
    MatchEvaluationJob,
    MatchEvaluationResult,
)
from src.evaluation.process import run_evaluation_job, write_evaluation_result
from src.evaluation.scheduling import (
    CheckpointPublication,
    ScheduledEvaluationSuite,
    checkpoint_at,
    jobs_for_suite,
    required_checkpoint_generations,
)
from src.evaluation.tensorboard import evaluation_tensorboard_categories
from src.experiment.configuration import ExperimentConfiguration
from src.training.checkpoint import CheckpointReference
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.log import log
from src.util.tensorboard import log_custom_scalar_layout, log_scalar, log_text


class EvaluationManagerState(FrozenModel):
    schema_version: Literal[1] = 1
    accumulated_elapsed_seconds: float = Field(ge=0.0)
    next_boundary_seconds: int = Field(gt=0)
    next_device_index: int = Field(ge=0)
    checkpoint_publications: tuple[CheckpointPublication, ...]
    scheduled_suites: tuple[ScheduledEvaluationSuite, ...]
    pending_jobs: tuple[EvaluationJob, ...]


def _terminate_job_process(process: mp.Process) -> None:
    if os.name == 'posix' and process.pid is not None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            return
        except ProcessLookupError:
            pass
    process.terminate()


def _kill_job_process(process: mp.Process) -> None:
    if os.name == 'posix' and process.pid is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            pass
    process.kill()


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
        log_custom_scalar_layout(evaluation_tensorboard_categories(self.configuration))
        self.run_path = Path(experiment.training.save_path)
        self.result_directory = self.run_path / 'evaluations'
        self.result_directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.result_directory / 'manager-state.json'
        self.reference_manifest_path = self.result_directory / 'reference-checkpoints.json'
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
                _terminate_job_process(process)
                process.join(timeout=2.0)
                if process.is_alive():
                    _kill_job_process(process)
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
            self._launch_available_jobs()
            self._save_state()
        return tuple(completed)

    def schedule_due_jobs(self, checkpoint: CheckpointReference) -> tuple[EvaluationJob, ...]:
        elapsed_seconds = self.elapsed_seconds
        self._record_checkpoint_publication(checkpoint, elapsed_seconds)
        scheduled: list[EvaluationJob] = []
        next_boundary = self._state.next_boundary_seconds
        while next_boundary <= elapsed_seconds:
            candidate = checkpoint_at(self._state.checkpoint_publications, next_boundary)
            suite = ScheduledEvaluationSuite(boundary_seconds=next_boundary, checkpoint=candidate)
            jobs, next_device_index = jobs_for_suite(
                self.experiment,
                self.run_path,
                self.result_directory,
                suite,
                self._state.scheduled_suites,
                self._state.next_device_index,
            )
            self._state = self._state.model_copy(
                update={
                    'next_boundary_seconds': next_boundary + self.configuration.cadence_seconds,
                    'next_device_index': next_device_index,
                    'scheduled_suites': (*self._state.scheduled_suites, suite),
                    'pending_jobs': (*self._state.pending_jobs, *jobs),
                }
            )
            self._save_state()
            self._launch_available_jobs()
            scheduled.extend(jobs)
            next_boundary += self.configuration.cadence_seconds
        return tuple(scheduled)

    def seconds_until_next_boundary(self) -> float:
        return max(0.0, self._state.next_boundary_seconds - self.elapsed_seconds)

    @property
    def required_checkpoint_generations(self) -> tuple[int, ...]:
        return required_checkpoint_generations(
            self.configuration,
            self._state.scheduled_suites,
            self._state.pending_jobs,
        )

    def close(self) -> None:
        deadline = self.clock() + self.configuration.shutdown_grace_seconds
        while self._processes and self.clock() < deadline:
            self.collect_completed_jobs()
            if self._processes:
                time.sleep(0.05)
        remaining = tuple(self._processes.items())
        for _, (process, _) in remaining:
            if process.is_alive():
                _terminate_job_process(process)
        termination_deadline = self.clock() + 2.0
        while any(process.is_alive() for _, (process, _) in remaining) and self.clock() < termination_deadline:
            time.sleep(0.01)
        for _, (process, _) in remaining:
            if process.is_alive():
                _kill_job_process(process)
        for job_id, (process, started_at) in remaining:
            job = self._pending_job(job_id)
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
        running_job_ids = {job_id for job_id, _ in remaining}
        for job in self._state.pending_jobs:
            if job.job_id in running_job_ids:
                continue
            result = self._failure_result(
                job,
                EvaluationFailurePhase.CANCELLED,
                'Queued evaluation job was cancelled during coordinator shutdown.',
                None,
                0.0,
            )
            write_evaluation_result(result, job.result_path)
            self._report(result)
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

    def _launch_available_jobs(self) -> None:
        available_slots = self.configuration.maximum_concurrent_jobs - len(self._processes)
        if available_slots <= 0:
            return
        queued_jobs = tuple(job for job in self._state.pending_jobs if job.job_id not in self._processes)
        for job in queued_jobs[:available_slots]:
            self._launch(job)

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
        self._launch_available_jobs()
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
        generation = result.job.candidate.generation
        optimizer_steps = generation * self.experiment.training.lifecycle.credit.optimizer_steps_per_quantum
        log_scalar('evaluation_metadata/progress/model_generation', generation, step)
        log_scalar('evaluation_metadata/progress/optimizer_steps', optimizer_steps, step)
        log_scalar(f'evaluation_metadata/{definition_id}/duration_seconds', result.duration_seconds, step)
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
                log_scalar(
                    f'evaluation_metadata/{definition_id}/first_player_score',
                    result.aggregate.first_player_score,
                    step,
                )
                log_scalar(
                    f'evaluation_metadata/{definition_id}/second_player_score',
                    result.aggregate.second_player_score,
                    step,
                )
                log(
                    f'Evaluation {definition_id} at {step}s: '
                    f'{result.aggregate.wins}/{result.aggregate.draws}/{result.aggregate.losses}'
                )
            case 'failed':
                log(f'Evaluation {definition_id} at {step}s failed: {result.message}')
        self._write_boundary_summary(step, generation, optimizer_steps)

    def _write_boundary_summary(self, boundary_seconds: int, generation: int, optimizer_steps: int) -> None:
        results: list[EvaluationResult] = []
        for path in sorted(self.result_directory.glob(f'{boundary_seconds:010d}-*.json')):
            results.append(TypeAdapter(EvaluationResult).validate_json(path.read_text(encoding='utf-8')))
        header = (
            f'Generation **{generation}**, optimizer steps **{optimizer_steps}**, '
            f'elapsed boundary **{boundary_seconds}s**.\n\n'
        )
        rows = (
            '| Evaluation | Status | Score / accuracy | W-D-L | First | Second | Duration |\n'
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |\n'
        )
        lines = [header, rows]
        for result in results:
            definition_id = result.job.definition.definition_id
            match result:
                case MatchEvaluationResult():
                    aggregate = result.aggregate
                    lines.append(
                        f'| {definition_id} | complete | {aggregate.score:.3f} | '
                        f'{aggregate.wins}-{aggregate.draws}-{aggregate.losses} | '
                        f'{aggregate.first_player_score:.3f} | {aggregate.second_player_score:.3f} | '
                        f'{result.duration_seconds:.1f}s |\n'
                    )
                case FixedDatasetEvaluationResult():
                    lines.append(
                        f'| {definition_id} | complete | {result.top_action_accuracy:.3f} '
                        f'(CE {result.policy_cross_entropy:.3f}) | — | — | — | {result.duration_seconds:.1f}s |\n'
                    )
                case FailedEvaluationResult():
                    lines.append(
                        f'| {definition_id} | {result.phase.value} | — | — | — | — | {result.duration_seconds:.1f}s |\n'
                    )
        summary = ''.join(lines)
        write_text_atomically(self.result_directory / f'{boundary_seconds:010d}-summary.md', summary)
        log_text('evaluation/summaries', summary, boundary_seconds)

    def _save_state(self) -> None:
        snapshot = self._state.model_copy(update={'accumulated_elapsed_seconds': self.elapsed_seconds})
        write_text_atomically(self.state_path, snapshot.model_dump_json(indent=2) + '\n')
        reference_manifest = EvaluationReferenceManifest(checkpoints=snapshot.scheduled_suites)
        write_text_atomically(
            self.reference_manifest_path,
            reference_manifest.model_dump_json(indent=2) + '\n',
        )

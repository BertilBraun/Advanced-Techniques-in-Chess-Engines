from __future__ import annotations

import copy
import os
import time
import uuid
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.multiprocessing import Process

from src.cluster.EvaluationProcess import run_evaluation_process
from src.experiment.credit_telemetry import CreditEvaluationTelemetryStatus
from src.train.CreditPublication import (
    CreditPublicationManifest,
    file_sha256,
    load_credit_publication_manifest,
    publication_manifest_path,
)
from src.train.TrainingArgs import TrainingArgs
from src.util.log import log, warn


CREDIT_EVALUATION_SCHEMA_VERSION = 1


class EvaluationProcessHandle(Protocol):
    pid: int | None
    exitcode: int | None

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...


class CreditEvaluationStatus(str, Enum):
    SUCCEEDED = 'succeeded'
    RETRY_PENDING = 'retry_pending'
    PERMANENT_FAILURE = 'permanent_failure'
    INTERRUPTED = 'interrupted'
    COALESCED = 'coalesced'


class CreditEvaluationSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    model_version: int = Field(ge=0)
    completed_optimizer_steps: int = Field(ge=0)
    trained_position_presentations: int = Field(ge=0)
    publication_manifest_path: str = Field(min_length=1)
    publication_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    jit_model_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class PendingCreditEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    source: CreditEvaluationSource
    next_attempt: int = Field(ge=1)
    not_before_seconds: float = Field(ge=0)


class ActiveCreditEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    source: CreditEvaluationSource
    attempt: int = Field(ge=1)
    started_at_seconds: float = Field(ge=0)
    process_id: int | None


class CreditEvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    source: CreditEvaluationSource
    attempt: int = Field(ge=1)
    status: CreditEvaluationStatus
    started_at_seconds: float = Field(ge=0)
    completed_at_seconds: float = Field(ge=0)
    process_exit_code: int | None
    failure: str | None


class CreditEvaluationSchedulerState(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schema_version: int = CREDIT_EVALUATION_SCHEMA_VERSION
    pending: PendingCreditEvaluation | None = None
    active: ActiveCreditEvaluation | None = None
    results: tuple[CreditEvaluationResult, ...] = ()

    @model_validator(mode='after')
    def validate_schema(self) -> CreditEvaluationSchedulerState:
        if self.schema_version != CREDIT_EVALUATION_SCHEMA_VERSION:
            raise ValueError(f'Unsupported credit-evaluation schema {self.schema_version}.')
        return self


class CreditEvaluationScheduler:
    """Persistent single-suite scheduler that cannot gate credit training."""

    def __init__(
        self,
        run_id: int,
        args: TrainingArgs,
    ) -> None:
        parameters = args.training.credit_training
        if parameters is None:
            raise ValueError('Credit evaluation scheduling requires credit-driven training.')

        self.run_id = run_id
        self.args = args
        self.timeout_seconds = parameters.evaluation_timeout_seconds
        self.maximum_attempts = parameters.evaluation_maximum_attempts
        self.retry_backoff_seconds = parameters.evaluation_retry_backoff_seconds
        self.evaluation_interval_optimizer_steps = parameters.evaluation_interval_optimizer_steps
        self.state_path = Path(args.save_path) / 'credit-evaluation-state.json'
        self._process: EvaluationProcessHandle | None = None
        self._state = self._load_state()
        self._recover_interrupted_evaluation()

    @property
    def state(self) -> CreditEvaluationSchedulerState:
        return self._state

    @property
    def pinned_model_versions(self) -> frozenset[int]:
        versions: set[int] = set()
        if self._state.pending is not None:
            versions.add(self._state.pending.source.model_version)
        if self._state.active is not None:
            versions.add(self._state.active.source.model_version)
        return frozenset(versions)

    @property
    def completed_unpinned_model_versions(self) -> frozenset[int]:
        completed = {result.source.model_version for result in self._state.results}
        return frozenset(completed - self.pinned_model_versions)

    @property
    def current_source_version(self) -> int | None:
        if self._state.active is not None:
            return self._state.active.source.model_version
        if self._state.pending is not None:
            return self._state.pending.source.model_version
        if self._state.results:
            return self._state.results[-1].source.model_version
        return None

    @property
    def current_status(self) -> CreditEvaluationTelemetryStatus:
        if self._state.active is not None:
            return CreditEvaluationTelemetryStatus.ACTIVE
        if self._state.pending is not None:
            return CreditEvaluationTelemetryStatus.RETRY_PENDING
        if self._state.results:
            match self._state.results[-1].status:
                case CreditEvaluationStatus.SUCCEEDED:
                    return CreditEvaluationTelemetryStatus.SUCCEEDED
                case CreditEvaluationStatus.RETRY_PENDING:
                    return CreditEvaluationTelemetryStatus.RETRY_PENDING
                case CreditEvaluationStatus.PERMANENT_FAILURE:
                    return CreditEvaluationTelemetryStatus.PERMANENT_FAILURE
                case CreditEvaluationStatus.INTERRUPTED:
                    return CreditEvaluationTelemetryStatus.INTERRUPTED
                case CreditEvaluationStatus.COALESCED:
                    return CreditEvaluationTelemetryStatus.COALESCED
        return CreditEvaluationTelemetryStatus.IDLE

    def offer(self, publication: CreditPublicationManifest) -> None:
        if publication.completed_optimizer_steps == 0:
            return
        if publication.completed_optimizer_steps % self.evaluation_interval_optimizer_steps:
            return
        source = self._source(publication)
        active_version = self._state.active.source.model_version if self._state.active is not None else -1
        pending_version = self._state.pending.source.model_version if self._state.pending is not None else -1
        completed_versions = {result.source.model_version for result in self._state.results}
        if source.model_version in completed_versions or source.model_version <= max(active_version, pending_version):
            return
        self._state = self._state.model_copy(
            update={
                'pending': PendingCreditEvaluation(
                    source=source,
                    next_attempt=1,
                    not_before_seconds=0,
                )
            }
        )
        self._persist()

    def poll(self) -> None:
        try:
            self._poll()
        except Exception as error:
            warn(f'Credit evaluation scheduler recovered from {type(error).__name__}: {error}')
            self._recover_scheduler_error(error)

    def close(self) -> None:
        if self._process is None:
            return
        if not self._process.is_alive():
            self.poll()
            if self._process is None:
                return
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=10)
        active = self._state.active
        if active is not None:
            self._record_failure(
                active,
                self._process.exitcode,
                'Evaluation interrupted by training shutdown.',
                CreditEvaluationStatus.INTERRUPTED,
            )
        self._process = None

    def _poll(self) -> None:
        active = self._state.active
        if active is not None:
            if self._process is None:
                self._recover_interrupted_evaluation()
                return
            now = time.time()
            if self._process.is_alive():
                if now - active.started_at_seconds <= self.timeout_seconds:
                    return
                self._process.terminate()
                self._process.join(timeout=10)
                self._record_failure(
                    active,
                    self._process.exitcode,
                    f'Evaluation exceeded {self.timeout_seconds:.0f} second timeout.',
                )
                self._process = None
                return
            self._process.join()
            exit_code = self._process.exitcode
            self._process = None
            if exit_code == 0:
                result = CreditEvaluationResult(
                    source=active.source,
                    attempt=active.attempt,
                    status=CreditEvaluationStatus.SUCCEEDED,
                    started_at_seconds=active.started_at_seconds,
                    completed_at_seconds=now,
                    process_exit_code=exit_code,
                    failure=None,
                )
                self._state = self._state.model_copy(update={'active': None, 'results': (*self._state.results, result)})
                self._persist()
                log(
                    f'Finished evaluation at model {active.source.model_version} '
                    f'in {now - active.started_at_seconds:.2f} seconds.'
                )
                return
            self._record_failure(active, exit_code, f'Evaluation process exited with code {exit_code}.')
            return

        pending = self._state.pending
        if pending is None or time.time() < pending.not_before_seconds:
            return
        self._start(pending)

    def _start(self, pending: PendingCreditEvaluation) -> None:
        self._validate_source(pending.source)
        evaluation_args = credit_evaluation_arguments(
            self.args,
            pending.source.completed_optimizer_steps,
        )
        process = Process(
            target=run_evaluation_process,
            args=(
                self.run_id,
                evaluation_args,
                pending.source.model_version,
                pending.source.trained_position_presentations,
            ),
            name=f'credit-evaluation-model-{pending.source.model_version}-attempt-{pending.next_attempt}',
        )
        started_at = time.time()
        try:
            process.start()
        except Exception as error:
            active = ActiveCreditEvaluation(
                source=pending.source,
                attempt=pending.next_attempt,
                started_at_seconds=started_at,
                process_id=None,
            )
            self._state = self._state.model_copy(update={'pending': None, 'active': active})
            self._record_failure(active, None, f'Failed to start evaluation: {type(error).__name__}: {error}')
            return
        self._process = process
        self._state = self._state.model_copy(
            update={
                'pending': None,
                'active': ActiveCreditEvaluation(
                    source=pending.source,
                    attempt=pending.next_attempt,
                    started_at_seconds=started_at,
                    process_id=process.pid,
                ),
            }
        )
        self._persist()
        log(
            f'Starting evaluation at model {pending.source.model_version} '
            f'(optimizer step {pending.source.completed_optimizer_steps}).'
        )

    def _validate_source(self, source: CreditEvaluationSource) -> None:
        run_path = Path(self.args.save_path)
        manifest_path = run_path / source.publication_manifest_path
        if not manifest_path.is_file() or file_sha256(manifest_path) != source.publication_manifest_sha256:
            raise ValueError('Credit evaluation source references an invalid publication manifest.')
        publication = load_credit_publication_manifest(
            run_path,
            source.model_version,
            verify_artifacts=False,
        )
        if (
            publication.completed_optimizer_steps != source.completed_optimizer_steps
            or publication.trained_position_presentations != source.trained_position_presentations
            or publication.jit_model.sha256 != source.jit_model_sha256
        ):
            raise ValueError('Credit evaluation source provenance differs from its immutable publication.')
        jit_path = run_path / publication.jit_model.path
        if not jit_path.is_file() or file_sha256(jit_path) != source.jit_model_sha256:
            raise ValueError('Credit evaluation source JIT artifact hash does not match its publication.')

    def _record_failure(
        self,
        active: ActiveCreditEvaluation,
        exit_code: int | None,
        failure: str,
        terminal_status: CreditEvaluationStatus | None = None,
    ) -> None:
        existing_pending = self._state.pending
        superseded = (
            existing_pending is not None
            and existing_pending.source.completed_optimizer_steps > active.source.completed_optimizer_steps
        )
        retry = active.attempt < self.maximum_attempts and terminal_status is None and not superseded
        status = (
            CreditEvaluationStatus.RETRY_PENDING
            if retry
            else terminal_status
            or (CreditEvaluationStatus.COALESCED if superseded else CreditEvaluationStatus.PERMANENT_FAILURE)
        )
        result = CreditEvaluationResult(
            source=active.source,
            attempt=active.attempt,
            status=status,
            started_at_seconds=active.started_at_seconds,
            completed_at_seconds=time.time(),
            process_exit_code=exit_code,
            failure=failure,
        )
        retry_pending = (
            PendingCreditEvaluation(
                source=active.source,
                next_attempt=active.attempt + 1,
                not_before_seconds=time.time() + self.retry_backoff_seconds * active.attempt,
            )
            if retry
            else None
        )
        pending = (
            existing_pending
            if existing_pending is not None
            and (
                retry_pending is None
                or existing_pending.source.completed_optimizer_steps > retry_pending.source.completed_optimizer_steps
            )
            else retry_pending
        )
        self._state = self._state.model_copy(
            update={
                'active': None,
                'pending': pending,
                'results': (*self._state.results, result),
            }
        )
        self._persist()
        warn(
            f'Credit evaluation model {active.source.model_version} attempt {active.attempt} '
            f'finished as {status.value}: {failure}'
        )

    def _recover_interrupted_evaluation(self) -> None:
        active = self._state.active
        if active is None:
            return
        self._record_failure(
            active,
            None,
            'Evaluation process was not attached after scheduler restart.',
        )

    def _recover_scheduler_error(self, error: Exception) -> None:
        active = self._state.active
        if active is None and self._state.pending is not None:
            pending = self._state.pending
            active = ActiveCreditEvaluation(
                source=pending.source,
                attempt=pending.next_attempt,
                started_at_seconds=time.time(),
                process_id=None,
            )
            self._state = self._state.model_copy(update={'pending': None, 'active': active})
        if active is not None:
            if self._process is not None:
                try:
                    if self._process.is_alive():
                        self._process.terminate()
                    self._process.join(timeout=10)
                except Exception as cleanup_error:
                    warn(f'Failed to clean up evaluation process after scheduler error: {cleanup_error}')
            self._record_failure(
                active,
                self._process.exitcode if self._process is not None else None,
                f'Scheduler error: {type(error).__name__}: {error}',
            )
            self._process = None

    def _source(self, publication: CreditPublicationManifest) -> CreditEvaluationSource:
        path = publication_manifest_path(Path(self.args.save_path), publication.model_version)
        return CreditEvaluationSource(
            model_version=publication.model_version,
            completed_optimizer_steps=publication.completed_optimizer_steps,
            trained_position_presentations=publication.trained_position_presentations,
            publication_manifest_path=path.relative_to(self.args.save_path).as_posix(),
            publication_manifest_sha256=file_sha256(path),
            jit_model_sha256=publication.jit_model.sha256,
        )

    def _load_state(self) -> CreditEvaluationSchedulerState:
        if not self.state_path.exists():
            return CreditEvaluationSchedulerState()
        return CreditEvaluationSchedulerState.model_validate_json(self.state_path.read_text(encoding='utf-8'))

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.state_path.with_name(f'.{self.state_path.name}.{uuid.uuid4().hex}.tmp')
        with temporary_path.open('x', encoding='utf-8') as output:
            output.write(self._state.model_dump_json(indent=2) + '\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, self.state_path)


def credit_evaluation_arguments(
    args: TrainingArgs,
    completed_optimizer_steps: int,
) -> TrainingArgs:
    parameters = args.training.credit_training
    evaluation = args.evaluation
    if parameters is None or evaluation is None:
        return args
    if completed_optimizer_steps % parameters.evaluation_interval_optimizer_steps:
        raise ValueError('Credit evaluation must use a complete evaluation-checkpoint boundary.')
    versions_per_evaluation = parameters.evaluation_interval_optimizer_steps // parameters.optimizer_steps_per_quantum
    translated = copy.deepcopy(args)
    assert translated.evaluation is not None
    translated.evaluation.previous_model_offsets = tuple(
        offset * versions_per_evaluation for offset in evaluation.previous_model_offsets
    )
    translated.evaluation.historical_model_iterations = tuple(
        checkpoint_ordinal * versions_per_evaluation for checkpoint_ordinal in evaluation.historical_model_iterations
    )
    translated.evaluation.every_n_iterations = versions_per_evaluation
    translated.artifact_retention = replace(
        translated.artifact_retention,
        milestone_inference_interval=versions_per_evaluation,
    )
    return translated

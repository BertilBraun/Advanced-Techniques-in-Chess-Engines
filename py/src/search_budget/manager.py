from __future__ import annotations

import hashlib
import multiprocessing
import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from src.replay.contracts import ReplaySample
from src.replay.label_source import ReplayLabelGameLocator
from src.replay.manager import LabelledReplayWritebackState
from src.search_budget.analysis_log import analysis_log_path, append_analysis_records, read_analysis_records
from src.search_budget.artifacts import (
    LabelShardManifest,
    LabelShardPhase,
    load_persisted_model,
    validate_complete_shard_coverage,
    write_persisted_model,
)
from src.search_budget.calibration import (
    IDENTITY_CORRECTOR_REFERENCE,
    BudgetCalibrationParameters,
    BudgetCalibrationState,
    BudgetDecisionReason,
    BudgetPolicyPublication,
    CorrectorReference,
    initial_calibration_state,
    load_calibration_state_fail_closed,
    publication_for_generation,
    publish_fail_closed,
    save_calibration_state,
    update_calibration,
    working_policy,
)
from src.search_budget.configuration import LabelArtifactRetention, SearchBudgetConfiguration
from src.search_budget.corrector import LoadedCurveCorrector, export_corrector, fit_curve_corrector
from src.search_budget.labeling import (
    DeepSearchShardArtifact,
    DistributionSummary,
    GenerationFinalization,
    LabelGenerationSource,
    LabelPositionSource,
    PredictionShardArtifact,
    ReplaySampleProvider,
    build_generation_source,
    finalize_generation,
    prediction_map,
)
from src.search_budget.policy import (
    BUDGET_CURVE_MULTIPLES,
    BUDGET_CURVE_POINTS,
    CurveCorrection,
    SearchBudgetPolicy,
    identity_correction,
)
from src.search_budget.retention import (
    FailedLabelArtifactCleanupReceipt,
    LabelArtifactCleanupEvidence,
    cleanup_completed_generation_artifacts,
)
from src.search_budget.sampling import LabelPositionIdentity, partition_generation_sample
from src.search_budget.worker import (
    DeepSearchShardTask,
    LabelWorkerRuntimeFactory,
    PredictionShardTask,
    execute_deep_search_shard,
    execute_prediction_shard,
    initialize_label_worker,
)
from src.training.checkpoint import CheckpointReference
from src.util.atomic_file import write_bytes_atomically
from src.util.frozen_model import FrozenModel
from src.util.log import warn


class LabelJobStatus(str, Enum):
    QUEUED = 'queued'
    PREDICTING = 'predicting'
    DEEP_SEARCHING = 'deep_searching'
    FINALIZING = 'finalizing'
    COMPLETED = 'completed'
    SKIPPED = 'skipped'
    FAILED = 'failed'


class LabelGenerationJob(FrozenModel):
    source_generation: int = Field(ge=0)
    source_path: Path | None
    status: LabelJobStatus
    skip_reason: str | None = None
    failure: str | None = None

    @model_validator(mode='after')
    def validate_terminal_details(self) -> LabelGenerationJob:
        if (self.status is LabelJobStatus.SKIPPED) != (self.skip_reason is not None):
            raise ValueError('Only skipped label jobs may carry a skip reason.')
        if (self.status is LabelJobStatus.FAILED) != (self.failure is not None):
            raise ValueError('Only failed label jobs may carry a failure message.')
        if self.status is not LabelJobStatus.SKIPPED and self.source_path is None:
            raise ValueError('Every executable label job requires a persisted source path.')
        return self


class LabelManagerState(FrozenModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    configuration_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    highest_started_production_generation: int = Field(ge=-1)
    jobs: tuple[LabelGenerationJob, ...] = ()

    def model_post_init(self, __context: object) -> None:
        generations = tuple(job.source_generation for job in self.jobs)
        if generations != tuple(sorted(set(generations))):
            raise ValueError('Label manager jobs must use unique increasing source generations.')
        active = sum(
            job.status in {LabelJobStatus.PREDICTING, LabelJobStatus.DEEP_SEARCHING, LabelJobStatus.FINALIZING}
            for job in self.jobs
        )
        if active > 1:
            raise ValueError('Only one logical source-generation label job may be active.')
        if any(job.source_generation > self.highest_started_production_generation for job in self.jobs):
            raise ValueError('Label jobs cannot originate after the highest started production generation.')


class CurvePointReport(FrozenModel):
    curve_index: int = Field(ge=0, lt=BUDGET_CURVE_POINTS)
    multiple: float = Field(gt=0.0)
    grid_visits: int = Field(gt=0)
    sigma: float = Field(gt=0.0)
    mean_target_log_kl: float
    mean_predicted_log_kl: float
    mean_absolute_error: float = Field(ge=0.0)
    selected_count: int = Field(ge=0)


class GenerationLabelReport(FrozenModel):
    schema_version: int = Field(default=4, ge=4, le=4)
    source_generation: int = Field(ge=0)
    model_generation: int = Field(ge=0)
    inference_model_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    population_position_count: int = Field(gt=0)
    selected_position_count: int = Field(gt=0)
    baseline_raw_kl_distribution: DistributionSummary
    predicted_baseline_log_kl_distribution: DistributionSummary
    target_baseline_log_kl_distribution: DistributionSummary
    curve_points: tuple[CurvePointReport, ...]
    replay_samples_written: int = Field(gt=0)
    replay_write_applied: bool
    prediction_shard_seconds: float = Field(ge=0.0)
    deep_search_shard_seconds: float = Field(ge=0.0)
    total_gpu_seconds: float = Field(ge=0.0)
    prediction_retry_count: int = Field(ge=0)
    deep_search_retry_count: int = Field(ge=0)
    completion_generation_lag: int = Field(ge=0)
    queued_generation_count: int = Field(ge=0)
    current_validation_gain: float | None
    ema_validation_gain: float | None
    lagrange_multiplier: float = Field(ge=0.0)
    corrector_applied: bool
    corrector_sha256: str | None
    realized_mean_multiple: float = Field(gt=0.0)
    realized_mean_assigned_visits: float = Field(gt=0.0)
    flat_mean_assigned_visits: float = Field(gt=0.0)
    assigned_new_visits_variance: float = Field(ge=0.0)
    selected_index_counts: tuple[int, ...] = Field(min_length=BUDGET_CURVE_POINTS, max_length=BUDGET_CURVE_POINTS)
    published_apply_learned: bool
    failed_eligibility_conditions: tuple[str, ...]
    application_generation: int = Field(ge=0)
    decision_reason: str


class FailedLabelJobReport(FrozenModel):
    schema_version: int = Field(default=2, ge=2, le=2)
    source_generation: int = Field(ge=0)
    failure: str = Field(min_length=1)
    published_apply_learned: bool
    application_generation: int = Field(ge=0)
    decision_reason: str


class SkippedLabelJobReport(FrozenModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    source_generation: int = Field(ge=0)
    population_position_count: int = Field(ge=0)
    selected_position_count: int = Field(ge=0)
    reason: str = Field(min_length=1)


class PreservedManagerStateEvidence(FrozenModel):
    kind: Literal['preserved'] = 'preserved'
    path: Path
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class UnavailableManagerStateEvidence(FrozenModel):
    kind: Literal['unavailable'] = 'unavailable'


ManagerStateEvidence = PreservedManagerStateEvidence | UnavailableManagerStateEvidence


class ManagerStateRecoveryReport(FrozenModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    decision_reason: BudgetDecisionReason
    failure: str = Field(min_length=1)
    evidence: ManagerStateEvidence = Field(discriminator='kind')


LabelManagerEvent = GenerationLabelReport | FailedLabelJobReport | SkippedLabelJobReport


class ReplayWritebackResult(Protocol):
    @property
    def row_count(self) -> int: ...

    @property
    def applied(self) -> bool: ...


ReplayWriter = Callable[[int, tuple[ReplaySample, ...]], ReplayWritebackResult]
PredictionWorker = Callable[[PredictionShardTask], LabelShardManifest]
DeepSearchWorker = Callable[[DeepSearchShardTask], LabelShardManifest]
ShardTask: TypeAlias = PredictionShardTask | DeepSearchShardTask
MAXIMUM_SHARD_ATTEMPTS = 3


class InvalidLabelComputeError(ValueError):
    pass


@dataclass(frozen=True)
class LabelEnqueueResult:
    accepted: bool
    skipped: bool


class SearchBudgetLabelManager:
    def __init__(
        self,
        run_path: Path,
        configuration_sha256: str,
        device_ids: tuple[int, ...],
        runtime_factory: LabelWorkerRuntimeFactory,
        action_size: int,
        maximum_policy_entries: int,
        sample_provider: ReplaySampleProvider,
        replay_writer: ReplayWriter,
        initial_first_unstarted_production_generation: int,
        configuration: SearchBudgetConfiguration,
        executor: Executor | None = None,
        prediction_worker: PredictionWorker = execute_prediction_shard,
        deep_search_worker: DeepSearchWorker = execute_deep_search_shard,
    ) -> None:
        if action_size <= 0 or maximum_policy_entries <= 0:
            raise ValueError('Label finalization requires positive action and retained-policy sizes.')
        unique_devices = tuple(dict.fromkeys(device_ids))
        if not unique_devices:
            raise ValueError('At least one trainer GPU must be eligible for deep labelling.')
        if any(device_id < 0 for device_id in unique_devices):
            raise ValueError('Label worker GPU identifiers must be nonnegative.')
        if initial_first_unstarted_production_generation < 0:
            raise ValueError('The initial first-unstarted production generation must be nonnegative.')
        self.run_path = run_path
        self.jobs_path = run_path / 'search-budget-labels'
        self.jobs_path.mkdir(parents=True, exist_ok=True)
        self.state_path = self.jobs_path / 'manager-state.json'
        self.calibration_path = self.jobs_path / 'calibration-state.json'
        self.configuration_sha256 = configuration_sha256
        self.configuration = configuration
        self.action_size = action_size
        self.maximum_policy_entries = maximum_policy_entries
        self.sample_provider = sample_provider
        self.replay_writer = replay_writer
        self.initial_first_unstarted_production_generation = initial_first_unstarted_production_generation
        self.prediction_worker = prediction_worker
        self.deep_search_worker = deep_search_worker
        self._manager_state_failure_reason: BudgetDecisionReason | None = None
        self._state = self._load_state()
        self._calibration = self._load_calibration()
        if self._manager_state_failure_reason is not None:
            self._calibration = publish_fail_closed(
                self._calibration,
                self._first_unstarted_production_generation,
                self._manager_state_failure_reason,
            )
            save_calibration_state(self.calibration_path, self._calibration)
        self._condition = threading.Condition()
        self._closing = False
        self._reported_generations: set[int] = set()
        self._cleanup_completed_jobs()
        self._device_claims: multiprocessing.queues.Queue[int] | None = None
        if executor is None:
            context = multiprocessing.get_context('spawn')
            claims = context.Queue()
            for device_id in unique_devices:
                claims.put(device_id)
            self._device_claims = claims
            self._executor: Executor = ProcessPoolExecutor(
                max_workers=len(unique_devices),
                mp_context=context,
                initializer=initialize_label_worker,
                initargs=(claims, runtime_factory),
            )
        else:
            self._executor = executor
        self._thread = threading.Thread(target=self._run, name='search-budget-label-manager', daemon=True)
        self._thread.start()

    @property
    def required_checkpoint_generations(self) -> tuple[int, ...]:
        with self._condition:
            return tuple(
                job.source_generation
                for job in self._state.jobs
                if job.status
                in {
                    LabelJobStatus.QUEUED,
                    LabelJobStatus.PREDICTING,
                    LabelJobStatus.DEEP_SEARCHING,
                    LabelJobStatus.FINALIZING,
                }
            )

    @property
    def accounted_source_generations(self) -> tuple[int, ...]:
        with self._condition:
            return tuple(job.source_generation for job in self._state.jobs)

    def publication_for_generation(self, production_generation: int) -> BudgetPolicyPublication:
        with self._condition:
            return publication_for_generation(self._calibration, production_generation)

    def publication_for_starting_generation(self, production_generation: int) -> BudgetPolicyPublication:
        with self._condition:
            highest_started = self._state.highest_started_production_generation
            if production_generation < highest_started or production_generation > highest_started + 1:
                raise ValueError('Production generations must start monotonically without gaps.')
            if production_generation == highest_started + 1:
                self._replace_state(self._state.jobs, highest_started_production_generation=production_generation)
            return publication_for_generation(self._calibration, production_generation)

    def enqueue_replay_generation(
        self,
        source_generation: int,
        label_source_games: tuple[ReplayLabelGameLocator, ...],
        checkpoint: CheckpointReference,
        baseline_new_visits: int,
        run_seed: int,
    ) -> LabelEnqueueResult:
        population = sum(len(game.observation_plies) for game in label_source_games)
        if population == 0:
            return self._enqueue_skipped_generation(
                source_generation,
                0,
                0,
                'generation population is empty',
            )
        source = build_generation_source(
            source_generation,
            label_source_games,
            checkpoint,
            baseline_new_visits,
            run_seed,
            self.configuration.labeling.sample_fraction,
            self.sample_provider,
        )
        if source is None:
            return self._enqueue_skipped_generation(
                source_generation,
                population,
                0,
                'generation population produces zero positions at the configured sample fraction',
            )
        return self.enqueue(source)

    def enqueue(
        self,
        source: LabelGenerationSource,
    ) -> LabelEnqueueResult:
        with self._condition:
            existing = next(
                (job for job in self._state.jobs if job.source_generation == source.source_generation),
                None,
            )
            if existing is not None:
                return LabelEnqueueResult(accepted=False, skipped=existing.status is LabelJobStatus.SKIPPED)
            if self._state.jobs and source.source_generation < self._state.jobs[-1].source_generation:
                raise ValueError('Source-generation label jobs must be enqueued in source order.')
            source_path = self._job_path(source.source_generation) / 'source.json'
            source_path.parent.mkdir(parents=True, exist_ok=True)
            write_persisted_model(source_path, source)
            lag = self._state.highest_started_production_generation - source.source_generation
            if lag > self.configuration.labeling.maximum_unstarted_generation_lag:
                job = LabelGenerationJob(
                    source_generation=source.source_generation,
                    source_path=source_path,
                    status=LabelJobStatus.SKIPPED,
                    skip_reason=f'unstarted source-generation lag {lag} exceeds the configured maximum',
                )
                skipped = True
                self._write_skipped_report(
                    source.source_generation,
                    source.population_position_count,
                    len(source.selected_positions),
                    job.skip_reason,
                )
            else:
                job = LabelGenerationJob(
                    source_generation=source.source_generation,
                    source_path=source_path,
                    status=LabelJobStatus.QUEUED,
                )
                skipped = False
            self._replace_state((*self._state.jobs, job))
            self._condition.notify_all()
            return LabelEnqueueResult(accepted=not skipped, skipped=skipped)

    def poll(self) -> tuple[LabelManagerEvent, ...]:
        with self._condition:
            terminal = tuple(
                job
                for job in self._state.jobs
                if job.status in {LabelJobStatus.COMPLETED, LabelJobStatus.FAILED, LabelJobStatus.SKIPPED}
                and job.source_generation not in self._reported_generations
            )
            events: list[LabelManagerEvent] = []
            for job in terminal:
                match job.status:
                    case LabelJobStatus.COMPLETED:
                        report_type = GenerationLabelReport
                    case LabelJobStatus.FAILED:
                        report_type = FailedLabelJobReport
                    case LabelJobStatus.SKIPPED:
                        report_type = SkippedLabelJobReport
                    case _:
                        raise AssertionError('Only terminal label jobs may be polled.')
                events.append(load_persisted_model(self._report_path(job.source_generation), report_type))
                self._reported_generations.add(job.source_generation)
            return tuple(events)

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join()
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
        finally:
            if self._device_claims is not None:
                self._device_claims.close()

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._closing:
                    return
                job = self._next_job()
                while job is None and not self._closing:
                    self._condition.wait()
                    job = self._next_job()
                if job is None:
                    return
            try:
                self._execute_job(job)
            except BaseException as error:  # noqa: BLE001
                self._fail_job(job.source_generation, error)

    def _execute_job(self, job: LabelGenerationJob) -> None:
        assert job.source_path is not None
        source = load_persisted_model(job.source_path, LabelGenerationSource)
        shards = partition_generation_sample(tuple(position.identity for position in source.selected_positions))
        positions_by_identity = {position.identity: position for position in source.selected_positions}
        try:
            self._set_status(job.source_generation, LabelJobStatus.PREDICTING)
            prediction_manifests = self._run_prediction_phase(source, shards, positions_by_identity)
            validate_complete_shard_coverage(
                tuple(position.identity for position in source.selected_positions),
                prediction_manifests,
                LabelShardPhase.PREDICTION,
            )
            predictions = tuple(
                load_persisted_model(manifest.artifact_path, PredictionShardArtifact)
                for manifest in prediction_manifests
            )
            by_identity = prediction_map(source, predictions)
            self._set_status(job.source_generation, LabelJobStatus.DEEP_SEARCHING)
            deep_manifests = self._run_deep_phase(source, shards, positions_by_identity)
            validate_complete_shard_coverage(
                tuple(position.identity for position in source.selected_positions),
                deep_manifests,
                LabelShardPhase.DEEP_SEARCH,
            )
            deep_artifacts = tuple(
                load_persisted_model(manifest.artifact_path, DeepSearchShardArtifact) for manifest in deep_manifests
            )
            self._set_status(job.source_generation, LabelJobStatus.FINALIZING)
            with self._condition:
                shadow_policy = working_policy(self._calibration)
            finalized = finalize_generation(
                source,
                by_identity,
                shadow_policy,
                deep_artifacts,
                self.action_size,
                self.maximum_policy_entries,
                self._shadow_correction(shadow_policy),
            )
        except ValueError as error:
            raise InvalidLabelComputeError(str(error)) from error
        writeback = self.replay_writer(source.source_generation, finalized.replay_samples)
        if writeback.row_count != len(finalized.replay_samples):
            raise ValueError('Replay writer did not acknowledge the complete finalized label generation.')
        self._append_analysis_records(source.source_generation, finalized)
        corrector, corrector_applied = self._fit_corrector(source.source_generation, finalized.analysis_records)
        with self._condition:
            first_unstarted = self._first_unstarted_production_generation
            update = update_calibration(
                self._calibration,
                source.source_generation,
                finalized.evidence,
                first_unstarted,
                self._calibration_parameters(),
                corrector,
            )
            self._calibration = update.state
            save_calibration_state(self.calibration_path, self._calibration)
        report = self._generation_report(
            source,
            finalized,
            writeback,
            prediction_manifests,
            deep_manifests,
            corrector_applied,
        )
        write_persisted_model(self._report_path(source.source_generation), report)
        self._set_status(job.source_generation, LabelJobStatus.COMPLETED)
        self._cleanup_completed_jobs()

    def _run_prediction_phase(
        self,
        source: LabelGenerationSource,
        shards: tuple[tuple[LabelPositionIdentity, ...], ...],
        positions_by_identity: dict[LabelPositionIdentity, LabelPositionSource],
    ) -> tuple[LabelShardManifest, ...]:
        tasks = tuple(
            PredictionShardTask(
                source_generation=source.source_generation,
                shard_index=index,
                attempt=1,
                checkpoint=source.checkpoint,
                positions=tuple(positions_by_identity[identity] for identity in identities),
                artifact_path=self._phase_path(
                    source.source_generation, 'prediction', index, 'artifact-attempt-1.json'
                ),
                manifest_path=self._phase_path(source.source_generation, 'prediction', index, 'attempt-1.json'),
            )
            for index, identities in enumerate(shards)
        )
        return self._execute_with_retry(tasks, self.prediction_worker)

    def _run_deep_phase(
        self,
        source: LabelGenerationSource,
        shards: tuple[tuple[LabelPositionIdentity, ...], ...],
        positions_by_identity: dict[LabelPositionIdentity, LabelPositionSource],
    ) -> tuple[LabelShardManifest, ...]:
        tasks = tuple(
            DeepSearchShardTask(
                source_generation=source.source_generation,
                shard_index=index,
                attempt=1,
                checkpoint=source.checkpoint,
                positions=tuple(positions_by_identity[identity] for identity in identities),
                checkpoint_visits=tuple(source.checkpoint_visits for _ in identities),
                deep_visit_limit=source.deep_visit_limit,
                parallel_searches=self.configuration.labeling.parallel_searches,
                artifact_path=self._phase_path(
                    source.source_generation, 'deep-search', index, 'artifact-attempt-1.json'
                ),
                manifest_path=self._phase_path(source.source_generation, 'deep-search', index, 'attempt-1.json'),
            )
            for index, identities in enumerate(shards)
        )
        return self._execute_with_retry(tasks, self.deep_search_worker)

    def _execute_with_retry(
        self,
        tasks: tuple[ShardTask, ...],
        worker: Callable[[ShardTask], LabelShardManifest],
    ) -> tuple[LabelShardManifest, ...]:
        manifests: list[LabelShardManifest | None] = [None] * len(tasks)
        attempts = [1] * len(tasks)
        pending: dict[Future[LabelShardManifest], tuple[int, ShardTask]] = {
            self._executor.submit(worker, task): (index, task) for index, task in enumerate(tasks)
        }
        while pending:
            future = next(iter(pending))
            index, task = pending.pop(future)
            try:
                manifests[index] = future.result()
            except BaseException:
                with self._condition:
                    if self._closing:
                        raise
                if attempts[index] >= MAXIMUM_SHARD_ATTEMPTS:
                    raise
                attempts[index] += 1
                retry = task.model_copy(
                    update={
                        'attempt': attempts[index],
                        'artifact_path': task.artifact_path.with_name(f'artifact-attempt-{attempts[index]}.json'),
                        'manifest_path': task.manifest_path.with_name(f'attempt-{attempts[index]}.json'),
                    }
                )
                pending[self._executor.submit(worker, retry)] = (index, retry)
        assert all(manifest is not None for manifest in manifests)
        return tuple(manifest for manifest in manifests if manifest is not None)

    def _generation_report(
        self,
        source: LabelGenerationSource,
        finalized: GenerationFinalization,
        writeback: ReplayWritebackResult,
        prediction_manifests: tuple[LabelShardManifest, ...],
        deep_manifests: tuple[LabelShardManifest, ...],
        corrector_applied: bool,
    ) -> GenerationLabelReport:
        curve_points = tuple(
            CurvePointReport(
                curve_index=diagnostic.curve_index,
                multiple=BUDGET_CURVE_MULTIPLES[diagnostic.curve_index],
                grid_visits=diagnostic.grid_visits,
                sigma=self._calibration.sigma[diagnostic.curve_index],
                mean_target_log_kl=diagnostic.mean_target_log_kl,
                mean_predicted_log_kl=diagnostic.mean_predicted_log_kl,
                mean_absolute_error=diagnostic.mean_absolute_error,
                selected_count=diagnostic.selected_count,
            )
            for diagnostic in finalized.curve_point_diagnostics
        )
        return GenerationLabelReport(
            source_generation=source.source_generation,
            model_generation=source.checkpoint.generation,
            inference_model_sha256=source.checkpoint.inference_model_sha256,
            population_position_count=source.population_position_count,
            selected_position_count=len(source.selected_positions),
            baseline_raw_kl_distribution=finalized.baseline_raw_kl_distribution,
            predicted_baseline_log_kl_distribution=finalized.predicted_baseline_log_kl_distribution,
            target_baseline_log_kl_distribution=finalized.target_baseline_log_kl_distribution,
            curve_points=curve_points,
            replay_samples_written=writeback.row_count,
            replay_write_applied=writeback.applied,
            prediction_shard_seconds=sum(manifest.duration_seconds for manifest in prediction_manifests),
            deep_search_shard_seconds=sum(manifest.duration_seconds for manifest in deep_manifests),
            total_gpu_seconds=sum(manifest.duration_seconds for manifest in (*prediction_manifests, *deep_manifests)),
            prediction_retry_count=sum(manifest.attempt - 1 for manifest in prediction_manifests),
            deep_search_retry_count=sum(manifest.attempt - 1 for manifest in deep_manifests),
            completion_generation_lag=max(
                0,
                self._state.highest_started_production_generation - source.source_generation,
            ),
            queued_generation_count=sum(job.status is LabelJobStatus.QUEUED for job in self._state.jobs),
            current_validation_gain=self._calibration.current_validation_gain,
            ema_validation_gain=self._calibration.ema_validation_gain,
            lagrange_multiplier=self._calibration.lagrange_multiplier,
            corrector_applied=corrector_applied,
            corrector_sha256=self._calibration.corrector_sha256,
            realized_mean_multiple=finalized.evidence.realized_mean_multiple,
            realized_mean_assigned_visits=finalized.evidence.realized_mean_assigned_visits,
            flat_mean_assigned_visits=finalized.evidence.flat_mean_assigned_visits,
            assigned_new_visits_variance=finalized.assigned_new_visits_variance,
            selected_index_counts=finalized.evidence.selected_index_counts,
            published_apply_learned=self._calibration.published_policy.apply_learned,
            failed_eligibility_conditions=tuple(
                failure.value for failure in self._calibration.failed_eligibility_conditions
            ),
            application_generation=self._calibration.application_generation,
            decision_reason=self._calibration.decision_reason.value,
        )

    def _append_analysis_records(self, source_generation: int, finalized: GenerationFinalization) -> None:
        # Append-only diagnostics; a failure here must never fail the label job or affect training.
        try:
            append_analysis_records(
                analysis_log_path(self.jobs_path, source_generation),
                finalized.analysis_records,
            )
        except (OSError, ValueError) as error:
            warn(
                f'Search-budget analysis log append for source generation {source_generation} was not applied: '
                f'{type(error).__name__}: {error}'
            )

    def _shadow_correction(self, shadow_policy: SearchBudgetPolicy) -> CurveCorrection:
        if shadow_policy.corrector_path is None:
            return identity_correction
        return LoadedCurveCorrector.load(shadow_policy.corrector_path, shadow_policy.corrector_sha256)

    def _previous_corrector(self) -> CorrectorReference:
        with self._condition:
            return CorrectorReference(
                path=self._calibration.corrector_path,
                sha256=self._calibration.corrector_sha256,
            )

    def _fit_corrector(
        self,
        source_generation: int,
        current_records: npt.NDArray[np.void],
    ) -> tuple[CorrectorReference, bool]:
        configuration = self.configuration.calibration.corrector
        if not configuration.enabled:
            return IDENTITY_CORRECTOR_REFERENCE, False
        with self._condition:
            previous_generations = self._calibration.finalized_source_generations
        window: list[npt.NDArray[np.void]] = []
        previous_count = configuration.window_generations - 1
        for generation in previous_generations[-previous_count:] if previous_count > 0 else ():
            path = analysis_log_path(self.jobs_path, generation)
            try:
                window.append(read_analysis_records(path))
            except (OSError, ValueError) as error:
                warn(
                    f'Search-budget corrector skipped the analysis log for source generation {generation}: '
                    f'{type(error).__name__}: {error}'
                )
        window.append(current_records)
        try:
            fit = fit_curve_corrector(np.concatenate(window))
        except (ValueError, RuntimeError) as error:
            warn(
                f'Search-budget corrector fit for source generation {source_generation} failed and keeps '
                f'the previous corrector: {type(error).__name__}: {error}'
            )
            return self._previous_corrector(), False
        if not fit.applied or fit.network is None:
            warn(
                f'Search-budget corrector fit for source generation {source_generation} was rejected and keeps '
                f'the previous corrector: {fit.rejection_reason}'
            )
            return self._previous_corrector(), False
        corrector_path = self.jobs_path / f'corrector-generation-{source_generation:08d}.jit.pt'
        try:
            corrector_sha256 = export_corrector(fit.network, corrector_path)
        except (OSError, ValueError, RuntimeError) as error:
            warn(
                f'Search-budget corrector export for source generation {source_generation} failed and keeps '
                f'the previous corrector: {type(error).__name__}: {error}'
            )
            return self._previous_corrector(), False
        return CorrectorReference(path=corrector_path, sha256=corrector_sha256), True

    def _calibration_parameters(self) -> BudgetCalibrationParameters:
        calibration = self.configuration.calibration
        return BudgetCalibrationParameters(
            warmup_completed_generations=calibration.warmup_completed_source_generations,
            sigma_ema_decay=calibration.sigma_ema_decay,
            validation_gain_ema_decay=calibration.validation_gain_ema_decay,
            lambda_step_ratio=calibration.lambda_step_ratio,
        )

    def _cleanup_completed_jobs(self) -> None:
        if self.configuration.labeling.artifact_retention is LabelArtifactRetention.RETAIN_ALL:
            return
        for job in self._state.jobs:
            if job.status is LabelJobStatus.COMPLETED:
                self._cleanup_completed_job(job.source_generation)

    def _cleanup_completed_job(self, source_generation: int) -> None:
        if self.configuration.labeling.artifact_retention is LabelArtifactRetention.RETAIN_ALL:
            return
        try:
            evidence = self._validate_cleanup_preconditions(source_generation)
            receipt = cleanup_completed_generation_artifacts(
                self._job_path(source_generation),
                source_generation,
                evidence,
            )
        except (OSError, ValueError) as error:
            warn(
                f'Search-budget artifact cleanup for source generation {source_generation} was not applied: '
                f'{type(error).__name__}: {error}'
            )
            return
        if isinstance(receipt, FailedLabelArtifactCleanupReceipt):
            warn(
                f'Search-budget artifact cleanup for source generation {source_generation} was not applied: '
                f'{receipt.failure}'
            )

    def _validate_cleanup_preconditions(self, source_generation: int) -> LabelArtifactCleanupEvidence:
        persisted_state = load_persisted_model(self.state_path, LabelManagerState)
        job = next(
            (candidate for candidate in persisted_state.jobs if candidate.source_generation == source_generation),
            None,
        )
        if job is None or job.status is not LabelJobStatus.COMPLETED:
            raise ValueError('Artifact cleanup requires a durably completed label manager job.')
        report_path = self._report_path(source_generation)
        report = load_persisted_model(report_path, GenerationLabelReport)
        load_persisted_model(self.calibration_path, BudgetCalibrationState)
        writeback_path = self.run_path / 'completed-games' / 'labelled-replay-writebacks.json'
        writebacks = load_persisted_model(writeback_path, LabelledReplayWritebackState)
        writeback = next(
            (receipt for receipt in writebacks.receipts if receipt.source_generation == source_generation),
            None,
        )
        if writeback is None or not writeback.committed:
            raise ValueError('Artifact cleanup requires a committed replay write-back receipt.')
        if writeback.row_count != report.replay_samples_written:
            raise ValueError('Replay write-back receipt does not match the final generation report.')
        return LabelArtifactCleanupEvidence(
            final_report_path=report_path.relative_to(self.run_path),
            manager_state_path=self.state_path.relative_to(self.run_path),
            calibration_state_path=self.calibration_path.relative_to(self.run_path),
            replay_writeback_state_path=writeback_path.relative_to(self.run_path),
        )

    def _fail_job(self, source_generation: int, error: BaseException) -> None:
        reason = _failure_decision_reason(error)
        with self._condition:
            first_unstarted = self._first_unstarted_production_generation
            self._calibration = publish_fail_closed(
                self._calibration,
                first_unstarted,
                reason,
            )
            save_calibration_state(self.calibration_path, self._calibration)
        report = FailedLabelJobReport(
            source_generation=source_generation,
            failure=f'{type(error).__name__}: {error}',
            published_apply_learned=self._calibration.published_policy.apply_learned,
            application_generation=self._calibration.application_generation,
            decision_reason=self._calibration.decision_reason.value,
        )
        write_persisted_model(self._report_path(source_generation), report)
        self._set_status(source_generation, LabelJobStatus.FAILED, failure=report.failure)

    def _next_job(self) -> LabelGenerationJob | None:
        while True:
            job = next((candidate for candidate in self._state.jobs if candidate.status is LabelJobStatus.QUEUED), None)
            if job is None:
                return None
            latest_started = self._state.highest_started_production_generation
            lag = latest_started - job.source_generation
            if lag <= self.configuration.labeling.maximum_unstarted_generation_lag:
                return job
            assert job.source_path is not None
            source = load_persisted_model(job.source_path, LabelGenerationSource)
            reason = f'unstarted source-generation lag {lag} exceeds the configured maximum at job start'
            self._write_skipped_report(
                job.source_generation,
                source.population_position_count,
                len(source.selected_positions),
                reason,
            )
            jobs = tuple(
                candidate.model_copy(update={'status': LabelJobStatus.SKIPPED, 'skip_reason': reason})
                if candidate.source_generation == job.source_generation
                else candidate
                for candidate in self._state.jobs
            )
            self._replace_state(jobs)

    def _enqueue_skipped_generation(
        self,
        source_generation: int,
        population_position_count: int,
        selected_position_count: int,
        reason: str,
    ) -> LabelEnqueueResult:
        with self._condition:
            existing = next(
                (job for job in self._state.jobs if job.source_generation == source_generation),
                None,
            )
            if existing is not None:
                return LabelEnqueueResult(accepted=False, skipped=existing.status is LabelJobStatus.SKIPPED)
            if self._state.jobs and source_generation < self._state.jobs[-1].source_generation:
                raise ValueError('Source-generation label jobs must be enqueued in source order.')
            job = LabelGenerationJob(
                source_generation=source_generation,
                source_path=None,
                status=LabelJobStatus.SKIPPED,
                skip_reason=reason,
            )
            self._write_skipped_report(
                source_generation,
                population_position_count,
                selected_position_count,
                reason,
            )
            self._replace_state((*self._state.jobs, job))
            self._condition.notify_all()
            return LabelEnqueueResult(accepted=False, skipped=True)

    def _write_skipped_report(
        self,
        source_generation: int,
        population_position_count: int,
        selected_position_count: int,
        reason: str,
    ) -> None:
        self._job_path(source_generation).mkdir(parents=True, exist_ok=True)
        report = SkippedLabelJobReport(
            source_generation=source_generation,
            population_position_count=population_position_count,
            selected_position_count=selected_position_count,
            reason=reason,
        )
        write_persisted_model(self._report_path(source_generation), report)

    def _set_status(self, source_generation: int, status: LabelJobStatus, failure: str | None = None) -> None:
        with self._condition:
            jobs = tuple(
                job.model_copy(update={'status': status, 'failure': failure})
                if job.source_generation == source_generation
                else job
                for job in self._state.jobs
            )
            self._replace_state(jobs)
            self._condition.notify_all()

    @property
    def _first_unstarted_production_generation(self) -> int:
        return self._state.highest_started_production_generation + 1

    def _replace_state(
        self,
        jobs: tuple[LabelGenerationJob, ...],
        highest_started_production_generation: int | None = None,
    ) -> None:
        highest_started = (
            self._state.highest_started_production_generation
            if highest_started_production_generation is None
            else highest_started_production_generation
        )
        self._state = LabelManagerState(
            configuration_sha256=self.configuration_sha256,
            highest_started_production_generation=highest_started,
            jobs=jobs,
        )
        write_persisted_model(self.state_path, self._state)

    def _load_state(self) -> LabelManagerState:
        if not self.state_path.exists():
            return self._initial_manager_state()
        try:
            raw_state = self.state_path.read_bytes()
        except OSError as error:
            self._record_manager_state_failure(
                BudgetDecisionReason.UNREADABLE_STATE,
                f'{type(error).__name__}: {error}',
                None,
            )
            return self._initial_manager_state()
        try:
            loaded = LabelManagerState.model_validate_json(raw_state)
        except ValueError as error:
            self._record_manager_state_failure(
                BudgetDecisionReason.UNREADABLE_STATE,
                f'{type(error).__name__}: {error}',
                raw_state,
            )
            return self._initial_manager_state()
        if loaded.configuration_sha256 != self.configuration_sha256:
            self._record_manager_state_failure(
                BudgetDecisionReason.INCOMPATIBLE_STATE,
                'Persisted label manager configuration digest does not match the active run configuration.',
                raw_state,
            )
            return self._initial_manager_state()
        jobs = tuple(
            job.model_copy(update={'status': LabelJobStatus.QUEUED})
            if job.status in {LabelJobStatus.PREDICTING, LabelJobStatus.DEEP_SEARCHING, LabelJobStatus.FINALIZING}
            else job
            for job in loaded.jobs
        )
        return LabelManagerState(
            configuration_sha256=self.configuration_sha256,
            highest_started_production_generation=loaded.highest_started_production_generation,
            jobs=jobs,
        )

    def _initial_manager_state(self) -> LabelManagerState:
        return LabelManagerState(
            configuration_sha256=self.configuration_sha256,
            highest_started_production_generation=self.initial_first_unstarted_production_generation - 1,
        )

    def _record_manager_state_failure(
        self,
        reason: BudgetDecisionReason,
        failure: str,
        raw_state: bytes | None,
    ) -> None:
        if raw_state is None:
            evidence: ManagerStateEvidence = UnavailableManagerStateEvidence()
        else:
            digest = hashlib.sha256(raw_state).hexdigest()
            evidence_path = self.jobs_path / f'manager-state-invalid-{digest}.json'
            write_bytes_atomically(evidence_path, raw_state)
            evidence = PreservedManagerStateEvidence(path=evidence_path, sha256=digest)
        report = ManagerStateRecoveryReport(decision_reason=reason, failure=failure, evidence=evidence)
        write_persisted_model(self.jobs_path / 'manager-state-recovery.json', report)
        self._manager_state_failure_reason = reason

    def _load_calibration(self) -> BudgetCalibrationState:
        if not self.calibration_path.exists():
            return initial_calibration_state(self.configuration_sha256)
        return load_calibration_state_fail_closed(
            self.calibration_path,
            self.configuration_sha256,
            self._first_unstarted_production_generation,
        )

    def _job_path(self, source_generation: int) -> Path:
        return self.jobs_path / f'generation-{source_generation:08d}'

    def _phase_path(self, source_generation: int, phase: str, shard_index: int, name: str) -> Path:
        path = self._job_path(source_generation) / phase / f'shard-{shard_index:05d}' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _report_path(self, source_generation: int) -> Path:
        return self._job_path(source_generation) / 'final-report.json'


def _failure_decision_reason(error: BaseException) -> BudgetDecisionReason:
    if isinstance(error, InvalidLabelComputeError):
        return BudgetDecisionReason.INVALID_COMPUTE
    return BudgetDecisionReason.TERMINAL_FAILURE

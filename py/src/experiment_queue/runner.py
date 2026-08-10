from __future__ import annotations

from collections.abc import Callable
import signal
import sys
import time
from datetime import datetime, timezone
from types import FrameType

from src.experiment_queue.configuration import QueuedExperiment, ResourceSlot, slot_satisfies_request
from src.experiment_queue.process import RunningProcess, launch_process, terminate_process_group
from src.experiment_queue.scheduler import ResourceAssignment, schedule_experiments
from src.experiment_queue.state import (
    CompletedExperimentStatus,
    ExecutionIdentity,
    FailedExperimentStatus,
    ExperimentStatus,
    PendingExperimentStatus,
    QueueSummary,
    RunningExperimentStatus,
    load_queue_summary,
    write_queue_summary,
)
from src.experiment_queue.validation import ValidatedQueue, ValidatedQueuedExperiment


class ExperimentQueueRunner:
    def __init__(self, load_queue: Callable[[], ValidatedQueue]) -> None:
        self._load_queue = load_queue
        self._queue = load_queue()
        self._restart_block_reason: str | None = None
        self._reload_error: str | None = None
        self._summary = self._load_or_create_summary()
        self._running_processes: dict[str, RunningProcess] = {}
        self._stop_requested = False
        self._validate_pending_log_paths()

    @property
    def summary(self) -> QueueSummary:
        return self._summary

    @property
    def active_process_count(self) -> int:
        return len(self._running_processes)

    @property
    def configuration_reload_error(self) -> str | None:
        return self._reload_error

    def run(self) -> QueueSummary:
        if self._restart_block_reason is not None:
            raise ValueError(self._restart_block_reason)
        previous_sigint = signal.signal(signal.SIGINT, self._handle_termination_signal)
        previous_sigterm = signal.signal(signal.SIGTERM, self._handle_termination_signal)
        try:
            while True:
                self.refresh_configuration()
                self._collect_finished_processes()
                if self._stop_requested:
                    self._terminate_running_processes()
                    break
                if self._reload_error is None:
                    self._launch_ready_experiments()
                if not self._has_pending_or_running_experiments():
                    if not self._queue.configuration.wait_for_updates_when_empty:
                        break
                time.sleep(self._queue.configuration.poll_interval_seconds)
            self._collect_finished_processes()
            return self._summary
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
            if self._running_processes:
                self._terminate_running_processes()

    def refresh_configuration(self) -> None:
        try:
            refreshed = self._load_queue()
            self._reconcile_configuration(refreshed)
        except (OSError, ValueError) as error:
            message = f'{type(error).__name__}: {error}'
            if message != self._reload_error:
                print(f'Queue configuration reload rejected: {message}', file=sys.stderr, flush=True)
            self._reload_error = message
            return
        if self._reload_error is not None:
            print('Queue configuration reload recovered.', file=sys.stderr, flush=True)
        self._reload_error = None

    def _load_or_create_summary(self) -> QueueSummary:
        summary_path = self._queue.configuration.summary_path
        if not summary_path.exists():
            timestamp = _now()
            summary = QueueSummary(
                queue_fingerprint=self._queue.fingerprint,
                created_at=timestamp,
                updated_at=timestamp,
                experiments=tuple(
                    PendingExperimentStatus(
                        experiment_id=experiment.definition.experiment_id,
                        queued_at=timestamp,
                        configuration_sha256=experiment.configuration_sha256,
                    )
                    for experiment in self._queue.experiments
                ),
            )
            write_queue_summary(summary_path, summary)
            return summary

        summary = load_queue_summary(summary_path)
        self._validate_existing_summary(summary)
        recovered_summary = self._fail_unrecoverable_running_experiments(summary)
        if recovered_summary != summary:
            write_queue_summary(summary_path, recovered_summary)
            self._restart_block_reason = (
                'Prior running experiments were marked failed but not signalled because their persisted process IDs may be '
                'stale. Verify those process groups have ended, then invoke the queue again to continue pending work.'
            )
        return recovered_summary

    def _validate_existing_summary(self, summary: QueueSummary) -> None:
        if summary.queue_fingerprint != self._queue.fingerprint:
            raise ValueError('Existing queue summary does not match the immutable queue runtime configuration.')
        for status in summary.experiments:
            if isinstance(status, PendingExperimentStatus):
                continue
            self._validate_persisted_assignment(status.execution.assignment)

    def _validate_persisted_assignment(self, assignment: ResourceAssignment) -> None:
        slot = self._slot(assignment.slot_id)
        if (
            assignment.cuda_devices != slot.cuda_devices
            or assignment.cpu_affinity != slot.cpu_affinity[: len(assignment.cpu_affinity)]
            or assignment.ram_limit_bytes > slot.ram_capacity_bytes
            or assignment.working_directory != slot.working_directory
            or assignment.log_directory != slot.log_directory
        ):
            raise ValueError(f'Persisted assignment for slot {assignment.slot_id!r} is invalid.')

    def _reconcile_configuration(self, refreshed: ValidatedQueue) -> None:
        if refreshed.fingerprint != self._queue.fingerprint:
            raise ValueError('Running queue resources, summary, or termination policy cannot change.')
        existing = {status.experiment_id: status for status in self._summary.experiments}
        refreshed_ids = {experiment.definition.experiment_id for experiment in refreshed.experiments}
        statuses: list[ExperimentStatus] = [
            status
            for status in self._summary.experiments
            if not isinstance(status, PendingExperimentStatus) and status.experiment_id not in refreshed_ids
        ]
        timestamp = _now()
        for experiment in refreshed.experiments:
            experiment_id = experiment.definition.experiment_id
            status = existing.get(experiment_id)
            if status is None:
                statuses.append(
                    PendingExperimentStatus(
                        experiment_id=experiment_id,
                        queued_at=timestamp,
                        configuration_sha256=experiment.configuration_sha256,
                    )
                )
                continue
            expected_sha256 = (
                status.configuration_sha256
                if isinstance(status, PendingExperimentStatus)
                else status.execution.configuration_sha256
            )
            if not isinstance(status, PendingExperimentStatus) and expected_sha256 != experiment.configuration_sha256:
                raise ValueError(f'Running or terminal experiment {experiment_id!r} cannot change configuration.')
            if isinstance(status, PendingExperimentStatus):
                statuses.append(
                    PendingExperimentStatus(
                        experiment_id=experiment_id,
                        queued_at=(
                            status.queued_at if expected_sha256 == experiment.configuration_sha256 else timestamp
                        ),
                        configuration_sha256=experiment.configuration_sha256,
                    )
                )
            else:
                statuses.append(status)
        reconciled = QueueSummary(
            queue_fingerprint=self._summary.queue_fingerprint,
            created_at=self._summary.created_at,
            updated_at=timestamp,
            experiments=tuple(statuses),
        )
        self._queue = refreshed
        self._validate_pending_log_paths(reconciled)
        if reconciled.experiments != self._summary.experiments:
            self._summary = reconciled
            write_queue_summary(self._queue.configuration.summary_path, self._summary)

    def _fail_unrecoverable_running_experiments(self, summary: QueueSummary) -> QueueSummary:
        timestamp = _now()
        changed = False
        statuses: list[ExperimentStatus] = []
        for status in summary.experiments:
            if isinstance(status, RunningExperimentStatus):
                changed = True
                statuses.append(
                    FailedExperimentStatus(
                        experiment_id=status.experiment_id,
                        execution=status.execution,
                        finished_at=timestamp,
                        exit_code=None,
                        reason='Queue supervisor restarted; the prior process is not recovered or adopted.',
                    )
                )
            else:
                statuses.append(status)
        if not changed:
            return summary
        return QueueSummary(
            queue_fingerprint=summary.queue_fingerprint,
            created_at=summary.created_at,
            updated_at=timestamp,
            experiments=tuple(statuses),
        )

    def _launch_ready_experiments(self) -> None:
        pending = tuple(
            self._experiment_definition(status.experiment_id)
            for status in self._summary.experiments
            if isinstance(status, PendingExperimentStatus)
        )
        occupied_slot_ids = {process.assignment.slot_id for process in self._running_processes.values()}
        available_slots = tuple(
            slot for slot in self._queue.configuration.slots if slot.slot_id not in occupied_slot_ids
        )
        for scheduled in schedule_experiments(pending, available_slots):
            self._launch_experiment(scheduled.experiment, scheduled.assignment)

    def _validate_pending_log_paths(self, summary: QueueSummary | None = None) -> None:
        checked_summary = self._summary if summary is None else summary
        for status in checked_summary.experiments:
            if not isinstance(status, PendingExperimentStatus):
                continue
            experiment = self._experiment_definition(status.experiment_id)
            index = next(
                index
                for index, configured_experiment in enumerate(self._queue.configuration.experiments)
                if configured_experiment.experiment_id == status.experiment_id
            )
            log_stem = f'{index:04d}-{experiment.experiment_id}'
            for slot in self._queue.configuration.slots:
                if not slot_satisfies_request(slot, experiment.resources):
                    continue
                for suffix in ('stdout.log', 'stderr.log'):
                    log_path = slot.log_directory / f'{log_stem}.{suffix}'
                    if log_path.exists():
                        raise ValueError(f'Pending experiment log path already exists: {log_path}')

    def _launch_experiment(self, experiment: QueuedExperiment, assignment: ResourceAssignment) -> None:
        index = next(
            index
            for index, configured_experiment in enumerate(self._queue.configuration.experiments)
            if configured_experiment.experiment_id == experiment.experiment_id
        )
        log_stem = f'{index:04d}-{experiment.experiment_id}'
        stdout_path = assignment.log_directory / f'{log_stem}.stdout.log'
        stderr_path = assignment.log_directory / f'{log_stem}.stderr.log'
        command = (
            *self._queue.configuration.runner.command,
            self._queue.configuration.runner.experiment_path_argument,
            str(experiment.experiment_file),
        )
        running_process = launch_process(
            experiment_id=experiment.experiment_id,
            command=command,
            assignment=assignment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        self._running_processes[experiment.experiment_id] = running_process
        started_at = _now()
        execution = ExecutionIdentity(
            configuration_sha256=self._validated_experiment(experiment.experiment_id).configuration_sha256,
            command=command,
            assignment=assignment,
            started_at=started_at,
            pid=running_process.process.pid,
            process_group_id=running_process.process.pid,
            stdout_log=stdout_path,
            stderr_log=stderr_path,
        )
        self._replace_status(
            experiment.experiment_id,
            RunningExperimentStatus(experiment_id=experiment.experiment_id, execution=execution),
        )

    def _collect_finished_processes(self) -> None:
        for experiment_id, running_process in tuple(self._running_processes.items()):
            resident_memory_bytes = running_process.resident_memory_bytes
            if resident_memory_bytes > running_process.assignment.ram_limit_bytes:
                exit_code = terminate_process_group(
                    running_process,
                    self._queue.configuration.termination_grace_seconds,
                )
                self._finalize_process(
                    experiment_id,
                    running_process,
                    FailedExperimentStatus(
                        experiment_id=experiment_id,
                        execution=self._running_status(experiment_id).execution,
                        finished_at=_now(),
                        exit_code=exit_code,
                        reason=(
                            f'Process-tree RSS {resident_memory_bytes} exceeded the configured limit '
                            f'{running_process.assignment.ram_limit_bytes}.'
                        ),
                    ),
                )
                continue
            exit_code = running_process.process.poll()
            if exit_code is None or running_process.has_live_processes:
                continue
            running_status = self._running_status(experiment_id)
            if exit_code == 0:
                final_status = CompletedExperimentStatus(
                    experiment_id=experiment_id,
                    execution=running_status.execution,
                    finished_at=_now(),
                )
            else:
                final_status = FailedExperimentStatus(
                    experiment_id=experiment_id,
                    execution=running_status.execution,
                    finished_at=_now(),
                    exit_code=exit_code,
                    reason=f'Runner process exited with code {exit_code}.',
                )
            self._finalize_process(experiment_id, running_process, final_status)

    def _terminate_running_processes(self) -> None:
        for experiment_id, running_process in tuple(self._running_processes.items()):
            exit_code = terminate_process_group(
                running_process,
                self._queue.configuration.termination_grace_seconds,
            )
            running_status = self._running_status(experiment_id)
            self._finalize_process(
                experiment_id,
                running_process,
                FailedExperimentStatus(
                    experiment_id=experiment_id,
                    execution=running_status.execution,
                    finished_at=_now(),
                    exit_code=exit_code,
                    reason='Queue termination requested; the complete process group was terminated.',
                ),
            )

    def _finalize_process(
        self,
        experiment_id: str,
        running_process: RunningProcess,
        final_status: CompletedExperimentStatus | FailedExperimentStatus,
    ) -> None:
        running_process.close_logs()
        del self._running_processes[experiment_id]
        self._replace_status(experiment_id, final_status)

    def _replace_status(self, experiment_id: str, replacement: ExperimentStatus) -> None:
        statuses = tuple(
            replacement if status.experiment_id == experiment_id else status for status in self._summary.experiments
        )
        self._summary = QueueSummary(
            queue_fingerprint=self._summary.queue_fingerprint,
            created_at=self._summary.created_at,
            updated_at=_now(),
            experiments=statuses,
        )
        write_queue_summary(self._queue.configuration.summary_path, self._summary)

    def _running_status(self, experiment_id: str) -> RunningExperimentStatus:
        status = next(status for status in self._summary.experiments if status.experiment_id == experiment_id)
        assert isinstance(status, RunningExperimentStatus)
        return status

    def _experiment_definition(self, experiment_id: str) -> QueuedExperiment:
        return next(
            experiment
            for experiment in self._queue.configuration.experiments
            if experiment.experiment_id == experiment_id
        )

    def _validated_experiment(self, experiment_id: str) -> ValidatedQueuedExperiment:
        return next(
            experiment for experiment in self._queue.experiments if experiment.definition.experiment_id == experiment_id
        )

    def _slot(self, slot_id: str) -> ResourceSlot:
        slot = next((slot for slot in self._queue.configuration.slots if slot.slot_id == slot_id), None)
        if slot is None:
            raise ValueError(f'Persisted assignment references unknown slot {slot_id!r}.')
        return slot

    def _has_pending_or_running_experiments(self) -> bool:
        return any(
            isinstance(status, PendingExperimentStatus | RunningExperimentStatus)
            for status in self._summary.experiments
        )

    def _handle_termination_signal(self, signal_number: int, frame: FrameType | None) -> None:
        del signal_number, frame
        self._stop_requested = True


def _now() -> datetime:
    return datetime.now(timezone.utc)

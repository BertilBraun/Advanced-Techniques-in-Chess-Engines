from __future__ import annotations

import signal
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
from src.experiment_queue.validation import ValidatedQueue


class ExperimentQueueRunner:
    def __init__(self, queue: ValidatedQueue) -> None:
        self._queue = queue
        self._restart_block_reason: str | None = None
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

    def run(self) -> QueueSummary:
        if self._restart_block_reason is not None:
            raise ValueError(self._restart_block_reason)
        previous_sigint = signal.signal(signal.SIGINT, self._handle_termination_signal)
        previous_sigterm = signal.signal(signal.SIGTERM, self._handle_termination_signal)
        try:
            while self._has_pending_or_running_experiments():
                self._collect_finished_processes()
                if self._stop_requested:
                    self._terminate_running_processes()
                    break
                self._launch_ready_experiments()
                if self._running_processes:
                    time.sleep(self._queue.configuration.poll_interval_seconds)
            self._collect_finished_processes()
            return self._summary
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
            if self._running_processes:
                self._terminate_running_processes()

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
            raise ValueError('Existing queue summary does not match the validated queue configuration and experiments.')
        expected_ids = tuple(experiment.definition.experiment_id for experiment in self._queue.experiments)
        actual_ids = tuple(experiment.experiment_id for experiment in summary.experiments)
        if actual_ids != expected_ids:
            raise ValueError('Existing queue summary experiment order does not match the queue configuration.')
        for status in summary.experiments:
            if isinstance(status, PendingExperimentStatus):
                continue
            self._validate_persisted_assignment(status.experiment_id, status.execution.assignment)

    def _validate_persisted_assignment(self, experiment_id: str, assignment: ResourceAssignment) -> None:
        experiment = self._experiment_definition(experiment_id)
        slot = self._slot(assignment.slot_id)
        scheduled = schedule_experiments((experiment,), (slot,))
        if not scheduled or scheduled[0].assignment != assignment:
            raise ValueError(f'Persisted assignment for experiment {experiment_id!r} is invalid.')

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

    def _validate_pending_log_paths(self) -> None:
        for index, status in enumerate(self._summary.experiments):
            if not isinstance(status, PendingExperimentStatus):
                continue
            experiment = self._experiment_definition(status.experiment_id)
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
            exit_code = running_process.process.poll()
            if exit_code is None:
                continue
            running_process.close_logs()
            del self._running_processes[experiment_id]
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
            self._replace_status(experiment_id, final_status)

    def _terminate_running_processes(self) -> None:
        for experiment_id, running_process in tuple(self._running_processes.items()):
            exit_code = terminate_process_group(
                running_process,
                self._queue.configuration.termination_grace_seconds,
            )
            running_process.close_logs()
            del self._running_processes[experiment_id]
            running_status = self._running_status(experiment_id)
            self._replace_status(
                experiment_id,
                FailedExperimentStatus(
                    experiment_id=experiment_id,
                    execution=running_status.execution,
                    finished_at=_now(),
                    exit_code=exit_code,
                    reason='Queue termination requested; the complete process group was terminated.',
                ),
            )

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

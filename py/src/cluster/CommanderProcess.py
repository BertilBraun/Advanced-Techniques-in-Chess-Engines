from __future__ import annotations

from typing import Protocol
import time
from dataclasses import dataclass
import torch
import psutil
from torch.multiprocessing import Process
from pathlib import Path
from time import monotonic

from src.cluster.CreditEvaluationScheduler import CreditEvaluationScheduler
from src.cluster.CudaProcess import start_process_on_cuda_device
from src.cluster.TrainerProcess import QuantumResult, ReplayState, TrainerProcess
from src.experiment.configuration import ChessExperimentConfiguration
from src.games.training_contract import TrainingGameImplementation
from src.train.TrainingArgs import CreditTrainingParams
from src.util.communication import (
    START_CONTINUOUS_SELF_PLAY,
    Communication,
    LATEST_SELF_PLAY_MODEL_VERSION,
    pause_self_play_workers,
    resume_self_play_workers,
    self_play_model_refreshed_message,
)
from src.util.log import log, warn
from src.util.save_paths import (
    CheckpointManifest,
    checkpoint_manifest_path,
    load_checkpoint_manifest,
    load_model_and_optimizer,
    save_model_and_optimizer,
)
from src.cluster.SelfPlayProcess import run_self_play_process
from src.train.CreditTrainingLedger import (
    CreditTrainingLedger,
    CreditTrainingProgress,
    PreparedTrainingQuantum,
)
from src.train.CreditPublication import (
    CreditPublicationManifest,
    CreditPublicationPointer,
    create_credit_publication_manifest,
    write_credit_publication_manifest,
)
from src.experiment.credit_telemetry import (
    append_credit_training_telemetry,
    build_credit_training_telemetry,
    load_last_credit_training_telemetry,
)
from src.util.tensorboard import TensorboardWriter
from src.experiment.resource_telemetry import process_tree_open_file_counts
from src.experiment.cost_accounting import estimated_cost


def credit_training_progress_axis(
    progress: CreditTrainingProgress,
    global_batch_size: int,
) -> int:
    """Return trained position presentations, the primary credit-training axis."""
    if global_batch_size <= 0:
        raise ValueError('Global batch size must be positive.')
    return progress.completed_optimizer_steps * global_batch_size


@dataclass(frozen=True)
class PublicationResult:
    manifest: CreditPublicationManifest
    publication_seconds: float
    acknowledgement_seconds: float


@dataclass
class TrainingLifecycle:
    ledger: CreditTrainingLedger
    trainer: TrainerProcess
    evaluation_scheduler: EvaluationScheduler
    previous_progress: CreditTrainingProgress
    previous_credited_completed_searches: int
    credit_wait_started_at: float
    credit_observation_started_at: float


@dataclass(frozen=True)
class CreditObservation:
    replay_state: ReplayState
    progress: CreditTrainingProgress
    replay_capacity: int
    loader_wait_seconds: float
    observation_seconds: float
    completed_at: float


@dataclass(frozen=True)
class CompletedQuantum:
    result: QuantumResult
    progress: CreditTrainingProgress
    publication: PublicationResult


class EvaluationScheduler(Protocol):
    def offer(self, publication: CreditPublicationManifest) -> None: ...

    def poll(self) -> None: ...

    def close(self) -> None: ...


class NoEvaluationScheduler:
    def offer(self, publication: CreditPublicationManifest) -> None:
        return None

    def poll(self) -> None:
        return None

    def close(self) -> None:
        return None


class CommanderProcess:
    """Coordinate persistent self-play, replay, training, publication, and evaluation."""

    def __init__(
        self,
        run: int,
        game: TrainingGameImplementation,
        started_at: float,
    ) -> None:
        self.run_id = run
        self.game = game
        self.configuration = game.configuration
        self.args = game.training

        self.communication_folder = f'communication/{self.run_id}'
        self.communication = Communication(self.communication_folder)
        self.communication.clear_all()

        self.self_play_processes: list[Process] = []
        self.started_at = started_at
        self.final_stop_reason: str | None = None
        self.latest_completed_model_version = 0

    def run(self) -> None:
        Path(self.args.save_path).mkdir(parents=True, exist_ok=True)
        self._run_training_lifecycle(self.args.lifecycle.credit)

    def _run_training_lifecycle(
        self,
        parameters: CreditTrainingParams,
    ) -> None:
        lifecycle: TrainingLifecycle | None = None
        try:
            with TensorboardWriter(self.run_id, 'credit_training', postfix_pid=False):
                lifecycle = self._initialize_and_recover(parameters)
                while lifecycle.ledger.progress.completed_optimizer_steps < parameters.maximum_optimizer_steps:
                    lifecycle.evaluation_scheduler.poll()
                    stop_reason = self._stop_reason()
                    if stop_reason is not None:
                        self.final_stop_reason = stop_reason
                        warn(f'Stopping training: {stop_reason}')
                        break
                    self._ensure_processes_are_running()
                    observation = self._observe_replay_credits(lifecycle, parameters)
                    if observation is None:
                        time.sleep(1)
                        continue
                    completed = self._run_optimizer_quantum(lifecycle, observation)
                    self._record_quantum_telemetry(lifecycle, observation, completed)
                    self._advance_lifecycle(lifecycle, observation, completed.progress)
        finally:
            try:
                if lifecycle is not None:
                    try:
                        lifecycle.trainer.close()
                    finally:
                        lifecycle.evaluation_scheduler.close()
            finally:
                self._shutdown()

    def _initialize_and_recover(self, parameters: CreditTrainingParams) -> TrainingLifecycle:
        self._ensure_model_exists(0)
        ledger = CreditTrainingLedger(
            Path(self.args.save_path),
            parameters,
            self.args.trainer.global_batch_size,
        )
        self._validate_credit_recovery_checkpoint(ledger.progress.model_version, ledger.run_path)
        self.latest_completed_model_version = ledger.progress.model_version
        self._setup_connections()
        self.communication.send_to_id('START USAGE LOGGER', node_id=0)
        initial_manifest = create_credit_publication_manifest(
            ledger.run_path,
            ledger.progress,
            self.args.trainer.global_batch_size,
        )
        initial_pointer = write_credit_publication_manifest(ledger.run_path, initial_manifest)
        self._publish_credit_manifest(initial_manifest, initial_pointer)
        self.communication.boardcast(START_CONTINUOUS_SELF_PLAY)
        if ledger.prepared_quantum is not None:
            self._finish_prepared_publication(ledger, ledger.prepared_quantum)
        evaluation_scheduler: EvaluationScheduler = (
            CreditEvaluationScheduler(self.run_id, self.args, self.configuration.chess.evaluation)
            if isinstance(self.configuration, ChessExperimentConfiguration)
            else NoEvaluationScheduler()
        )
        evaluation_scheduler.offer(
            create_credit_publication_manifest(
                ledger.run_path,
                ledger.progress,
                self.args.trainer.global_batch_size,
            )
        )
        evaluation_scheduler.poll()
        previous_telemetry = load_last_credit_training_telemetry(ledger.run_path / 'credit-training-telemetry.jsonl')
        started_at = monotonic()
        try:
            trainer = TrainerProcess(self.game, self.run_id, ledger.progress.model_version)
        except BaseException:
            evaluation_scheduler.close()
            raise
        return TrainingLifecycle(
            ledger=ledger,
            trainer=trainer,
            evaluation_scheduler=evaluation_scheduler,
            previous_progress=ledger.progress,
            previous_credited_completed_searches=(
                previous_telemetry.credited_completed_searches if previous_telemetry is not None else 0
            ),
            credit_wait_started_at=started_at,
            credit_observation_started_at=started_at,
        )

    def _observe_replay_credits(
        self,
        lifecycle: TrainingLifecycle,
        parameters: CreditTrainingParams,
    ) -> CreditObservation | None:
        replay_capacity = parameters.replay_capacity_for_model_version(lifecycle.ledger.progress.model_version)
        replay_state = lifecycle.trainer.maintain_replay(replay_capacity)
        progress = lifecycle.ledger.reconcile_credited_samples(replay_state.credited_unique_samples)
        required_credits = parameters.presentation_credits_per_quantum(self.args.trainer.global_batch_size)
        if not progress.can_train(required_credits):
            return None
        completed_at = monotonic()
        return CreditObservation(
            replay_state=replay_state,
            progress=progress,
            replay_capacity=replay_capacity,
            loader_wait_seconds=completed_at - lifecycle.credit_wait_started_at,
            observation_seconds=completed_at - lifecycle.credit_observation_started_at,
            completed_at=completed_at,
        )

    def _run_optimizer_quantum(
        self,
        lifecycle: TrainingLifecycle,
        observation: CreditObservation,
    ) -> CompletedQuantum:
        result = self._train_quantum_with_self_play_cleanup(
            lifecycle.trainer,
            global_step=observation.progress.sampler_global_step,
            model_version=observation.progress.model_version + 1,
        )
        prepared = lifecycle.ledger.prepare_quantum(result.checkpoint_manifest)
        publication = self._publish_prepared_quantum(prepared)
        progress = lifecycle.ledger.commit_prepared_quantum()
        self._schedule_evaluation_and_prune(lifecycle.evaluation_scheduler, publication, progress)
        return CompletedQuantum(result, progress, publication)

    def _schedule_evaluation_and_prune(
        self,
        scheduler: CreditEvaluationScheduler,
        publication: PublicationResult,
        progress: CreditTrainingProgress,
    ) -> None:
        scheduler.offer(publication.manifest)
        scheduler.poll()
        for model_version in scheduler.completed_unpinned_model_versions:
            self._prune_nonretained_credit_checkpoint(model_version)
        self._prune_nonretained_credit_checkpoint(progress.model_version - 1, scheduler.pinned_model_versions)

    def _record_quantum_telemetry(
        self,
        lifecycle: TrainingLifecycle,
        observation: CreditObservation,
        completed: CompletedQuantum,
    ) -> None:
        replay_state = observation.replay_state
        result = completed.result
        scheduler = lifecycle.evaluation_scheduler
        telemetry = build_credit_training_telemetry(
            previous_progress=lifecycle.previous_progress,
            progress=completed.progress,
            previous_credited_completed_searches=lifecycle.previous_credited_completed_searches,
            credited_completed_searches=replay_state.credited_completed_searches,
            live_replay_positions=replay_state.live_unique_samples,
            replay_capacity_unique_positions=observation.replay_capacity,
            replay_evicted_unique_positions=replay_state.evicted_unique_samples,
            replay_memory_bytes=replay_state.replay_memory_bytes,
            optimizer_seconds=result.optimizer_seconds,
            decode_seconds=result.decode_seconds,
            transfer_seconds=result.transfer_seconds,
            loader_wait_seconds=observation.loader_wait_seconds,
            credit_observation_seconds=observation.observation_seconds,
            replay_payload_open_count=result.payload_open_count,
            replay_selected_rows=result.selected_rows,
            replay_rows_read=result.rows_read,
            replay_selected_bytes=result.selected_bytes,
            replay_bytes_read=result.bytes_read,
            replay_oldest_source_model_version=replay_state.oldest_source_model_version,
            replay_newest_source_model_version=replay_state.newest_source_model_version,
            replay_weighted_mean_source_model_version_midpoint=replay_state.weighted_mean_source_model_version_midpoint,
            replay_oldest_position_age_seconds=replay_state.oldest_position_age_seconds,
            replay_weighted_mean_position_age_seconds=replay_state.weighted_mean_position_age_seconds,
            publication_seconds=completed.publication.publication_seconds,
            acknowledgement_seconds=completed.publication.acknowledgement_seconds,
            global_batch_size=self.args.trainer.global_batch_size,
            evaluation_source_model_version=scheduler.current_source_version,
            evaluation_status=scheduler.current_status,
        )
        append_credit_training_telemetry(
            Path(self.args.save_path) / 'credit-training-telemetry.jsonl',
            telemetry,
        )
        log(telemetry.console_summary())
        telemetry.log_to_tensorboard(result.training_stats)

    def _advance_lifecycle(
        self,
        lifecycle: TrainingLifecycle,
        observation: CreditObservation,
        progress: CreditTrainingProgress,
    ) -> None:
        self.latest_completed_model_version = progress.model_version
        lifecycle.previous_progress = progress
        lifecycle.previous_credited_completed_searches = observation.replay_state.credited_completed_searches
        lifecycle.credit_wait_started_at = monotonic()
        lifecycle.credit_observation_started_at = observation.completed_at

    def _finish_prepared_publication(
        self,
        ledger: CreditTrainingLedger,
        prepared: PreparedTrainingQuantum,
    ) -> PublicationResult:
        self._validate_credit_recovery_checkpoint(
            prepared.prepared_progress.model_version,
            ledger.run_path,
        )
        log(
            f'Retrying publication of prepared model version '
            f'{prepared.prepared_progress.model_version} without retraining.'
        )
        publication = self._publish_prepared_quantum(prepared)
        committed = ledger.commit_prepared_quantum()
        self._prune_nonretained_credit_checkpoint(committed.model_version - 1)
        self.latest_completed_model_version = committed.model_version
        return publication

    def _validate_credit_recovery_checkpoint(
        self,
        model_version: int,
        run_path: Path,
    ) -> None:
        manifest = load_checkpoint_manifest(model_version, run_path)
        if manifest.iteration != model_version:
            raise ValueError(f'Credit recovery checkpoint version {manifest.iteration} does not match {model_version}.')

    def _publish_prepared_quantum(
        self,
        prepared: PreparedTrainingQuantum,
    ) -> PublicationResult:
        started_at = monotonic()
        manifest = create_credit_publication_manifest(
            Path(self.args.save_path),
            prepared.prepared_progress,
            self.args.trainer.global_batch_size,
        )
        pointer = write_credit_publication_manifest(Path(self.args.save_path), manifest)
        publication_seconds = monotonic() - started_at
        acknowledgement_seconds = self._publish_credit_manifest(manifest, pointer)
        return PublicationResult(
            manifest=manifest,
            publication_seconds=publication_seconds,
            acknowledgement_seconds=acknowledgement_seconds,
        )

    def _publish_credit_manifest(
        self,
        manifest: CreditPublicationManifest,
        pointer: CreditPublicationPointer,
    ) -> float:
        model_version = manifest.model_version
        acknowledgement = self_play_model_refreshed_message(model_version)
        node_ids = tuple(range(len(self.self_play_processes)))
        for node_id in node_ids:
            self.communication.try_receive_value_from_id(acknowledgement, node_id)
        acknowledgement_started_at = monotonic()
        self.communication.publish_persistent_value(
            LATEST_SELF_PLAY_MODEL_VERSION,
            pointer.model_dump_json(),
        )
        self._wait_for_model_acknowledgements(
            model_version=model_version,
            jit_sha256=manifest.jit_model.sha256,
            node_ids=node_ids,
            timeout_seconds=120,
        )
        return monotonic() - acknowledgement_started_at

    def _wait_for_model_acknowledgements(
        self,
        model_version: int,
        jit_sha256: str,
        node_ids: tuple[int, ...],
        timeout_seconds: float,
    ) -> None:
        acknowledgement = self_play_model_refreshed_message(model_version)
        pending_node_ids = set(node_ids)
        deadline = monotonic() + timeout_seconds
        while pending_node_ids:
            for node_id in tuple(pending_node_ids):
                acknowledged_hash = self.communication.try_receive_value_from_id(
                    acknowledgement,
                    node_id,
                )
                if acknowledged_hash is None:
                    continue
                if acknowledged_hash != jit_sha256:
                    raise ValueError(
                        f'Self-play worker {node_id} acknowledged model version {model_version} '
                        f'with JIT hash {acknowledged_hash}, expected {jit_sha256}.'
                    )
                pending_node_ids.remove(node_id)
            if not pending_node_ids:
                return
            self._ensure_processes_are_running()
            if monotonic() >= deadline:
                raise RuntimeError(
                    f'Self-play workers did not acknowledge model version {model_version}: {sorted(pending_node_ids)}'
                )
            time.sleep(0.05)

    def _prune_nonretained_credit_checkpoint(
        self,
        model_version: int,
        pinned_model_versions: frozenset[int] = frozenset(),
    ) -> None:
        if model_version <= 0:
            return
        if model_version in pinned_model_versions:
            return
        parameters = self.args.lifecycle.credit
        optimizer_step = model_version * parameters.optimizer_steps_per_quantum
        if optimizer_step % parameters.retained_checkpoint_interval_steps == 0:
            return
        root = Path(self.args.save_path)
        manifest_path = checkpoint_manifest_path(model_version, root)
        if not manifest_path.exists():
            return
        checkpoint = CheckpointManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
        transient_evaluation_checkpoint = optimizer_step % self.args.lifecycle.evaluation.interval_optimizer_steps == 0
        file_names = [checkpoint.model_path, checkpoint.optimizer_path]
        if not transient_evaluation_checkpoint:
            file_names.append(checkpoint.jit_model_path)
        for file_name in file_names:
            path = root / file_name
            if path.exists():
                path.unlink()

    def _train_quantum_with_self_play_cleanup(
        self,
        trainer: TrainerProcess,
        global_step: int,
        model_version: int,
    ) -> QuantumResult:
        node_ids = self.args.topology.self_play.node_ids_to_pause_during_training
        primary_error: BaseException | None = None
        try:
            if node_ids:
                pause_self_play_workers(
                    self.communication,
                    node_ids,
                    timeout_seconds=120,
                )
                log('Paused self-play workers before training:', node_ids)
            return trainer.train_quantum(global_step=global_step, model_version=model_version)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if node_ids:
                try:
                    resume_self_play_workers(
                        self.communication,
                        node_ids,
                        timeout_seconds=120,
                    )
                    log('Resumed self-play workers after training:', node_ids)
                except BaseException as resume_error:
                    if primary_error is None:
                        raise
                    warn(f'Failed to resume self-play while handling {type(primary_error).__name__}: {resume_error}')

    def _shutdown(self) -> None:
        log('Training complete. Sending STOP to all processes.')
        self.communication.boardcast('STOP')

        for process in self.self_play_processes:
            process.join(timeout=10)
        for process in self.self_play_processes:
            if process.is_alive():
                warn(f'Force-terminating self-play process {process.pid} after graceful shutdown timeout.')
                process.terminate()
                process.join(timeout=10)

    def _stop_reason(self) -> str | None:
        limits = self.args.limits
        elapsed_seconds = monotonic() - self.started_at
        if elapsed_seconds >= limits.maximum_wall_time_seconds:
            return (
                f'wall time {elapsed_seconds / 3600:.2f} h reached '
                f'{limits.maximum_wall_time_seconds / 3600:.2f} h limit'
            )

        current_estimated_cost = estimated_cost(limits.hourly_price, elapsed_seconds)
        if limits.maximum_cost is not None and current_estimated_cost >= limits.maximum_cost:
            return (
                f'estimated cost {limits.cost_currency.value} {current_estimated_cost:.2f} reached '
                f'{limits.cost_currency.value} {limits.maximum_cost:.2f} limit'
            )

        maximum_process_open_file_count, _ = process_tree_open_file_counts(psutil.Process())
        if maximum_process_open_file_count >= limits.maximum_open_file_count:
            return (
                f'per-process open file count {maximum_process_open_file_count} reached '
                f'{limits.maximum_open_file_count} limit'
            )

        host_ram_percent = psutil.virtual_memory().percent
        if host_ram_percent >= limits.maximum_host_ram_percent:
            return f'host RAM usage {host_ram_percent:.1f}% reached {limits.maximum_host_ram_percent:.1f}% limit'

        free_disk_gib = psutil.disk_usage(self.args.save_path).free / 2**30
        if free_disk_gib <= limits.minimum_free_disk_gib:
            return f'free disk {free_disk_gib:.1f} GiB reached {limits.minimum_free_disk_gib:.1f} GiB minimum'
        return None

    def _setup_connections(self) -> None:
        for node_id, device_id in enumerate(self.args.topology.self_play.device_ids):
            self.self_play_processes.append(self._start_self_play_process(node_id, device_id))

        log(f'Started {len(self.self_play_processes)} SelfPlay processes on {torch.cuda.device_count()} devices.')

    def _start_self_play_process(self, node_id: int, device_id: int) -> Process:
        """Starts a SelfPlay process for the given node_id and returns the process."""
        process = Process(
            target=run_self_play_process,
            args=(self.run_id, self.game, self.communication_folder, 0, node_id),
        )
        start_process_on_cuda_device(process, device_id)
        return process

    def _ensure_processes_are_running(self) -> None:
        for i, process in enumerate(list(self.self_play_processes)):
            # 15 minutes since we check in after every move was played, so not very long timeouts required
            if self._ensure_process_is_running(process, f'SELF PLAY {i}', timeout=15 * 60):
                # if the process is not alive, restart it
                device_id = self.args.topology.self_play.device_ids[i]
                self.self_play_processes[i] = self._start_self_play_process(i, device_id)

    def _ensure_process_is_running(self, process: Process, name: str, timeout: int) -> bool:
        """Ensures that the given process is running and alive. If not, it returns true, to indicate that the process should be restarted."""
        alive = process.is_alive()
        heartbeat = self.communication.is_alive(name, timeout=timeout)
        if not alive or not heartbeat:
            warn(f'{name} process {process.pid} is alive ({alive}) and heartbeat ({heartbeat}). Restarting...')
            process.terminate()  # terminate the process
            process.join(timeout=10)  # wait for the process to finish
            return True
        return False

    def _ensure_model_exists(self, starting_model_version: int) -> None:
        if checkpoint_manifest_path(starting_model_version, self.args.save_path).exists():
            load_model_and_optimizer(
                starting_model_version,
                self.args.network,
                torch.device(
                    self.args.topology.trainer.device_type,
                    self.args.topology.trainer.rank_zero_device_id,
                ),
                self.args.save_path,
                self.args.trainer.optimizer,
                self.game.network_dimensions,
            )
            return
        model, optimizer = load_model_and_optimizer(
            starting_model_version,
            self.args.network,
            torch.device(
                self.args.topology.trainer.device_type,
                self.args.topology.trainer.rank_zero_device_id,
            ),
            self.args.save_path,
            self.args.trainer.optimizer,
            self.game.network_dimensions,
        )
        save_model_and_optimizer(model, optimizer, starting_model_version, self.args.save_path)

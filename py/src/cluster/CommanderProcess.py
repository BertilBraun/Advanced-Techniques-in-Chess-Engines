from typing import Generator
import time
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
import torch
import psutil
from torch.multiprocessing import Process
from pathlib import Path
from time import monotonic

from src.cluster.GatingProcess import GatingProcess
from src.cluster.CreditTrainerProcess import CreditTrainerProcess
from src.cluster.CreditEvaluationScheduler import CreditEvaluationScheduler
from src.cluster.CudaProcess import start_process_on_cuda_device
from src.train.TrainingArgs import TrainingArgs
from src.train.TrainingStats import TrainingStats
from src.util.communication import (
    START_CONTINUOUS_SELF_PLAY,
    Communication,
    LATEST_SELF_PLAY_MODEL_VERSION,
    pause_self_play_workers,
    resume_self_play_workers,
    self_play_model_refreshed_message,
)
from src.util.exceptions import log_exceptions
from src.util.log import log, warn
from src.util.save_paths import (
    checkpoint_manifest_path,
    get_latest_model_iteration,
    load_checkpoint_manifest,
    load_model_and_optimizer,
    save_model_and_optimizer,
)
from src.cluster.EvaluationProcess import run_evaluation_process
from src.cluster.SelfPlayProcess import run_self_play_process
from src.cluster.TrainerProcess import TrainerProcess, number_of_games_in_iteration
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
)
from src.util.tensorboard import TensorboardWriter
from src.util.timing import reset_times
from src.experiment.resource_telemetry import process_tree_open_file_counts
from src.experiment.progress_telemetry import IterationTelemetry, append_iteration_telemetry
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
class CreditPublicationResult:
    manifest: CreditPublicationManifest
    publication_seconds: float
    acknowledgement_seconds: float


class CommanderProcess:
    """The CommanderProcess is the main process that manages the communication between the Trainer, SelfPlay and InferenceServer processes.

    It starts the Trainer and SelfPlay processes and sends them the current iteration number.
    Once the Trainer is finished, it receives the training stats from the Trainer.
    It then starts the EvaluationProcess and sends the new iteration number to the Trainer and SelfPlay processes.

    Once all iterations are done, it sends a STOP message to all processes and waits for them to finish.
    """

    def __init__(self, run: int, args: TrainingArgs, started_at: float) -> None:
        self.run_id = run
        self.args = args

        self.communication_folder = f'communication/{self.run_id}'
        self.communication = Communication(self.communication_folder)
        self.communication.clear_all()

        self.self_play_processes: list[Process] = []
        self.evaluation_processes: list[Process] = []

        self.trainer_device_id = self.args.cluster.trainer_rank_zero_device_id
        self.started_at = started_at
        self.final_stop_reason: str | None = None
        self.latest_completed_iteration = 0

    def run(self) -> Generator[tuple[int, TrainingStats], None, None]:
        """The main loop of the CommanderProcess. The resulting generator yields after each iteration. If the Generator is not consumed, no further iterations will be trained."""

        Path(self.args.save_path).mkdir(parents=True, exist_ok=True)
        if self.args.training.credit_training is not None:
            yield from self._run_credit_training()
            return

        starting_iteration = get_latest_model_iteration(self.args.save_path)
        self.latest_completed_iteration = starting_iteration
        log(f'Starting training at iteration {starting_iteration}.')

        self._ensure_model_exists(starting_iteration)
        evaluation = self.args.evaluation
        if starting_iteration == 0 and evaluation is not None and evaluation.evaluate_initial_checkpoint:
            process = self._start_evaluation_process(starting_iteration)
            if process is None:
                return
            if not self._wait_for_all_evaluations():
                return

        current_best_iteration = starting_iteration

        trainer, gating = self._initialize_workers(
            starting_iteration,
            current_best_iteration,
        )

        try:
            with log_exceptions('Commander process'):
                yield from self._run_iterations(
                    trainer,
                    gating,
                    starting_iteration,
                    current_best_iteration,
                )
        finally:
            try:
                trainer.close()
            finally:
                self._shutdown()

    def _run_credit_training(self) -> Generator[tuple[int, TrainingStats], None, None]:
        parameters = self.args.training.credit_training
        assert parameters is not None
        self._ensure_model_exists(0)
        ledger = CreditTrainingLedger(
            Path(self.args.save_path),
            parameters,
            self.args.training.global_batch_size,
        )
        self._validate_credit_recovery_checkpoint(
            ledger.progress.model_version,
            Path(self.args.save_path),
        )
        self.latest_completed_iteration = ledger.progress.model_version
        trainer: CreditTrainerProcess | None = None
        evaluation_scheduler: CreditEvaluationScheduler | None = None
        try:
            with TensorboardWriter(self.run_id, 'credit_training', postfix_pid=False):
                self._setup_connections()
                self.communication.send_to_id('START USAGE LOGGER', node_id=0)
                initial_manifest = create_credit_publication_manifest(
                    Path(self.args.save_path),
                    ledger.progress,
                    self.args.training.global_batch_size,
                )
                initial_pointer = write_credit_publication_manifest(
                    Path(self.args.save_path),
                    initial_manifest,
                )
                self._publish_credit_manifest(initial_manifest, initial_pointer)
                self.communication.boardcast(START_CONTINUOUS_SELF_PLAY)
                prepared = ledger.prepared_quantum
                if prepared is not None:
                    self._finish_prepared_publication(ledger, prepared)

                evaluation_scheduler = CreditEvaluationScheduler(self.run_id, self.args)
                current_publication = create_credit_publication_manifest(
                    Path(self.args.save_path),
                    ledger.progress,
                    self.args.training.global_batch_size,
                )
                evaluation_scheduler.offer(current_publication)
                evaluation_scheduler.poll()
                trainer = CreditTrainerProcess(
                    self.args,
                    self.run_id,
                    ledger.progress.model_version,
                )
                credit_wait_started_at = monotonic()
                credit_observation_started_at = credit_wait_started_at
                previous_quantum_progress = ledger.progress
                while ledger.progress.completed_optimizer_steps < parameters.maximum_optimizer_steps:
                    evaluation_scheduler.poll()
                    stop_reason = self._stop_reason()
                    if stop_reason is not None:
                        self.final_stop_reason = stop_reason
                        warn(f'Stopping credit-driven training: {stop_reason}')
                        break
                    self._ensure_processes_are_running()

                    replay_state = trainer.maintain_replay(compact_below_credited_unique_samples=None)
                    progress = ledger.reconcile_credited_samples(replay_state.credited_unique_samples)
                    required_credits = parameters.presentation_credits_per_quantum(self.args.training.global_batch_size)
                    if not progress.can_train(required_credits):
                        minimum_credited_unique_samples = int(
                            (
                                (progress.consumed_position_credits + Decimal(required_credits))
                                / parameters.replay_ratio
                            ).to_integral_value(rounding=ROUND_CEILING)
                        )
                        replay_state = trainer.maintain_replay(
                            compact_below_credited_unique_samples=minimum_credited_unique_samples
                        )
                        time.sleep(1)
                        continue

                    credit_observation_completed_at = monotonic()
                    loader_wait_seconds = credit_observation_completed_at - credit_wait_started_at
                    credit_observation_seconds = credit_observation_completed_at - credit_observation_started_at
                    result = trainer.train_quantum(
                        global_step=progress.sampler_global_step,
                        model_version=progress.model_version + 1,
                    )
                    prepared = ledger.prepare_quantum(result.checkpoint_manifest)
                    publication = self._publish_prepared_quantum(prepared)
                    committed = ledger.commit_prepared_quantum()
                    evaluation_scheduler.offer(publication.manifest)
                    evaluation_scheduler.poll()
                    self._prune_nonretained_credit_checkpoint(
                        committed.model_version - 1,
                        evaluation_scheduler.pinned_model_versions,
                    )
                    telemetry = build_credit_training_telemetry(
                        previous_progress=previous_quantum_progress,
                        progress=committed,
                        live_replay_positions=replay_state.live_unique_samples,
                        optimizer_seconds=result.optimizer_seconds,
                        decode_seconds=result.decode_seconds,
                        loader_wait_seconds=loader_wait_seconds,
                        credit_observation_seconds=credit_observation_seconds,
                        replay_payload_open_count=result.payload_open_count,
                        replay_selected_rows=result.selected_rows,
                        replay_rows_read=result.rows_read,
                        replay_selected_bytes=result.selected_bytes,
                        replay_bytes_read=result.bytes_read,
                        replay_oldest_source_model_version=replay_state.oldest_source_model_version,
                        replay_newest_source_model_version=replay_state.newest_source_model_version,
                        replay_weighted_mean_source_model_version_midpoint=(
                            replay_state.weighted_mean_source_model_version_midpoint
                        ),
                        replay_oldest_position_age_seconds=replay_state.oldest_position_age_seconds,
                        replay_weighted_mean_position_age_seconds=(replay_state.weighted_mean_position_age_seconds),
                        publication_seconds=publication.publication_seconds,
                        acknowledgement_seconds=publication.acknowledgement_seconds,
                        global_batch_size=self.args.training.global_batch_size,
                        evaluation_source_model_version=evaluation_scheduler.current_source_version,
                        evaluation_status=evaluation_scheduler.current_status,
                    )
                    append_credit_training_telemetry(
                        Path(self.args.save_path) / 'credit-training-telemetry.jsonl',
                        telemetry,
                    )
                    log(telemetry.console_summary())
                    telemetry.log_to_tensorboard(result.training_stats)
                    self.latest_completed_iteration = committed.model_version
                    previous_quantum_progress = committed
                    credit_wait_started_at = monotonic()
                    credit_observation_started_at = credit_observation_completed_at
                    yield (
                        credit_training_progress_axis(
                            committed,
                            self.args.training.global_batch_size,
                        ),
                        result.training_stats,
                    )
        finally:
            if trainer is not None:
                trainer.close()
            if evaluation_scheduler is not None:
                evaluation_scheduler.close()
            self._shutdown()

    def _finish_prepared_publication(
        self,
        ledger: CreditTrainingLedger,
        prepared: PreparedTrainingQuantum,
    ) -> CreditPublicationResult:
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
        self.latest_completed_iteration = committed.model_version
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
    ) -> CreditPublicationResult:
        started_at = monotonic()
        manifest = create_credit_publication_manifest(
            Path(self.args.save_path),
            prepared.prepared_progress,
            self.args.training.global_batch_size,
        )
        pointer = write_credit_publication_manifest(Path(self.args.save_path), manifest)
        publication_seconds = monotonic() - started_at
        acknowledgement_seconds = self._publish_credit_manifest(manifest, pointer)
        return CreditPublicationResult(
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
        parameters = self.args.training.credit_training
        assert parameters is not None
        optimizer_step = model_version * parameters.optimizer_steps_per_quantum
        if optimizer_step % parameters.retained_checkpoint_interval_steps == 0:
            return
        checkpoint = load_checkpoint_manifest(model_version, self.args.save_path)
        root = Path(self.args.save_path)
        for file_name in (
            checkpoint.model_path,
            checkpoint.optimizer_path,
            checkpoint.jit_model_path,
        ):
            path = root / file_name
            if path.exists():
                path.unlink()

    def _initialize_workers(
        self,
        starting_iteration: int,
        current_best_iteration: int,
    ) -> tuple[TrainerProcess, GatingProcess]:
        try:
            log('Setting up connections...')
            self._setup_connections()
            trainer = TrainerProcess(self.args, self.run_id, starting_iteration)
            gating = GatingProcess(self.args, self.run_id, self.trainer_device_id)
            log('Connections set up.')

            self.communication.send_to_id('START USAGE LOGGER', node_id=0)
            self.communication.boardcast(f'LOAD MODEL: {current_best_iteration}')
            self.communication.boardcast(f'START AT ITERATION: {starting_iteration}')
            return trainer, gating
        except Exception:
            self._shutdown()
            raise

    def _run_iterations(
        self,
        trainer: TrainerProcess,
        gating: GatingProcess,
        starting_iteration: int,
        current_best_iteration: int,
    ) -> Generator[tuple[int, TrainingStats], None, None]:
        for iteration in range(starting_iteration, self.args.num_iterations):
            stop_reason = self._stop_reason()
            if stop_reason is not None:
                self.final_stop_reason = stop_reason
                warn(f'Stopping before iteration {iteration}: {stop_reason}')
                break

            self._ensure_processes_are_running()
            log(f'Starting iteration {iteration}.')
            self.communication.boardcast(f'START AT ITERATION: {iteration}')

            games_at_wait_start = number_of_games_in_iteration(
                iteration,
                self.args.save_path,
            )
            wait_started_at = monotonic()
            if not trainer.wait_for_enough_training_samples(
                iteration,
                self._stop_reason,
            ):
                stop_reason = self._stop_reason()
                self.final_stop_reason = stop_reason
                warn(f'Stopping while waiting for iteration {iteration}: {stop_reason}')
                break
            wait_finished_at = monotonic()
            games_at_wait_end = number_of_games_in_iteration(
                iteration,
                self.args.save_path,
            )
            trainer.load_all_memories_to_train_on_for_iteration(iteration)

            self._reap_evaluation_processes()
            training_started_at = monotonic()
            training_result = self._train_with_self_play_cleanup(trainer, iteration)
            training_finished_at = monotonic()
            self.latest_completed_iteration = iteration + 1
            self._write_iteration_telemetry(
                iteration=iteration,
                games_at_wait_start=games_at_wait_start,
                games_at_wait_end=games_at_wait_end,
                wait_seconds=wait_finished_at - wait_started_at,
                replay_samples_loaded=trainer.replay_stats.num_samples,
                replay_games_loaded=trainer.replay_stats.num_games,
                training_seconds=training_finished_at - training_started_at,
            )
            yield iteration, training_result
            log(f'Training finished for iteration {iteration}')
            reset_times()

            current_best_iteration = gating.run(iteration, current_best_iteration)

            if self._should_evaluate(iteration + 1):
                process = self._start_evaluation_process(iteration + 1)
                if process is None:
                    break
                log(f'Started evaluation process for iteration {iteration + 1}.')

    def _train_with_self_play_cleanup(
        self,
        trainer: TrainerProcess,
        iteration: int,
    ) -> TrainingStats:
        node_ids = self.args.cluster.self_play_node_ids_to_pause_during_training
        primary_error: BaseException | None = None
        try:
            if node_ids:
                pause_self_play_workers(
                    self.communication,
                    node_ids,
                    timeout_seconds=120,
                )
                log('Paused self-play workers before training:', node_ids)
            return trainer.train(iteration)
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

        self._wait_for_all_evaluations()

    def _stop_reason(self) -> str | None:
        limits = self.args.run_limits
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

    def _write_iteration_telemetry(
        self,
        iteration: int,
        games_at_wait_start: int,
        games_at_wait_end: int,
        wait_seconds: float,
        replay_samples_loaded: int,
        replay_games_loaded: int,
        training_seconds: float,
    ) -> None:
        elapsed_seconds = monotonic() - self.started_at
        games_generated = games_at_wait_end - games_at_wait_start
        maximum_process_open_file_count, total_open_file_count = process_tree_open_file_counts(psutil.Process())
        telemetry = IterationTelemetry(
            iteration=iteration,
            games_at_wait_start=games_at_wait_start,
            games_at_wait_end=games_at_wait_end,
            games_generated_while_waiting=games_generated,
            wait_seconds=wait_seconds,
            games_per_wait_second=games_generated / wait_seconds if wait_seconds > 0 else 0,
            replay_samples_loaded=replay_samples_loaded,
            replay_games_loaded=replay_games_loaded,
            training_seconds=training_seconds,
            elapsed_seconds=elapsed_seconds,
            cost_currency=self.args.run_limits.cost_currency,
            estimated_cost=estimated_cost(self.args.run_limits.hourly_price, elapsed_seconds),
            maximum_process_open_file_count=maximum_process_open_file_count,
            total_open_file_count=total_open_file_count,
        )
        append_iteration_telemetry(
            Path(self.args.save_path) / 'iteration-telemetry.jsonl',
            telemetry,
        )

    def _should_evaluate(self, iteration: int) -> bool:
        evaluation = self.args.evaluation
        return evaluation is not None and iteration % evaluation.every_n_iterations == 0

    def _start_evaluation_process(self, iteration: int) -> Process | None:
        """Starts an EvaluationProcess for the given iteration and returns the process."""
        self._reap_evaluation_processes()
        while len(self.evaluation_processes) >= self.args.cluster.max_concurrent_evaluations:
            if not self._wait_for_all_evaluations():
                return None

        process = Process(
            target=run_evaluation_process,
            args=(self.run_id, self.args, iteration),
        )
        process.start()
        self.evaluation_processes.append(process)
        return process

    def _reap_evaluation_processes(self) -> None:
        active_processes: list[Process] = []
        for process in self.evaluation_processes:
            if process.is_alive():
                active_processes.append(process)
            else:
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f'Evaluation process {process.pid} exited with code {process.exitcode}.')
        self.evaluation_processes = active_processes

    def _wait_for_all_evaluations(self) -> bool:
        while self.evaluation_processes:
            stop_reason = self._stop_reason()
            if stop_reason is not None:
                self.final_stop_reason = stop_reason
                warn(f'Stopping evaluation: {stop_reason}')
                for process in self.evaluation_processes:
                    process.terminate()
                    process.join(timeout=10)
                self.evaluation_processes = []
                return False

            for process in self.evaluation_processes:
                process.join(timeout=1)
            self._reap_evaluation_processes()
        return True

    def _setup_connections(self) -> None:
        for node_id, device_id in enumerate(self.args.cluster.self_play_device_ids):
            self.self_play_processes.append(self._start_self_play_process(node_id, device_id))

        log(f'Started {len(self.self_play_processes)} SelfPlay processes on {torch.cuda.device_count()} devices.')

    def _start_self_play_process(self, node_id: int, device_id: int) -> Process:
        """Starts a SelfPlay process for the given node_id and returns the process."""
        process = Process(
            target=run_self_play_process,
            args=(self.run_id, self.args, self.communication_folder, 0, node_id),
        )
        start_process_on_cuda_device(process, device_id)
        return process

    def _ensure_processes_are_running(self) -> None:
        for i, process in enumerate(list(self.self_play_processes)):
            # 15 minutes since we check in after every move was played, so not very long timeouts required
            if self._ensure_process_is_running(process, f'SELF PLAY {i}', timeout=15 * 60):
                # if the process is not alive, restart it
                device_id = self.args.cluster.self_play_device_ids[i]
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

    def _ensure_model_exists(self, starting_iteration: int) -> None:
        if checkpoint_manifest_path(starting_iteration, self.args.save_path).exists():
            load_model_and_optimizer(
                starting_iteration,
                self.args.network,
                torch.device(
                    self.args.cluster.trainer_device_type,
                    self.args.cluster.trainer_rank_zero_device_id,
                ),
                self.args.save_path,
                self.args.training.optimizer,
            )
            return
        model, optimizer = load_model_and_optimizer(
            starting_iteration,
            self.args.network,
            torch.device(
                self.args.cluster.trainer_device_type,
                self.args.cluster.trainer_rank_zero_device_id,
            ),
            self.args.save_path,
            self.args.training.optimizer,
        )
        save_model_and_optimizer(model, optimizer, starting_iteration, self.args.save_path)

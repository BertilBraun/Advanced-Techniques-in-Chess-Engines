from typing import Generator
import torch
import psutil
from torch.multiprocessing import Process
from pathlib import Path
from time import monotonic

from src.cluster.GatingProcess import GatingProcess
from src.train.TrainingArgs import TrainingArgs
from src.settings import USE_GPU
from src.train.TrainingStats import TrainingStats
from src.util.communication import Communication
from src.util.exceptions import log_exceptions
from src.util.log import log, warn
from src.util.save_paths import (
    checkpoint_manifest_path,
    get_latest_model_iteration,
    load_model_and_optimizer,
    save_model_and_optimizer,
)
from src.cluster.EvaluationProcess import run_evaluation_process
from src.cluster.SelfPlayProcess import run_self_play_process
from src.cluster.TrainerProcess import TrainerProcess, number_of_games_in_iteration
from src.util.tensorboard import TensorboardWriter
from src.util.timing import reset_times
from src.experiment.resource_telemetry import process_tree_open_file_counts
from src.experiment.progress_telemetry import IterationTelemetry, append_iteration_telemetry
from src.experiment.cost_accounting import estimated_cost


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

        self.self_play_nodes_on_trainer_device: list[int] = []

        self.trainer_device_id = self.args.cluster.trainer_device_id if USE_GPU else 0
        self.started_at = started_at
        self.final_stop_reason: str | None = None
        self.latest_completed_iteration = 0

    def run(self) -> Generator[tuple[int, tuple[TrainingStats, TrainingStats]], None, None]:
        """The main loop of the CommanderProcess. The resulting generator yields after each iteration. If the Generator is not consumed, no further iterations will be trained."""

        Path(self.args.save_path).mkdir(parents=True, exist_ok=True)

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
            self._shutdown()

    def _initialize_workers(
        self,
        starting_iteration: int,
        current_best_iteration: int,
    ) -> tuple[TrainerProcess, GatingProcess]:
        try:
            log('Setting up connections...')
            self._setup_connections()
            trainer = TrainerProcess(self.args, self.run_id, self.trainer_device_id)
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
    ) -> Generator[tuple[int, tuple[TrainingStats, TrainingStats]], None, None]:
        for iteration in range(starting_iteration, self.args.num_iterations):
            stop_reason = self._stop_reason()
            if stop_reason is not None:
                self.final_stop_reason = stop_reason
                warn(f'Stopping before iteration {iteration}: {stop_reason}')
                break

            self._ensure_processes_are_running()
            log(f'Starting iteration {iteration}.')
            self.communication.boardcast(f'START AT ITERATION: {iteration}')

            with TensorboardWriter(self.run_id, 'trainer', postfix_pid=False):
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

                for node_id in self.self_play_nodes_on_trainer_device[::2]:
                    self.communication.send_to_id('STOP SELF PLAY', node_id)

                training_started_at = monotonic()
                training_result = trainer.train(iteration)
                training_finished_at = monotonic()
                self.latest_completed_iteration = iteration + 1
                self._write_iteration_telemetry(
                    iteration=iteration,
                    games_at_wait_start=games_at_wait_start,
                    games_at_wait_end=games_at_wait_end,
                    wait_seconds=wait_finished_at - wait_started_at,
                    replay_samples_loaded=trainer.rolling_buffer.stats.num_samples,
                    replay_games_loaded=trainer.rolling_buffer.stats.num_games,
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

    def _shutdown(self) -> None:
        log('Training complete. Sending STOP to all processes.')
        self.communication.boardcast('STOP')

        for process in self.self_play_processes:
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

            if device_id == self.trainer_device_id:
                self.self_play_nodes_on_trainer_device.append(node_id)

        log(f'Started {len(self.self_play_processes)} SelfPlay processes on {torch.cuda.device_count()} devices.')

    def _start_self_play_process(self, node_id: int, device_id: int) -> Process:
        """Starts a SelfPlay process for the given node_id and returns the process."""
        process = Process(
            target=run_self_play_process,
            args=(self.run_id, self.args, self.communication_folder, device_id, node_id),
        )
        process.start()
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
                torch.device('cuda' if USE_GPU else 'cpu'),
                self.args.save_path,
                self.args.training.optimizer,
            )
            return
        model, optimizer = load_model_and_optimizer(
            starting_iteration,
            self.args.network,
            torch.device('cuda' if USE_GPU else 'cpu'),
            self.args.save_path,
            self.args.training.optimizer,
        )
        save_model_and_optimizer(model, optimizer, starting_iteration, self.args.save_path)

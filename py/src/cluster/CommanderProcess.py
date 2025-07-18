from typing import Generator
import torch
from torch.multiprocessing import Process
from pathlib import Path

from src.cluster.GatingProcess import GatingProcess
from src.train.TrainingArgs import TrainingArgs
from src.settings import USE_GPU
from src.train.TrainingStats import TrainingStats
from src.util.communication import Communication
from src.util.exceptions import log_exceptions
from src.util.log import log, warn
from src.util.save_paths import get_latest_model_iteration, load_model_and_optimizer, save_model_and_optimizer
from src.cluster.EvaluationProcess import run_evaluation_process
from src.cluster.SelfPlayProcess import run_self_play_process
from src.cluster.TrainerProcess import TrainerProcess
from src.util.tensorboard import TensorboardWriter
from src.util.timing import reset_times


class CommanderProcess:
    """The CommanderProcess is the main process that manages the communication between the Trainer, SelfPlay and InferenceServer processes.

    It starts the Trainer and SelfPlay processes and sends them the current iteration number.
    Once the Trainer is finished, it receives the training stats from the Trainer.
    It then starts the EvaluationProcess and sends the new iteration number to the Trainer and SelfPlay processes.

    Once all iterations are done, it sends a STOP message to all processes and waits for them to finish.
    """

    def __init__(self, run: int, args: TrainingArgs) -> None:
        self.run_id = run
        self.args = args

        self.communication_folder = f'communication/{self.run_id}'
        self.communication = Communication(self.communication_folder)
        self.communication.clear_all()

        self.self_play_processes: list[Process] = []

        self.self_play_nodes_on_trainer_device: list[int] = []

        self.trainer_device_id = torch.cuda.device_count() - 1 if USE_GPU else 0

    def run(self) -> Generator[tuple[int, tuple[TrainingStats, TrainingStats]], None, None]:
        """The main loop of the CommanderProcess. The resulting generator yields after each iteration. If the Generator is not consumed, no further iterations will be trained."""

        Path(self.args.save_path).mkdir(parents=True, exist_ok=True)

        starting_iteration = get_latest_model_iteration(self.args.save_path)
        log(f'Starting training at iteration {starting_iteration}.')

        self._ensure_model_exists(starting_iteration)

        current_best_iteration = starting_iteration

        log('Setting up connections...')
        self._setup_connections()

        trainer = TrainerProcess(self.args, self.run_id, self.trainer_device_id)
        gating = GatingProcess(self.args, self.run_id, self.trainer_device_id)
        log('Connections set up.')

        # Start CPU usage logger for one SelfPlay process
        self.communication.send_to_id('START USAGE LOGGER', node_id=0)

        # NOTE: Order is important here for the SelfPlayProcess communication.
        self.communication.boardcast(f'LOAD MODEL: {current_best_iteration}')
        self.communication.boardcast(f'START AT ITERATION: {starting_iteration}')

        with log_exceptions('Commander process'):
            for iteration in range(starting_iteration, self.args.num_iterations):
                self._ensure_processes_are_running()

                # send START AT ITERATION: iteration to Trainer and InferenceServers and SelfPlayers
                log(f'Starting iteration {iteration}.')
                self.communication.boardcast(f'START AT ITERATION: {iteration}')

                # Wait for Trainer to finish
                with TensorboardWriter(self.run_id, 'trainer', postfix_pid=False):
                    trainer.wait_for_enough_training_samples(iteration)
                    trainer.load_all_memories_to_train_on_for_iteration(iteration)

                    # TODO make the portion a hyperparameter?
                    for node_id in self.self_play_nodes_on_trainer_device[::2]:
                        # only stop half of the self-play processes for gating
                        self.communication.send_to_id('STOP SELF PLAY', node_id)

                    yield iteration, trainer.train(iteration)
                    log(f'Training finished for iteration {iteration}')

                    reset_times()

                current_best_iteration = gating.run(iteration, current_best_iteration)

                # start EvaluationProcess
                self._start_evaluation_process(iteration)
                log(f'Started evaluation process for iteration {iteration}.')
                # lot(f'Finished evaluation process for iteration {iteration}.')

        log('Training complete. Sending STOP to all processes.')
        self.communication.boardcast('STOP')

        for process in self.self_play_processes:
            process.terminate()

    def _start_evaluation_process(self, iteration: int) -> Process:
        """Starts an EvaluationProcess for the given iteration and returns the process."""
        process = Process(
            target=run_evaluation_process,
            args=(self.run_id, self.args, iteration + 1),
        )
        process.start()
        return process

    def _setup_connections(self) -> None:
        for node_id in range(self.args.cluster.num_self_play_nodes_on_cluster):
            self.self_play_processes.append(self._start_self_play_processes(node_id))

            if self._get_device_id(node_id) == self.trainer_device_id:
                self.self_play_nodes_on_trainer_device.append(node_id)

        log(f'Started {len(self.self_play_processes)} SelfPlay processes on {torch.cuda.device_count()} devices.')

    def _start_self_play_processes(self, node_id: int) -> Process:
        """Starts a SelfPlay process for the given node_id and returns the process."""
        device_id = self._get_device_id(node_id)
        process = Process(
            target=run_self_play_process,
            args=(self.run_id, self.args, self.communication_folder, device_id, node_id),
        )
        process.start()
        return process

    def _ensure_processes_are_running(self):
        for i, process in enumerate(list(self.self_play_processes)):
            # 15 minutes since we check in after every move was played, so not very long timeouts required
            if self._ensure_process_is_running(process, f'SELF PLAY {i}', timeout=15 * 60):
                # if the process is not alive, restart it
                self.self_play_processes[i] = self._start_self_play_processes(i)

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
        model, optimizer = load_model_and_optimizer(
            starting_iteration,
            self.args.network,
            torch.device('cuda' if USE_GPU else 'cpu'),
            self.args.save_path,
            self.args.training.optimizer,
        )
        save_model_and_optimizer(model, optimizer, starting_iteration, self.args.save_path)

    def _get_device_id(self, i: int) -> int:
        # device 0 should have only half the processes of the other devices as device 0 is 50% occupied by the Trainer
        if not USE_GPU:
            return 0

        total: int = self.args.cluster.num_self_play_nodes_on_cluster
        num_devices: int = torch.cuda.device_count()

        if num_devices == 1:
            log('Warning: Only one device available. Using device 0.')
            return 0

        assert num_devices > 1, 'There must be at least 2 devices to distribute the processes.'

        num_on_each_device = total / num_devices
        num_on_last_device = round((num_on_each_device) / 2)  # TODO parametrize this?

        if i < num_on_last_device:
            return torch.cuda.device_count() - 1

        device_id = (i - num_on_last_device) % (num_devices - 1)
        return device_id

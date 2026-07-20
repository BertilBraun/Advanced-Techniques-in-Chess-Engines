import random
import time
import numpy as np
import torch

from src.cluster.TrainerProcess import number_of_games_in_iteration
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import TensorboardWriter, USE_GPU, USE_CPP, TRAINING_ARGS
from src.util.communication import (
    Communication,
    RESUME_SELF_PLAY,
    SELF_PLAY_PAUSED,
    SELF_PLAY_RESUMED,
    STOP_SELF_PLAY,
)
from src.util.log import log
from src.util.exceptions import log_exceptions
from src.train.TrainingArgs import TrainingArgs
from src.util.profiler import start_cpu_usage_logger
from src.util.background_worker import BackgroundWorker
from src.util.timing import reset_times

if USE_CPP:
    from src.self_play.SelfPlayCpp import SelfPlayCpp as SelfPlay
else:
    from src.self_play.SelfPlayPy import SelfPlayPy as SelfPlay


def run_self_play_process(
    run: int, args: TrainingArgs, communication_folder: str, device_id: int, node_id: int
) -> None:
    if USE_GPU:
        # torch.cuda.set_per_process_memory_fraction(1 / 64, device=device_id)
        torch.cuda.set_device(device_id)

    worker_seed = args.random_seed + node_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)

    self_play_process = SelfPlayProcess(args, communication_folder, device_id=device_id, node_id=node_id, run_id=run)
    with (
        log_exceptions(f'Self play process {node_id} crashed.'),
        TensorboardWriter(
            run,
            'self_play',
            postfix_pid=False,
            enabled=node_id < args.cluster.self_play_tensorboard_processes,
        ),
    ):
        self_play_process.run()


class SelfPlayProcess:
    """This class provides functionality to run the self play process. It runs self play games and saves the dataset to disk. It listens to the commander for messages to start and stop the self play process."""

    def __init__(
        self, args: TrainingArgs, communication_folder: str, device_id: int, node_id: int, run_id: int
    ) -> None:
        self.args = args
        self.self_play = SelfPlay(device_id, args)
        self.communication = Communication(communication_folder)
        self.node_id = node_id
        self.run_id = run_id

    def run(self) -> None:
        current_iteration = -1
        running = False
        paused = False
        usage_logger: BackgroundWorker | None = None

        try:
            while True:
                if running:
                    with log_exceptions('Self playing failed'):
                        self.self_play.self_play()

                    if self.self_play.dataset.stats.num_games >= self.args.self_play.num_games_after_which_to_write:
                        self._save_dataset(current_iteration)

                    if (
                        number_of_games_in_iteration(current_iteration, self.args.save_path)
                        >= TRAINING_ARGS.num_games_per_iteration * 5
                    ):
                        log(f'Iteration {current_iteration} has enough games.')
                        self._save_dataset(current_iteration)
                        running = False
                else:
                    time.sleep(0.1)  # Sleep to avoid busy waiting

                if self.communication.is_received('STOP'):
                    break
                if self.communication.try_receive_from_id('START USAGE LOGGER', self.node_id):
                    if usage_logger is not None:
                        usage_logger.stop()
                    usage_logger = start_cpu_usage_logger(self.run_id, 'self_play')
                if self.communication.try_receive_from_id(STOP_SELF_PLAY, self.node_id):
                    self._save_dataset(current_iteration)
                    running = False
                    paused = True
                    self.communication.send_to_id(SELF_PLAY_PAUSED, self.node_id)
                if self.communication.try_receive_from_id(RESUME_SELF_PLAY, self.node_id):
                    paused = False
                    running = current_iteration >= 0
                    self.communication.send_to_id(SELF_PLAY_RESUMED, self.node_id)

                for iteration in range(self.args.num_iterations, current_iteration, -1) if not paused else ():
                    start_recieved = self.communication.is_received(f'START AT ITERATION: {iteration}')
                    load_received = self.communication.is_received(f'LOAD MODEL: {iteration}')
                    if start_recieved:
                        self._save_dataset(current_iteration)
                        current_iteration = iteration
                        running = True
                        reset_times()
                    if load_received:
                        self._save_dataset(current_iteration)
                        self.self_play.update_iteration(iteration)
                        current_iteration = iteration

                    if start_recieved or load_received:
                        break

                self.communication.send_heartbeat(f'SELF PLAY {self.node_id}')
        finally:
            if usage_logger is not None:
                usage_logger.stop()

        log('Self play process stopped.')

    def _save_dataset(self, iteration: int) -> None:
        if not len(self.self_play.dataset):
            return

        subsampled_size = int(self.args.self_play.portion_of_samples_to_keep * len(self.self_play.dataset))
        subsampled_dataset = self.self_play.dataset.sample(subsampled_size)
        subsampled_dataset.save(self.args.save_path, iteration)
        self.self_play.dataset = SelfPlayDataset()

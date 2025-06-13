import random
import time
import numpy as np
import torch
import torch.multiprocessing as mp

from src.cluster.TrainerProcess import number_of_games_in_iteration
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import TensorboardWriter, USE_GPU, USE_CPP, TRAINING_ARGS
from src.util.communication import Communication
from src.util.log import log, warn
from src.util.exceptions import log_exceptions
from src.train.TrainingArgs import TrainingArgs
from src.util.profiler import start_cpu_usage_logger

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

    # Seed for random number generation
    random.seed(mp.current_process().pid)
    torch.manual_seed(mp.current_process().pid)
    np.random.seed(mp.current_process().pid)

    self_play_process = SelfPlayProcess(args, communication_folder, device_id=device_id, node_id=node_id, run_id=run)
    with log_exceptions(f'Self play process {device_id} crashed.'), TensorboardWriter(run, 'self_play'):
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

    def run(self):
        current_iteration = -1
        running = False

        while True:
            if running:
                try:
                    self.self_play.self_play()
                except Exception as e:
                    warn(f'Self playing failed with error: {e}')

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
                start_cpu_usage_logger(self.run_id, 'self_play')
            if self.communication.try_receive_from_id('STOP SELF PLAY', self.node_id):
                self._save_dataset(current_iteration)
                running = False

            for iteration in range(current_iteration + 1, self.args.num_iterations):
                if self.communication.is_received(f'START AT ITERATION: {iteration}'):
                    self._save_dataset(current_iteration)
                    current_iteration = iteration
                    running = True
                if self.communication.is_received(f'LOAD MODEL: {iteration}'):
                    self._save_dataset(current_iteration)
                    self.self_play.update_iteration(iteration)

        log('Self play process stopped.')

    def _save_dataset(self, iteration: int):
        if not len(self.self_play.dataset):
            return

        subsampled_size = int(self.args.self_play.portion_of_samples_to_keep * len(self.self_play.dataset))
        # subsampled_dataset = self.self_play.dataset.choose_only_samples_with_high_policy_spikyness(subsampled_size)
        subsampled_dataset = self.self_play.dataset.sample_by_policy_spikyness(subsampled_size)
        subsampled_dataset.save(self.args.save_path, iteration)
        self.self_play.dataset = SelfPlayDataset()

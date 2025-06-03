import random
import numpy as np
import torch
import torch.multiprocessing as mp

from src.cluster.TrainerProcess import number_of_games_in_iteration
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import TensorboardWriter, USE_GPU, TRAINING_ARGS
from src.util.log import log
from src.self_play.SelfPlay import SelfPlay
from src.cluster.InferenceClient import InferenceClient
from src.util.exceptions import log_exceptions
from src.train.TrainingArgs import SelfPlayParams, TrainingArgs
from src.util.PipeConnection import PipeConnection
from src.util.profiler import start_cpu_usage_logger
from src.util.save_paths import model_save_path


def run_self_play_process(run: int, args: TrainingArgs, commander_pipe: PipeConnection, device_id: int):
    assert commander_pipe.readable and not commander_pipe.writable, 'Commander pipe must be readable and not writable.'

    if USE_GPU:
        # torch.cuda.set_per_process_memory_fraction(1 / 64, device=device_id)
        torch.cuda.set_device(device_id)

    # Seed for random number generation
    random.seed(mp.current_process().pid)
    torch.manual_seed(mp.current_process().pid)
    np.random.seed(mp.current_process().pid)

    client = InferenceClient(device_id, args.network, args.save_path)
    self_play_process = SelfPlayProcess(client, args.self_play, args.save_path, commander_pipe)
    with log_exceptions(f'Self play process {device_id} crashed.'), TensorboardWriter(run, 'self_play'):
        self_play_process.run()


class SelfPlayProcess:
    """This class provides functionality to run the self play process. It runs self play games and saves the dataset to disk. It listens to the commander for messages to start and stop the self play process."""

    def __init__(
        self, client: InferenceClient, args: SelfPlayParams, save_path: str, commander_pipe: PipeConnection
    ) -> None:
        self.save_path = save_path
        self.args = args
        self.self_play = SelfPlay(client, args)
        self.commander_pipe = commander_pipe

    def run(self):
        current_iteration = -1
        running = False

        while True:
            if running:
                self.self_play.self_play()

                if self.self_play.dataset.stats.num_games >= self.args.num_games_after_which_to_write:
                    self._save_dataset(current_iteration)

                if model_save_path(current_iteration + 1, self.save_path).exists():
                    self._save_dataset(current_iteration)
                    self.self_play.update_iteration(current_iteration + 1)
                    current_iteration += 1

                if (
                    number_of_games_in_iteration(current_iteration, self.save_path)
                    >= TRAINING_ARGS.num_games_per_iteration * 5
                ):
                    log(f'Iteration {current_iteration} has enough games.')
                    running = False
                    self.self_play.dataset = SelfPlayDataset()

            if self.commander_pipe.poll():
                message = self.commander_pipe.recv()
                assert isinstance(message, str), f'Expected message to be a string, got {message}'
                if message == 'STOP':
                    break
                elif message.startswith('START AT ITERATION:'):
                    self._save_dataset(current_iteration)
                    current_iteration = int(message.split(':')[-1])
                    running = True
                elif message.startswith('LOAD MODEL:'):
                    old_current_iteration = current_iteration
                    current_iteration = int(message.split(':')[-1])
                    if old_current_iteration != current_iteration:
                        self._save_dataset(current_iteration)
                        self.self_play.update_iteration(current_iteration)
                elif message.startswith('START USAGE LOGGER:'):
                    run_id = int(message.split(':')[-1])
                    start_cpu_usage_logger(run_id, 'self_play')

        log('Self play process stopped.')

    def _save_dataset(self, iteration: int):
        if not len(self.self_play.dataset):
            return

        subsampled_size = int(self.args.portion_of_samples_to_keep * len(self.self_play.dataset))
        # subsampled_dataset = self.self_play.dataset.choose_only_samples_with_high_policy_spikyness(subsampled_size)
        subsampled_dataset = self.self_play.dataset.sample_by_policy_spikyness(subsampled_size)
        subsampled_dataset.save(self.save_path, iteration)
        self.self_play.dataset = SelfPlayDataset()

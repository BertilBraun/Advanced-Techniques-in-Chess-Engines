import time
import torch
import numpy as np
from pathlib import Path

from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.settings import TRAINING_ARGS, USE_GPU, log_histogram, log_scalar, CurrentGame
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.util.save_paths import load_model_and_optimizer, save_model_and_optimizer
from src.alpha_zero.train.Trainer import Trainer
from src.alpha_zero.train.TrainingStats import TrainingStats
from src.util.PipeConnection import PipeConnection


def run_trainer_process(commander_pipe: PipeConnection, device_id: int):
    assert commander_pipe.readable and commander_pipe.writable, 'Commander pipe must be readable and writable.'
    assert 0 <= device_id < torch.cuda.device_count() or not USE_GPU, f'Invalid device ID ({device_id})'

    trainer_process = TrainerProcess(device_id, commander_pipe)
    with log_exceptions('Trainer process'):
        trainer_process.run()


class TrainerProcess:
    def __init__(self, device_id: int, commander_pipe: PipeConnection) -> None:
        self.args = TRAINING_ARGS
        self.device = torch.device('cuda', device_id) if USE_GPU else torch.device('cpu')

        self.commander_pipe = commander_pipe

    def run(self):
        while True:
            message = self.commander_pipe.recv()
            assert isinstance(message, str), f'Expected message to be a string, got {message}'
            if message == 'STOP':
                break
            elif message.startswith('START AT ITERATION:'):
                iteration = int(message.split(':')[-1])
                with log_exceptions('Training'):
                    self.train(iteration)
                self.commander_pipe.send('FINISHED')

        log('Training process stopped.')

    def train(self, iteration: int) -> None:
        model, optimizer = load_model_and_optimizer(iteration, self.args.network, self.device)

        trainer = Trainer(model, optimizer, self.args.training)

        self._wait_for_enough_training_samples(iteration)

        dataset = self._load_all_memories_to_train_on_for_iteration(iteration)
        log(f'Loaded {len(dataset)} self-play memories.')

        train_stats = TrainingStats()

        for epoch in range(self.args.training.num_epochs):
            epoch_train_stats = trainer.train(dataset, iteration)
            log(f'Epoch {epoch + 1}: {epoch_train_stats}')
            train_stats += epoch_train_stats

        save_model_and_optimizer(model, optimizer, iteration + 1)

        log(f'Iteration {iteration + 1}: {train_stats}')
        self._log_to_tensorboard(iteration, train_stats, dataset)

    def _wait_for_enough_training_samples(self, iteration):
        EXPECTED_NUM_SAMPLES = self.args.num_games_per_iteration * CurrentGame.average_num_moves_per_game

        def num_samples_in_current_and_previous_iteration(iteration: int) -> int:
            current_len = len(SelfPlayDataset.load_iteration(self.args.save_path, iteration))
            previous_len = len(SelfPlayDataset.load_iteration(self.args.save_path, iteration - 1))
            return current_len + previous_len

        i = 0
        while (samples := num_samples_in_current_and_previous_iteration(iteration)) < EXPECTED_NUM_SAMPLES:
            if i % 30 == 0:
                log(f'Waiting for enough samples for iteration {iteration} ({samples} < {EXPECTED_NUM_SAMPLES})')
            i += 1
            time.sleep(1)

    def _log_to_tensorboard(self, iteration: int, train_stats: TrainingStats, dataset: SelfPlayDataset) -> None:
        log_scalar('policy_loss', train_stats.policy_loss, iteration)
        log_scalar('value_loss', train_stats.value_loss, iteration)
        log_scalar('total_loss', train_stats.total_loss, iteration)

        log_scalar('num_training_samples', len(dataset), iteration)
        log_scalar('num_deduplicated_samples', len(dataset), iteration)

        spikiness = sum(policy_target.max() for policy_target in dataset.policy_targets) / len(dataset)
        log_scalar('policy_spikiness', spikiness, iteration)

        log_histogram('policy_targets', np.array(dataset.policy_targets), iteration)
        log_histogram('value_targets', np.array(dataset.value_targets), iteration)

    def _load_all_memories_to_train_on_for_iteration(self, iteration: int) -> SelfPlayDataset:
        window_size = self.args.training.sampling_window(iteration)

        log(
            f'Loading memories for iteration {iteration} with window size {window_size} ({max(iteration - window_size, 0)}-{iteration})'
        )

        dataset = SelfPlayDataset()
        for iter in range(max(iteration - window_size, 0), iteration + 1):
            iteration_dataset = SelfPlayDataset.load_iteration(self.args.save_path, iter)
            file_paths = SelfPlayDataset.get_files_to_load_for_iteration(self.args.save_path, iter)

            if len(iteration_dataset) != 0 and len(file_paths) > 1:
                iteration_dataset.deduplicate()
                for file_path in file_paths:
                    Path(file_path).unlink()
                iteration_dataset.save(self.args.save_path, iter, 'deduplicated')
                log(f'Deduplicated memory_{iter} to {len(iteration_dataset)}')

            dataset += iteration_dataset

        return dataset

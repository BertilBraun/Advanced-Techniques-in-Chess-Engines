import time
import torch

from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.SelfPlayTrainDataset import SelfPlayTrainDataset
from src.train.TrainingArgs import TrainingArgs
from src.settings import USE_GPU, TensorboardWriter
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.util.profiler import start_cpu_usage_logger
from src.util.timing import reset_times, timeit
from src.util.save_paths import load_model_and_optimizer, save_model_and_optimizer
from src.train.Trainer import Trainer
from src.train.TrainingStats import TrainingStats
from src.util.PipeConnection import PipeConnection


def run_trainer_process(run: int, args: TrainingArgs, commander_pipe: PipeConnection, device_id: int):
    assert commander_pipe.readable and commander_pipe.writable, 'Commander pipe must be readable and writable.'
    assert 0 <= device_id < torch.cuda.device_count() or not USE_GPU, f'Invalid device ID ({device_id})'

    start_cpu_usage_logger(run, 'trainer')

    trainer_process = TrainerProcess(args, run, device_id, commander_pipe)
    with log_exceptions('Trainer process'), TensorboardWriter(run, 'trainer', postfix_pid=False):
        trainer_process.run()


class TrainerProcess:
    """This class provides functionality to train the model on the self play data. It listens to the commander for messages to start and stop the training process. Once it receives a start message, it waits for enough samples to be available for the current iteration and then trains the model for the specified number of epochs. The training stats are sent back to the commander once training is finished."""

    def __init__(self, args: TrainingArgs, run_id: int, device_id: int, commander_pipe: PipeConnection) -> None:
        self.args = args
        self.run_id = run_id
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
                training_stats = self.train(iteration)
                reset_times()
                self.commander_pipe.send(training_stats)
                self.commander_pipe.send('FINISHED')

        log('Training process stopped.')

    @timeit
    def train(self, iteration: int) -> TrainingStats:
        model, optimizer = load_model_and_optimizer(iteration, self.args.network, self.device, self.args.save_path)

        trainer = Trainer(model, optimizer, self.args.training)

        self._wait_for_enough_training_samples(iteration)

        train_stats = TrainingStats()
        valid_stats = TrainingStats()

        for epoch in range(self.args.training.num_epochs):
            # Only loads a light wrapper around the entire dataset
            dataset, validation_dataset = self._load_all_memories_to_train_on_for_iteration(iteration)
            dataloader = dataset.as_dataloader(self.args.training.batch_size, self.args.training.num_workers)
            validation_dataloader = validation_dataset.as_dataloader(
                self.args.training.batch_size, self.args.training.num_workers
            )

            epoch_train_stats, epoch_valid_stats = trainer.train(dataloader, validation_dataloader, iteration)
            train_stats += epoch_train_stats
            valid_stats += epoch_valid_stats

            if train_stats.value_std < 0.01:
                exit()

            save_model_and_optimizer(model, optimizer, iteration + 1, self.args.save_path)

        train_stats.log_to_tensorboard(iteration, 'train')
        valid_stats.log_to_tensorboard(iteration, 'validation')

        return train_stats

    @timeit
    def _wait_for_enough_training_samples(self, iteration):
        def games(iteration: int) -> int:
            return SelfPlayDataset.load_iteration_stats(self.args.save_path, iteration).num_games

        while games(iteration) + 0.5 * games(iteration - 1) < self.args.num_games_per_iteration:
            time.sleep(10)

    @timeit
    def _load_all_memories_to_train_on_for_iteration(
        self, iteration: int
    ) -> tuple[SelfPlayTrainDataset, SelfPlayTrainDataset]:
        window_size = self.args.training.sampling_window(iteration)

        log(
            f'Loading memories for iteration {iteration} with window size {window_size} ({max(iteration - window_size, 0)}-{iteration})'
        )

        all_dataset_files = [
            (iteration, SelfPlayDataset.get_files_to_load_for_iteration(self.args.save_path, iteration))
            for iteration in range(max(iteration - window_size, 0), iteration + 1)
        ]
        validation_dataset_file = (all_dataset_files[-1][0], [all_dataset_files[-1][1].pop(-1)])

        dataset = SelfPlayTrainDataset(self.run_id, self.device)
        dataset.load_from_files(self.args.save_path, all_dataset_files)

        log(f'Loaded {dataset.stats.num_samples} samples from {dataset.stats.num_games} games')

        validation_dataset = SelfPlayTrainDataset(self.run_id, self.device)
        validation_dataset.load_from_files(self.args.save_path, [validation_dataset_file])

        return dataset, validation_dataset

import time
import torch
from tqdm import tqdm

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

    torch.cuda.set_device(device_id)

    trainer_process = TrainerProcess(args, run, device_id, commander_pipe)
    with log_exceptions('Trainer process'), TensorboardWriter(run, 'trainer', postfix_pid=False):
        trainer_process.run()


def number_of_games_in_iteration(iteration: int, save_path: str) -> int:
    """Returns the number of games in the given iteration."""
    return SelfPlayDataset.load_iteration_stats(save_path, iteration).num_games


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
        model, optimizer = load_model_and_optimizer(
            iteration,
            self.args.network,
            self.device,
            self.args.save_path,
            self.args.training.optimizer,
        )

        trainer = Trainer(model, optimizer, self.args.training)

        self._wait_for_enough_training_samples(iteration)
        dataset, validation_dataset = self._load_all_memories_to_train_on_for_iteration(iteration)

        train_stats: list[TrainingStats] = []
        valid_stats: list[TrainingStats] = []

        for epoch in range(self.args.training.num_epochs):
            dataloader = dataset.as_dataloader(self.args.training.batch_size, self.args.training.num_workers)
            validation_dataloader = validation_dataset.as_dataloader(
                self.args.training.batch_size, self.args.training.num_workers
            )

            epoch_train_stats, epoch_valid_stats = trainer.train(dataloader, validation_dataloader, iteration)
            train_stats.append(epoch_train_stats)
            valid_stats.append(epoch_valid_stats)

            if epoch_valid_stats.value_std < 0.01:
                log('Training stopped early due to low value std deviation.')
                exit()

            save_model_and_optimizer(model, optimizer, iteration + 1, self.args.save_path)
            if number_of_games_in_iteration(iteration, self.args.save_path) >= self.args.num_games_per_iteration:
                log('Enough games played, stopping training for this iteration.')
                break

        combined_train_stats = TrainingStats.combine(train_stats)
        combined_valid_stats = TrainingStats.combine(valid_stats)
        combined_train_stats.log_to_tensorboard(iteration, 'train')
        combined_valid_stats.log_to_tensorboard(iteration, 'validation')

        return combined_train_stats

    @timeit
    def _wait_for_enough_training_samples(self, iteration):
        def games(iteration: int) -> int:
            """Returns the number of games in the given iteration."""
            return number_of_games_in_iteration(iteration, self.args.save_path)

        target_games = self.args.num_games_per_iteration
        with tqdm(total=target_games, desc=f'Waiting for games (iter {iteration})') as pbar:
            current_games = games(iteration) + 0.5 * games(iteration - 1)
            pbar.update(int(current_games))

            while current_games < target_games:
                time.sleep(10)
                new_games = games(iteration) + 0.5 * games(iteration - 1)
                if new_games > current_games:
                    pbar.update(int(new_games - current_games))
                    current_games = new_games

    @timeit
    def _load_all_memories_to_train_on_for_iteration(
        self, iteration: int
    ) -> tuple[SelfPlayTrainDataset, SelfPlayTrainDataset]:
        window_size = self.args.training.sampling_window(iteration)

        log(
            f'Loading memories for iteration {iteration} with window size {window_size} ({max(iteration - window_size, 0)}-{iteration})'
        )

        all_dataset_files = [
            file
            for iteration in range(max(iteration - window_size, 0), iteration + 1)
            for file in SelfPlayDataset.get_files_to_load_for_iteration(self.args.save_path, iteration)
        ]

        validation_dataset = SelfPlayTrainDataset()
        while len(validation_dataset) == 0 and len(all_dataset_files) > 0:
            # The newest file is the validation dataset
            validation_dataset_file = all_dataset_files.pop(-1)
            validation_dataset.load_from_files([validation_dataset_file])

        dataset = SelfPlayTrainDataset()
        dataset.load_from_files(all_dataset_files)
        dataset.log_all_dataset_stats(self.run_id)

        log(f'Loaded {dataset.stats.num_samples} samples from {dataset.stats.num_games} games')

        return dataset, validation_dataset

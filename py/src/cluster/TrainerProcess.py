import os
import time
import torch
from tqdm import tqdm

from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.train.RollingSelfPlayBuffer import RollingSelfPlayBuffer
from src.train.TrainingArgs import TrainingArgs
from src.settings import USE_GPU
from src.util.log import error, log
from src.util.profiler import start_cpu_usage_logger
from src.util.tensorboard import log_scalar
from src.util.timing import timeit
from src.util.save_paths import load_model_and_optimizer, optimizer_save_path, save_model_and_optimizer
from src.train.Trainer import Trainer
from src.train.TrainingStats import TrainingStats


def number_of_games_in_iteration(iteration: int, save_path: str) -> int:
    """Returns the number of games in the given iteration."""
    return SelfPlayDataset.load_iteration_stats(save_path, iteration).num_games


class TrainerProcess:
    """This class provides functionality to train the model on the self play data. It listens to the commander for messages to start and stop the training process. Once it receives a start message, it waits for enough samples to be available for the current iteration and then trains the model for the specified number of epochs. The training stats are sent back to the commander once training is finished."""

    def __init__(self, args: TrainingArgs, run_id: int, device_id: int) -> None:
        assert 0 <= device_id < torch.cuda.device_count() or not USE_GPU, f'Invalid device ID ({device_id})'

        self.args = args
        self.run_id = run_id
        self.device = torch.device('cuda', device_id) if USE_GPU else torch.device('cpu')

        torch.set_num_threads(32)  # Set number of threads for CPU operations
        torch.set_num_interop_threads(4)  # Set number of threads for interop operations
        os.environ['OMP_NUM_THREADS'] = '32'
        os.environ['MKL_NUM_THREADS'] = '32'
        os.environ['OPENBLAS_NUM_THREADS'] = '32'

        if USE_GPU:
            torch.cuda.set_device(device_id)

        start_cpu_usage_logger(self.run_id, 'trainer')

        self.rolling_buffer = RollingSelfPlayBuffer(max_buffer_samples=args.training.max_buffer_samples)
        self.validation_dataset = SelfPlayDataset()

    @timeit
    def train(self, iteration: int) -> tuple[TrainingStats, TrainingStats]:
        model, optimizer = load_model_and_optimizer(
            iteration,
            self.args.network,
            self.device,
            self.args.save_path,
            self.args.training.optimizer,
        )

        trainer = Trainer(model, optimizer, self.args.training)

        dataloader = as_dataloader(
            self.rolling_buffer,
            self.args.training.batch_size,
            self.args.training.num_workers,
        )
        validation_dataloader = as_dataloader(
            self.validation_dataset,
            self.args.training.batch_size,
            self.args.training.num_workers,
            drop_last=True,
        )
        train_stats: list[TrainingStats] = []
        valid_stats: list[TrainingStats] = []

        for epoch in range(self.args.training.num_epochs):
            # print num threads and interop threads
            try:
                epoch_train_stats, epoch_valid_stats = trainer.train(dataloader, validation_dataloader, iteration)
            except RuntimeError as e:
                if 'returned nan values' in str(e):
                    error('Training failed due to NaN values in the model output. Retrying with a fresh optimizer.')
                    # Reset optimizer to avoid NaN issues
                    optimizer_save_path(iteration, self.args.save_path).unlink(missing_ok=True)
                    return self.train(iteration)
                else:
                    raise e

            train_stats.append(epoch_train_stats)
            valid_stats.append(epoch_valid_stats)

            log('Train stats: ', epoch_train_stats)
            log('Valid stats: ', epoch_valid_stats)

            if epoch_train_stats.value_std < 0.01:
                log('Training stopped early due to low value std deviation.')
                exit()

            save_model_and_optimizer(model, optimizer, iteration + 1, self.args.save_path)
            if number_of_games_in_iteration(iteration, self.args.save_path) >= self.args.num_games_per_iteration * 2:
                log('Enough games played, stopping training for this iteration.')
                break

        training_stats = TrainingStats.combine(train_stats)
        validation_stats = TrainingStats.combine(valid_stats)
        training_stats.log_to_tensorboard(iteration, 'train')
        validation_stats.log_to_tensorboard(iteration, 'validation')

        return training_stats, validation_stats

    @timeit
    def wait_for_enough_training_samples(self, iteration):
        def games(iteration: int) -> int:
            """Returns the number of games in the given iteration."""
            return number_of_games_in_iteration(iteration, self.args.save_path)

        target_games = self.args.num_games_per_iteration
        current_games = games(iteration) + 0.5 * games(iteration - 1)
        with tqdm(total=target_games, desc=f'Waiting for games (iter {iteration})', initial=int(current_games)) as pbar:
            while current_games < target_games:
                time.sleep(1)
                new_games = games(iteration) + 0.5 * games(iteration - 1)
                if new_games > current_games:
                    pbar.update(int(min(new_games, target_games) - current_games))
                    current_games = new_games

    @timeit
    def load_all_memories_to_train_on_for_iteration(self, iteration: int) -> None:
        window_size = self.args.training.sampling_window(iteration)

        log(
            f'Loading memories for iteration {iteration} with window size {window_size} ({max(iteration - window_size, 0)}-{iteration})'
        )
        log_scalar('training/window_size', window_size, iteration)

        dataset_files = SelfPlayDataset.get_files_to_load_for_iteration(self.args.save_path, iteration)

        dataset_stats = SelfPlayDataset.load_iteration_stats(self.args.save_path, iteration)

        required_validation_samples = dataset_stats.num_samples * self.args.training.validation_percentage

        self.validation_dataset = SelfPlayDataset()
        if dataset_files:
            while len(self.validation_dataset) < required_validation_samples and len(dataset_files) > 0:
                # The newest file is the validation dataset
                validation_dataset_file = dataset_files.pop(-1)
                self.validation_dataset += SelfPlayDataset.load(validation_dataset_file)
        else:
            previous_iteration_files = SelfPlayDataset.get_files_to_load_for_iteration(
                self.args.save_path, iteration - 1
            )
            assert previous_iteration_files, (
                f'No dataset files found at all for iteration {iteration} or {iteration - 1}'
            )
            while len(self.validation_dataset) < required_validation_samples and len(previous_iteration_files) > 0:
                # The newest file is the validation dataset
                validation_dataset_file = previous_iteration_files.pop(-1)
                self.validation_dataset += SelfPlayDataset.load(validation_dataset_file)

        if len(self.rolling_buffer) == 0:
            # Load all the iterations in the window into the rolling buffer
            for i in range(max(iteration - window_size, 0), iteration):
                iter_dataset_files = SelfPlayDataset.get_files_to_load_for_iteration(self.args.save_path, i)
                if not iter_dataset_files:
                    log(f'No dataset files found for iteration {i}, skipping')
                    continue

                self.rolling_buffer.update(i, window_size, iter_dataset_files)

        self.rolling_buffer.update(iteration, window_size, dataset_files)

        self.rolling_buffer.log_all_dataset_stats(self.run_id)

        log(f'Loaded {self.rolling_buffer.stats.num_samples} samples from {self.rolling_buffer.stats.num_games} games')


def as_dataloader(
    dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int, drop_last: bool = False
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        pin_memory=USE_GPU,
        prefetch_factor=16 if num_workers > 0 else None,
        # fork is not available on Windows
        multiprocessing_context='fork' if num_workers > 0 and os.name != 'nt' else None,
    )

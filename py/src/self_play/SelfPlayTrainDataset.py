from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.multiprocessing import Process

from src.mcts.MCTS import action_probabilities
from src.settings import log_histogram, log_scalar
from src.util.tensorboard import TensorboardWriter
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats

# TODO: The entire selfplaytraindataset should fit into Memory at once, with compressed state and Action probs. This could speed up training by reducing the number of file reads and writes - and significantly improve the shuffling process and thereby reduce the overfitting of the value head.


class SelfPlayTrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset to train the neural network on self-play data. It is a wrapper around multiple SelfPlayDatasets (i.e. Iterations). The Idea is, to load only chunks of the datasets into memory and return the next sample from the next dataset in a round-robin fashion."""

    def __init__(self) -> None:
        self.datasets: list[tuple[SelfPlayDataset | None, Path]] = []
        self.dataset_stats: list[SelfPlayDatasetStats] = []
        self.dataset_length_prefix_sums: list[int] = []

    @property
    def stats(self) -> SelfPlayDatasetStats:
        """Returns the stats of the dataset. This is a sum of all stats of the datasets in the dataset."""
        accumulated_stats = SelfPlayDatasetStats()
        for stats in self.dataset_stats:
            accumulated_stats += stats
        return accumulated_stats

    def load_from_files(self, files: list[Path]) -> None:
        for file in reversed(files):
            self.datasets.append((None, file))
            self.dataset_stats.append(SelfPlayDataset.load_stats(file))

            if self.stats.num_samples > 300_000:
                print(f'Loaded {self.stats.num_samples} datasets, stopping loading more.')
                break

        self.dataset_length_prefix_sums = [0] + list(np.cumsum([stats.num_samples for stats in self.dataset_stats]))

    def log_all_dataset_stats(self, run: int) -> None:
        Process(target=self._log_all_dataset_stats, args=(run,), daemon=True).start()

    def _log_all_dataset_stats(self, run: int) -> None:
        accumulated_stats = self.stats

        with TensorboardWriter(run, 'dataset', postfix_pid=False):
            datasets = [SelfPlayDataset.load(file) for _, file in self.datasets]

            policies = [action_probabilities(visits) for dataset in datasets for visits in dataset.visit_counts]

            spikiness = sum(policy.max() for policy in policies) / len(self)

            log_scalar('dataset/policy_spikiness', spikiness)

            log_histogram('dataset/policy_targets', np.array([policy.max() for policy in policies]))
            log_histogram(
                'dataset/value_targets',
                np.array([value for dataset in datasets for value in dataset.value_targets]),
            )

            # dataset.deduplicate()
            # log_histogram('dataset/value_targets_deduplicated', np.array(dataset.value_targets))

            log_scalar('dataset/num_games', accumulated_stats.num_games)
            log_scalar('dataset/average_game_length', accumulated_stats.game_lengths / accumulated_stats.num_games)
            log_scalar('dataset/num_too_long_games', accumulated_stats.num_too_long_games)

            log_scalar('dataset/num_samples', accumulated_stats.num_samples)
            log_scalar(
                'dataset/average_generation_time',
                accumulated_stats.total_generation_time / accumulated_stats.num_games,
            )

            if accumulated_stats.resignations > 0:
                log_scalar('dataset/num_resignations', accumulated_stats.resignations)
                log_scalar(
                    'dataset/average_resignation_percent',
                    accumulated_stats.resignations / accumulated_stats.num_games * 100,
                )

    def as_dataloader(self, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
        assert num_workers > 0, 'num_workers must be greater than 0'
        num_workers = 1  # Since the Dataset is already loaded into memory and loading each sample is cheap, multiple workers are unnecessary
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=False,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            prefetch_factor=8 if num_workers > 0 else None,
        )

    def __len__(self) -> int:
        return sum(stats.num_samples for stats in self.dataset_stats)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dataset_index = np.searchsorted(self.dataset_length_prefix_sums, idx + 1) - 1
        idx -= self.dataset_length_prefix_sums[dataset_index]
        assert 0 <= dataset_index < len(self.datasets), f'Index {idx} out of bounds for dataset {dataset_index}'

        dataset, path = self.datasets[dataset_index]
        if dataset is None:
            dataset = SelfPlayDataset.load(path)
            self.datasets[dataset_index] = (dataset, path)

        return dataset[idx]

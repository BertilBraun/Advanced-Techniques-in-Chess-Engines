from __future__ import annotations

import torch
import numpy as np
import numpy.typing as npt
from pathlib import Path
from collections import deque
from itertools import accumulate
from threading import Thread
from torch.utils.data import Dataset

from src.settings import log_histogram, log_scalar
from src.util.tensorboard import TensorboardWriter, log_scalars
from src.self_play.SelfPlayDataset import SelfPlayDataset, training_batch_from_raw_samples
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats


def maximum_action_probability(visit_counts: npt.NDArray[np.uint16]) -> float:
    counts = visit_counts[:, 1]
    total_visits = int(np.sum(counts))
    if total_visits <= 0:
        raise ValueError('Visit counts must contain a positive total')
    return float(np.max(counts) / total_visits)


class RollingSelfPlayBuffer(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Keeps a sliding window of recent SelfPlayDataset objects in RAM.
    • The window size in *iterations* comes from  args.training.sampling_window(i)
    • The window size in *samples* is never allowed to exceed MAX_BUFFER_SAMPLES
    """

    def __init__(self, max_buffer_samples: int) -> None:
        self._buf: deque[tuple[int, SelfPlayDataset]] = deque()  # (iteration, dataset)
        self._prefix: list[int] = [0]  # len prefix sums
        self._num_samples = 0
        self._max_buffer_samples = max_buffer_samples
        self._logging_threads: list[Thread] = []

    # ---------- public API used by TrainerProcess ------------------------- #
    def update(self, iteration: int, window_iter: int, files: list[Path]) -> None:
        """
        • Load **only** the hdf5 files for *this* iteration (if not already loaded)
        • Throw away datasets that fall outside the window in either dimension
        """
        self._add_iteration(iteration, files)

        # 1) enforce iteration-based window
        while self._buf and self._buf[0][0] < iteration - window_iter:
            self._drop_left()

        # 2) enforce sample-count ceiling
        while self._num_samples > self._max_buffer_samples and len(self._buf) > 1:
            self._drop_left()

        self._rebuild_prefix()

    @property
    def stats(self) -> SelfPlayDatasetStats:
        """Returns the stats of the dataset. This is a sum of all stats of the datasets in the dataset."""
        accumulated_stats = SelfPlayDatasetStats()
        for _, dataset in self._buf:
            accumulated_stats += dataset.stats
        return accumulated_stats

    # ---------- Dataset interface ---------------------------------------- #
    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds_idx = np.searchsorted(self._prefix, idx + 1) - 1
        inner = idx - self._prefix[ds_idx]
        return self._buf[ds_idx][1][inner]  # (state, π-target, v-target)

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_samples = [self._raw_sample(index) for index in indices]
        return training_batch_from_raw_samples(
            [sample[0] for sample in raw_samples],
            [sample[1] for sample in raw_samples],
            [sample[2] for sample in raw_samples],
        )

    def log_all_dataset_stats(self, run: int) -> None:
        self._logging_threads = [thread for thread in self._logging_threads if thread.is_alive()]
        thread = Thread(
            target=self._log_all_dataset_stats,
            args=(run,),
            daemon=True,
            name=f'dataset-stats-{run}',
        )
        thread.start()
        self._logging_threads.append(thread)

    def close(self) -> None:
        for thread in self._logging_threads:
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError(f'Dataset logging thread {thread.name!r} did not stop.')
        self._logging_threads = []

    def _raw_sample(self, idx: int) -> tuple[bytes, npt.NDArray[np.uint16], float]:
        dataset_index = np.searchsorted(self._prefix, idx + 1) - 1
        inner_index = idx - self._prefix[dataset_index]
        return self._buf[dataset_index][1].raw_sample(inner_index)

    def _log_all_dataset_stats(self, run: int) -> None:
        accumulated_stats = self.stats

        with TensorboardWriter(run, 'dataset', postfix_pid=False):
            policy_maxima = np.fromiter(
                (maximum_action_probability(visits) for _, dataset in self._buf for visits in dataset.visit_counts),
                dtype=np.float32,
                count=len(self),
            )
            value_targets = np.fromiter(
                (value for _, dataset in self._buf for value in dataset.value_targets),
                dtype=np.float32,
                count=len(self),
            )

            spikiness = float(np.mean(policy_maxima))

            log_scalar('dataset/policy_spikiness', spikiness)

            log_histogram('dataset/policy_targets', policy_maxima)
            log_histogram('dataset/value_targets', value_targets)

            # dataset.deduplicate()
            # log_histogram('dataset/value_targets_deduplicated', np.array(dataset.value_targets))

            # log game length and std
            log_scalars(
                'dataset/game_length',
                {
                    'mean': np.mean(accumulated_stats.game_lengths),
                    'max': np.max(accumulated_stats.game_lengths),
                    'min': np.min(accumulated_stats.game_lengths),
                    'std': np.std(accumulated_stats.game_lengths),
                },
            )

            if accumulated_stats.num_too_long_games > 0:
                log_scalar('dataset/num_too_long_games', accumulated_stats.num_too_long_games)

            log_scalar('dataset/num_games', accumulated_stats.num_games)
            log_scalar('dataset/num_samples', accumulated_stats.num_samples)
            log_scalar(
                'dataset/average_generation_time',
                accumulated_stats.total_generation_time / accumulated_stats.num_games,
            )

            if accumulated_stats.resignations > 0:
                log_scalar(
                    'resignation/average_resignation_percent',
                    accumulated_stats.resignations / accumulated_stats.num_games * 100,
                )
                # winnable resignations
                log_scalar(
                    'resignation/average_winnable_resignations',
                    (
                        accumulated_stats.num_winnable_resignations
                        / accumulated_stats.num_resignations_evaluated_to_end
                        * 100
                    )
                    if accumulated_stats.num_resignations_evaluated_to_end > 0
                    else 0,
                )
                # average moves after resignation
                log_scalar(
                    'resignation/average_moves_after_resignation',
                    accumulated_stats.num_moves_after_resignation / accumulated_stats.num_resignations_evaluated_to_end
                    if accumulated_stats.num_resignations_evaluated_to_end > 0
                    else 0,
                )

    # ---------- helpers --------------------------------------------------- #
    def _add_iteration(self, iteration: int, files: list[Path]) -> None:
        for file in files:
            assert file.exists(), f'File {file} does not exist for iteration {iteration}.'

            dataset = SelfPlayDataset.load(file)
            self._buf.append((iteration, dataset))
            self._num_samples += len(dataset)

    def _drop_left(self) -> None:
        _, old = self._buf.popleft()
        self._num_samples -= len(old)

    def _rebuild_prefix(self) -> None:
        lengths = [len(ds) for _, ds in self._buf]
        self._prefix = [0] + list(accumulate(lengths))

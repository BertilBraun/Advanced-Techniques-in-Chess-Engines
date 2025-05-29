from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from collections import deque
from itertools import accumulate
from torch.utils.data import Dataset
from torch.multiprocessing import Process

from src.mcts.MCTS import action_probabilities
from src.settings import log_histogram, log_scalar
from src.util.tensorboard import TensorboardWriter
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats


MAX_BUFFER_SAMPLES = 400_000  # hard ceiling


class RollingSelfPlayBuffer(Dataset[tuple[torch.Tensor, torch.Tensor, float]]):
    """
    Keeps a sliding window of recent SelfPlayDataset objects in RAM.
    • The window size in *iterations* comes from  args.training.sampling_window(i)
    • The window size in *samples* is never allowed to exceed MAX_BUFFER_SAMPLES
    """

    def __init__(self) -> None:
        self._buf: deque[tuple[int, SelfPlayDataset]] = deque()  # (iteration, dataset)
        self._prefix: list[int] = [0]  # len prefix sums
        self._num_samples = 0

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
        while self._num_samples > MAX_BUFFER_SAMPLES and len(self._buf) > 1:
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

    def __getitem__(self, idx: int):
        ds_idx = np.searchsorted(self._prefix, idx + 1) - 1
        inner = idx - self._prefix[ds_idx]
        return self._buf[ds_idx][1][inner]  # (state, π-target, v-target)

    def log_all_dataset_stats(self, run: int) -> None:
        Process(target=self._log_all_dataset_stats, args=(run,), daemon=True).start()

    def _log_all_dataset_stats(self, run: int) -> None:
        accumulated_stats = self.stats

        with TensorboardWriter(run, 'dataset', postfix_pid=False):
            policies = [action_probabilities(visits) for _, dataset in self._buf for visits in dataset.visit_counts]

            spikiness = sum(policy.max() for policy in policies) / len(self)

            log_scalar('dataset/policy_spikiness', spikiness)

            log_histogram('dataset/policy_targets', np.array([policy.max() for policy in policies]))
            log_histogram(
                'dataset/value_targets',
                np.array([value for _, dataset in self._buf for value in dataset.value_targets]),
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

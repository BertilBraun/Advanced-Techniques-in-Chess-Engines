from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.multiprocessing import Process

from src.Encoding import decode_board_state
from src.mcts.MCTS import action_probabilities
from src.settings import TORCH_DTYPE, log_histogram, log_scalar
from src.util.tensorboard import TensorboardWriter
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats


class SelfPlayTrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset to train the neural network on self-play data. It is a wrapper around multiple SelfPlayDatasets (i.e. Iterations). The Idea is, to load only chunks of the datasets into memory and return the next sample from the next dataset in a round-robin fashion."""

    def __init__(self, run: int, device: torch.device) -> None:
        self.run = run
        self.device = device

        self.all_chunks: list[list[Path]] = []

        self.stats = SelfPlayDatasetStats()

        self.sample_index: list[int] = []

        self.active_states: list[torch.Tensor] = []
        self.active_policies: list[torch.Tensor] = []
        self.active_values: list[torch.Tensor] = []

    def load_from_files(self, folder_path: str, origins: list[tuple[int, list[Path]]]) -> None:
        self.all_chunks = [[] for _ in range(len(origins))]
        self.sample_index = [0 for _ in range(len(origins))]
        self.active_states = [torch.zeros(0) for _ in range(len(origins))]
        self.active_policies = [torch.zeros(0) for _ in range(len(origins))]
        self.active_values = [torch.zeros(0) for _ in range(len(origins))]

        self.stats = SelfPlayDatasetStats()

        for i, (iteration, sublist) in enumerate(origins):
            iteration_dataset = SelfPlayDataset()
            for file in sublist:
                processed_indicator = file.parent / (file.stem + '.processed')
                if processed_indicator.exists():
                    continue
                iteration_dataset += SelfPlayDataset.load(file)
                processed_indicator.touch()

            iteration_dataset.shuffle().chunked_save(folder_path + '/shuffled', iteration, 5000)

            self.all_chunks[i] = SelfPlayDataset.get_files_to_load_for_iteration(folder_path + '/shuffled', iteration)

            for file in self.all_chunks[i]:
                self.stats += SelfPlayDataset.load_stats(file)

        thread = Process(
            target=self._log_all_dataset_stats,
            args=(self.all_chunks, self.run),
            daemon=True,
        )
        thread.start()

    @staticmethod
    def _log_all_dataset_stats(iterations: list[list[Path]], run: int) -> None:
        accumulated_stats = SelfPlayDatasetStats()

        with TensorboardWriter(run, 'dataset', postfix_pid=False):
            for files in iterations:
                for file in files:
                    dataset = SelfPlayDataset.load(file)
                    if len(dataset) == 0:
                        continue

                    accumulated_stats += dataset.stats

                    policies = [action_probabilities(visits) for visits in dataset.visit_counts]

                    spikiness = sum(policy.max() for policy in policies) / len(dataset)

                    log_scalar('dataset/policy_spikiness', spikiness)

                    log_histogram('dataset/policy_targets', np.array([policy.max() for policy in policies]))
                    log_histogram('dataset/value_targets', np.array(dataset.value_targets))

                    # dataset.deduplicate()
                    # log_histogram('dataset/value_targets_deduplicated', np.array(dataset.value_targets))

            log_scalar('dataset/num_games', accumulated_stats.num_games)
            log_scalar('dataset/num_resignations', accumulated_stats.resignations)
            log_scalar(
                'dataset/average_resignation_percent',
                accumulated_stats.resignations / accumulated_stats.num_games * 100,
            )
            log_scalar('dataset/num_samples', accumulated_stats.num_samples)
            log_scalar(
                'dataset/average_generation_time', accumulated_stats.total_generation_time / accumulated_stats.num_games
            )

    def as_dataloader(self, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
        assert num_workers > 0, 'num_workers must be greater than 0'
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=1024,
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = idx % len(self.all_chunks)

        if self.sample_index[index] >= len(self.active_states[index]):
            self.active_states[index], self.active_policies[index], self.active_values[index] = self._load_samples(
                index
            )

            self.sample_index[index] = 0

        state = self.active_states[index][self.sample_index[index]]
        policy_target = self.active_policies[index][self.sample_index[index]]
        value_target = self.active_values[index][self.sample_index[index]]
        self.sample_index[index] += 1

        return state, policy_target, value_target

    def __len__(self) -> int:
        return self.stats.num_samples

    def _load_samples(self, iteration: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_to_load = self.all_chunks[iteration].pop(0)
        self.all_chunks[iteration].append(chunk_to_load)

        dataset = SelfPlayDataset.load(chunk_to_load)

        states_np = np.array([decode_board_state(state) for state in dataset.encoded_states], dtype=np.float32)
        policy_targets_np = np.array(
            [action_probabilities(visit_counts) for visit_counts in dataset.visit_counts], dtype=np.float32
        )
        value_targets_np = np.array(dataset.value_targets, dtype=np.float32)

        states_torch = torch.from_numpy(states_np).to(device=self.device, dtype=TORCH_DTYPE, non_blocking=True)
        policies_torch = torch.from_numpy(policy_targets_np).to(
            device=self.device, dtype=TORCH_DTYPE, non_blocking=True
        )
        values_torch = torch.tensor(value_targets_np).to(device=self.device, dtype=TORCH_DTYPE, non_blocking=True)

        return states_torch, policies_torch, values_torch

from __future__ import annotations

import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from src.Encoding import decode_board_state
from src.mcts.MCTS import action_probabilities
from src.settings import TORCH_DTYPE, log_histogram, log_scalar
from src.util import random_id
from src.util.log import log
from src.util.tensorboard import TensorboardWriter
from src.util.timing import timeit
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats


class SelfPlayTrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset to train the neural network on self-play data. It is a wrapper around multiple SelfPlayDatasets (i.e. Iterations). The Idea is, to load only chunks of the datasets into memory and return the next sample from the next dataset in a round-robin fashion."""

    def __init__(self, run: int, chunk_size: int, device: torch.device) -> None:
        self.run = run
        self.chunk_size = chunk_size
        self.device = device

        self.all_chunks: list[Path] = []

        self.stats = SelfPlayDatasetStats()
        self.accumulated_stats = SelfPlayDatasetStats()

        self.sample_index = 0
        self.active_states = torch.zeros(0)
        self.active_policies = torch.zeros(0)
        self.active_values = torch.zeros(0)

    def load_from_files(self, folder_path: str, origins: list[list[Path]]) -> None:
        self.all_chunks = []
        self.stats = SelfPlayDatasetStats()

        for i, sublist in enumerate(origins):
            for j, file in enumerate(sublist):
                # keep the original file
                # copy the file to the folder_path
                chunk_file = Path(folder_path) / f'iteration_{i}/chunk_{j}_{random_id()}.hdf5'
                chunk_file.parent.mkdir(parents=True, exist_ok=True)
                chunk_file.hardlink_to(file)
                self.all_chunks.append(chunk_file)
                self.stats += SelfPlayDataset.load_stats(file)

        random.shuffle(self.all_chunks)
        return

        for i, files in enumerate(origins):
            if len(files) == 0:
                continue

            dataset = SelfPlayDataset()
            for file in files:
                dataset += SelfPlayDataset.load(file)

            self._log_dataset_stats(dataset)

            if len(files) > 1:
                dataset.deduplicate()
                # Remove the original files to avoid re-deduplication
                for file in files:
                    file.unlink()
                dataset.save(folder_path, i, suffix='deduplicated')

            dataset = dataset.shuffle()

            log_scalar('dataset/num_deduplicated_samples', len(dataset), i)

            self.stats += dataset.stats

            # write them in chunks to file, then simply keep a list of all chunks, shuffle them
            # Then load them in order, always 3 at a time, shuffling the values of these three chunks in memory and repeating once all values of these 3 chunks are used

            # split dataset into chunks of chunk_size
            for i in range(0, len(dataset), self.chunk_size):
                chunk = SelfPlayDataset()
                chunk.states = dataset.states[i : i + self.chunk_size]
                chunk.policy_targets = dataset.policy_targets[i : i + self.chunk_size]
                chunk.value_targets = dataset.value_targets[i : i + self.chunk_size]
                # Save the chunks to a different folder, to avoid mixing them with the original dataset
                save_file = chunk.save(folder_path + f'/iteration_{i}', i, suffix=f'chunk_{i // self.chunk_size}')
                self.all_chunks.append(save_file)

            del dataset

        random.shuffle(self.all_chunks)
        log(f'Loaded {len(self.all_chunks)} chunks with:\n{self.stats}')

    def _log_dataset_stats(self, dataset: SelfPlayDataset) -> None:
        self.accumulated_stats += dataset.stats

        policies = [action_probabilities(visits) for visits in dataset.visit_counts]

        spikiness = sum(policy.max() for policy in policies) / len(dataset)

        with TensorboardWriter(self.run, 'dataset', postfix_pid=False):
            log_scalar('dataset/num_games', self.accumulated_stats.num_games)
            log_scalar('dataset/num_resignations', self.accumulated_stats.resignations)
            log_scalar(
                'dataset/average_resignation_percent',
                self.accumulated_stats.resignations / self.accumulated_stats.num_games * 100,
            )
            log_scalar('dataset/num_samples', self.accumulated_stats.num_samples)
            log_scalar(
                'dataset/average_generation_time',
                self.accumulated_stats.total_generation_time / self.accumulated_stats.num_games,
            )
            log_scalar('dataset/policy_spikiness', spikiness)

            log_histogram('dataset/policy_targets', np.array([policy.max() for policy in policies]))
            log_histogram('dataset/value_targets', np.array(dataset.value_targets))

            dataset.deduplicate()
            log_histogram('dataset/value_targets_deduplicated', np.array(dataset.value_targets))

    def as_dataloader(self, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=100,
        )

    def cleanup(self) -> None:
        for chunk in self.all_chunks:
            chunk.unlink()

        # remove the folder with the chunks
        for chunk in self.all_chunks:
            try:
                chunk.parent.rmdir()
            except Exception:
                pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sample_index >= len(self.active_states):
            self.active_states, self.active_policies, self.active_values = self._load_samples()
            self.sample_index = 0

        state = self.active_states[self.sample_index]
        policy_target = self.active_policies[self.sample_index]
        value_target = self.active_values[self.sample_index]
        self.sample_index += 1

        return state, policy_target, value_target

    def __len__(self) -> int:
        return self.stats.num_samples

    @timeit
    def _load_samples(self, num_chunks: int = 50) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # load them in order, always num_chunks at a time, shuffling the values of these three chunks in memory and repeating once all values of these num_chunks chunks are used
        chunks_to_load = self.all_chunks[:num_chunks]
        self.all_chunks = self.all_chunks[num_chunks:] + chunks_to_load

        states: list[np.ndarray] = []
        visit_counts: list[list[tuple[int, int]]] = []
        value_targets: list[float] = []

        for chunk in chunks_to_load:
            dataset = SelfPlayDataset.load(chunk)
            self._log_dataset_stats(dataset)
            states += dataset.encoded_states
            visit_counts += dataset.visit_counts
            value_targets += dataset.value_targets

        indices = np.arange(len(states))
        np.random.shuffle(indices)

        states_np = np.array([decode_board_state(states[i]) for i in indices], dtype=np.float32)
        policy_targets_np = np.array([action_probabilities(visit_counts[i]) for i in indices], dtype=np.float32)
        value_targets_np = np.array([value_targets[i] for i in indices], dtype=np.float32)

        states_torch = torch.from_numpy(states_np).to(device=self.device, dtype=TORCH_DTYPE, non_blocking=True)
        policies_torch = torch.from_numpy(policy_targets_np).to(
            device=self.device, dtype=TORCH_DTYPE, non_blocking=True
        )
        values_torch = torch.tensor(value_targets_np, dtype=TORCH_DTYPE, device=self.device)

        return states_torch, policies_torch, values_torch

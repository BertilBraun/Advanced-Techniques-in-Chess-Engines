from __future__ import annotations

import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from src.Encoding import decode_board_state
from src.settings import TORCH_DTYPE, log_histogram, log_scalar
from src.util.log import log
from src.util.timing import timeit
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats


class SelfPlayTrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset to train the neural network on self-play data. It is a wrapper around multiple SelfPlayDatasets (i.e. Iterations). The Idea is, to load only chunks of the datasets into memory and return the next sample from the next dataset in a round-robin fashion."""

    def __init__(self, chunk_size: int, device: torch.device) -> None:
        self.chunk_size = chunk_size
        self.device = device

        self.all_chunks: list[Path] = []

        self.stats = SelfPlayDatasetStats()

        self.sample_index = 0
        self.active_states = torch.zeros(0)
        self.active_policies = torch.zeros(0)
        self.active_values = torch.zeros(0)

    def load_from_files(self, folder_path: str, origins: list[list[Path]]) -> None:
        for i, files in enumerate(origins):
            if len(files) == 0:
                continue

            dataset = SelfPlayDataset()
            for file in files:
                dataset += SelfPlayDataset.load(file)

            log_scalar('num_games', dataset.stats.num_games, i)
            log_scalar('num_resignations', dataset.stats.resignations, i)
            log_scalar('average_resignations_percent', dataset.stats.resignations / dataset.stats.num_games * 100, i)
            log_scalar('num_samples', len(dataset), i)
            log_scalar('total_generation_time', dataset.stats.total_generation_time, i)
            log_scalar('average_generation_time', dataset.stats.total_generation_time / dataset.stats.num_games, i)

            if len(files) > 1:
                dataset.deduplicate()
                # Remove the original files to avoid re-deduplication
                for file in files:
                    file.unlink()
                dataset.save(folder_path, i, suffix='deduplicated')

            dataset = dataset.shuffle()

            log_scalar('num_deduplicated_samples', len(dataset), i)

            spikiness = sum(policy_target.max() for policy_target in dataset.policy_targets) / len(dataset)
            log_scalar('policy_spikiness', spikiness, i)

            log_histogram('policy_targets', np.array(dataset.policy_targets), i)
            log_histogram('value_targets', np.array(dataset.value_targets), i)

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

    def as_dataloader(self, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
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

    @timeit
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
    def _load_samples(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # load them in order, always 3 at a time, shuffling the values of these three chunks in memory and repeating once all values of these 3 chunks are used
        chunks_to_load = self.all_chunks[:3]
        self.all_chunks = self.all_chunks[3:] + chunks_to_load

        states: list[np.ndarray] = []
        policy_targets: list[np.ndarray] = []
        value_targets: list[float] = []

        for chunk in chunks_to_load:
            dataset = SelfPlayDataset.load(chunk)
            states += dataset.states
            policy_targets += dataset.policy_targets
            value_targets += dataset.value_targets

        indices = np.arange(len(states))
        np.random.shuffle(indices)

        states_np = np.array([decode_board_state(states[i]) for i in indices], dtype=np.float32)
        policy_targets_np = np.array([policy_targets[i] for i in indices], dtype=np.float32)
        value_targets_np = np.array([value_targets[i] for i in indices], dtype=np.float32)

        states_torch = torch.from_numpy(states_np).to(device=self.device, dtype=TORCH_DTYPE, non_blocking=True)
        policies_torch = torch.from_numpy(policy_targets_np).to(
            device=self.device, dtype=TORCH_DTYPE, non_blocking=True
        )
        values_torch = torch.tensor(value_targets_np, dtype=TORCH_DTYPE, device=self.device)

        return states_torch, policies_torch, values_torch

from __future__ import annotations

import h5py
import torch
import random
import numpy as np
import numpy.typing as npt
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple
from torch.utils.data import Dataset

from src.Encoding import decode_board_state, encode_board_state
from src.settings import TORCH_DTYPE, CurrentGame, log_histogram, log_scalar
from src.util.log import log
from src.util import random_id


class SelfPlayDatasetStats(NamedTuple):
    num_samples: int
    num_games: int
    total_generation_time: float

    def __repr__(self) -> str:
        return f"""Num samples: {self.num_samples}
Num games: {self.num_games}
Total generation time: {self.total_generation_time:.2f}s
Average generation time: {self.total_generation_time / self.num_games:.2f}s/game"""

    def __add__(self, other: SelfPlayDatasetStats) -> SelfPlayDatasetStats:
        return SelfPlayDatasetStats(
            num_samples=self.num_samples + other.num_samples,
            num_games=self.num_games + other.num_games,
            total_generation_time=self.total_generation_time + other.total_generation_time,
        )


class SelfPlayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, float]]):
    """Each sample is represented by:
    state: torch.Tensor
    policy_targets: torch.Tensor
    value_target: float

    For efficiency, we store the states, policy targets and value targets in separate lists.

    We need functionality to:
    - Add a new sample
    - Get a sample by index
    - Get the number of samples
    - Deduplicate the samples
    - Load the samples from a file
    - Save the samples to a file
    """

    def __init__(self) -> None:
        self.states: list[npt.NDArray[np.uint64]] = []
        self.policy_targets: list[npt.NDArray[np.float32]] = []
        self.value_targets: list[float] = []
        self.stats = SelfPlayDatasetStats(0, 0, 0)

    def add_generation_stats(self, num_games: int, generation_time: float) -> None:
        self.stats += SelfPlayDatasetStats(0, num_games, generation_time)

    def add_sample(
        self, state: npt.NDArray[np.int8], policy_target: npt.NDArray[np.float32], value_target: float
    ) -> None:
        self.states.append(encode_board_state(state))
        self.policy_targets.append(policy_target)
        self.value_targets.append(value_target)
        self.stats += SelfPlayDatasetStats(1, 0, 0)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(decode_board_state(self.states[idx])),
            torch.from_numpy(self.policy_targets[idx]),
            torch.tensor(self.value_targets[idx], dtype=torch.float32),
        )

    def __add__(self, other: SelfPlayDataset) -> SelfPlayDataset:
        new_dataset = SelfPlayDataset()
        new_dataset.states = self.states + other.states
        new_dataset.policy_targets = self.policy_targets + other.policy_targets
        new_dataset.value_targets = self.value_targets + other.value_targets
        new_dataset.stats = self.stats + other.stats
        return new_dataset

    def deduplicate(self) -> None:
        """Deduplicate the data based on the board state by averaging the policy and value targets"""
        mp: dict[tuple[int, ...], tuple[int, tuple[npt.NDArray[np.float32], float]]] = {}

        for state, policy_target, value_target in zip(self.states, self.policy_targets, self.value_targets):
            state = tuple(state)
            if state in mp:
                count, (policy_target_sum, value_target_sum) = mp[state]
                mp[state] = (
                    count + 1,
                    (policy_target_sum + policy_target, value_target_sum + value_target),
                )
            else:
                mp[state] = (
                    1,
                    (policy_target, value_target),
                )

        self.states = [np.array(state) for state in mp.keys()]
        self.policy_targets = [policy_target / count for count, (policy_target, _) in mp.values()]
        self.value_targets = [value_target / count for count, (_, value_target) in mp.values()]

    def shuffle(self) -> SelfPlayDataset:
        indices = np.arange(len(self))
        np.random.shuffle(indices)

        shuffled_dataset = SelfPlayDataset()
        shuffled_dataset.states = [self.states[i] for i in indices]
        shuffled_dataset.policy_targets = [self.policy_targets[i] for i in indices]
        shuffled_dataset.value_targets = [self.value_targets[i] for i in indices]
        shuffled_dataset.stats = self.stats
        return shuffled_dataset

    @staticmethod
    def load(file_path: Path) -> SelfPlayDataset:
        dataset = SelfPlayDataset()

        try:
            with h5py.File(file_path, 'r') as file:
                metadata: dict[str, Any] = eval(file.attrs['metadata'])  # type: ignore
                message = f'Invalid metadata. Expected {SelfPlayDataset._get_current_metadata()}, got {metadata}'
                assert metadata == SelfPlayDataset._get_current_metadata(), message

                dataset.states = [state for state in file['states']]  # type: ignore
                dataset.policy_targets = [policy_target for policy_target in file['policy_targets']]  # type: ignore
                dataset.value_targets = [value_target for value_target in file['value_targets']]  # type: ignore

                stats: dict[str, Any] = eval(file.attrs['stats'])  # type: ignore
                dataset.stats = SelfPlayDatasetStats(**stats)
        except Exception as e:
            log(f'Error loading dataset from {file_path}: {e}')

        return dataset

    @staticmethod
    def load_iteration(folder_path: str | PathLike, iteration: int) -> SelfPlayDataset:
        dataset = SelfPlayDataset()
        for file_path in SelfPlayDataset.get_files_to_load_for_iteration(folder_path, iteration):
            dataset += SelfPlayDataset.load(file_path)
        return dataset

    @staticmethod
    def get_files_to_load_for_iteration(folder_path: str | PathLike, iteration: int) -> list[Path]:
        return [file_path for file_path in Path(folder_path).glob(f'memory_{iteration}_*.hdf5')]

    def save(self, folder_path: str | PathLike, iteration: int, suffix: str | None = None) -> Path:
        if not suffix:
            suffix = random_id()
        file_path = Path(folder_path) / f'memory_{iteration}_{suffix}.hdf5'
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # write a h5py file with states, policy targets and value targets in it
        with h5py.File(file_path, 'w') as file:
            file.create_dataset('states', data=np.array(self.states))
            file.create_dataset('policy_targets', data=np.array(self.policy_targets))
            file.create_dataset('value_targets', data=np.array(self.value_targets))
            # write the metadata information about the current game, action size, representation shape, etc.
            file.attrs['metadata'] = str(SelfPlayDataset._get_current_metadata())
            # write the stats information about the dataset, num_games, total_generation_time
            file.attrs['stats'] = str(self.stats._asdict())

        return file_path

    @staticmethod
    def _get_current_metadata() -> dict[str, Any]:
        # metadata information about current game, action size, representation shape, etc.
        return {
            'action_size': str(CurrentGame.action_size),
            'representation_shape': str(CurrentGame.representation_shape),
            'game': CurrentGame.__class__.__name__,
        }


class SelfPlayTrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset to train the neural network on self-play data. It is a wrapper around multiple SelfPlayDatasets (i.e. Iterations). The Idea is, to load only chunks of the datasets into memory and return the next sample from the next dataset in a round-robin fashion."""

    def __init__(self, iterations: list[int], folder_path: str, chunk_size: int, device: torch.device) -> None:
        self.iterations = iterations
        self.folder_path = folder_path
        self.chunk_size = chunk_size
        self.device = device

        self.all_chunks: list[Path] = []

        self.stats = SelfPlayDatasetStats(0, 0, 0)

        for iteration in self.iterations:
            files_for_iteration = SelfPlayDataset.get_files_to_load_for_iteration(folder_path, iteration)
            if len(files_for_iteration) == 0:
                continue

            dataset = SelfPlayDataset.load_iteration(folder_path, iteration)
            log_scalar('num_games', dataset.stats.num_games, iteration)
            log_scalar('num_samples', len(dataset), iteration)
            log_scalar('total_generation_time', dataset.stats.total_generation_time, iteration)
            log_scalar(
                'average_generation_time', dataset.stats.total_generation_time / dataset.stats.num_games, iteration
            )

            if len(files_for_iteration) > 1:
                dataset.deduplicate()
                dataset.save(folder_path, iteration, suffix='deduplicated')
                # Remove the original files to avoid re-deduplication
                for file in files_for_iteration:
                    file.unlink()

            dataset = dataset.shuffle()

            log_scalar('num_deduplicated_samples', len(dataset), iteration)

            spikiness = sum(policy_target.max() for policy_target in dataset.policy_targets) / len(dataset)
            log_scalar('policy_spikiness', spikiness, iteration)

            log_histogram('policy_targets', np.array(dataset.policy_targets), iteration)
            log_histogram('value_targets', np.array(dataset.value_targets), iteration)

            self.stats += dataset.stats

            # write them in chunks to file, then simply keep a list of all chunks, shuffle them
            # Then load them in order, always 3 at a time, shuffling the values of these three chunks in memory and repeating once all values of these 3 chunks are used

            # split dataset into chunks of chunk_size
            for i in range(0, len(dataset), chunk_size):
                chunk = SelfPlayDataset()
                chunk.states = dataset.states[i : i + chunk_size]
                chunk.policy_targets = dataset.policy_targets[i : i + chunk_size]
                chunk.value_targets = dataset.value_targets[i : i + chunk_size]
                # Save the chunks to a different folder, to avoid mixing them with the original dataset
                save_file = chunk.save(
                    folder_path + f'/iteration_{iteration}', iteration, suffix=f'chunk_{i // chunk_size}'
                )
                self.all_chunks.append(save_file)

            del dataset

        random.shuffle(self.all_chunks)
        self.sample_index = 0
        self.active_states, self.active_policies, self.active_values = self._load_samples()
        self.total_num_samples = 0

        log(f'Loaded {len(self.all_chunks)} chunks with:\n{self.stats}')

    def cleanup(self) -> None:
        for chunk in self.all_chunks:
            chunk.unlink()

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

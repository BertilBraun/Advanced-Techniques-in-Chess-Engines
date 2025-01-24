from __future__ import annotations

import h5py
import torch
import numpy as np
import numpy.typing as npt
from os import PathLike
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset

from src.Encoding import decode_board_state, encode_board_state
from src.settings import TORCH_DTYPE, CurrentGame
from src.util import random_id
from src.util.timing import timeit
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats


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
        self.stats = SelfPlayDatasetStats()

    def add_generation_stats(self, num_games: int, generation_time: float, resignation: bool) -> None:
        self.stats += SelfPlayDatasetStats(
            num_games=num_games,
            total_generation_time=generation_time,
            resignations=int(resignation),
        )

    def add_sample(
        self, state: npt.NDArray[np.int8], policy_target: npt.NDArray[np.float32], value_target: float
    ) -> None:
        self.states.append(encode_board_state(state))
        self.policy_targets.append(policy_target)
        self.value_targets.append(value_target)
        self.stats += SelfPlayDatasetStats(num_samples=1)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(decode_board_state(self.states[idx])).to(dtype=TORCH_DTYPE, non_blocking=True),
            torch.from_numpy(self.policy_targets[idx]).to(dtype=TORCH_DTYPE, non_blocking=True),
            torch.tensor(self.value_targets[idx], dtype=torch.float32),
        )

    def __add__(self, other: SelfPlayDataset) -> SelfPlayDataset:
        new_dataset = SelfPlayDataset()
        new_dataset.states = self.states + other.states
        new_dataset.policy_targets = self.policy_targets + other.policy_targets
        new_dataset.value_targets = self.value_targets + other.value_targets
        new_dataset.stats = self.stats + other.stats
        return new_dataset

    @timeit
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

    @timeit
    def shuffle(self) -> SelfPlayDataset:
        indices = np.arange(len(self))
        np.random.shuffle(indices)

        shuffled_dataset = SelfPlayDataset()
        shuffled_dataset.states = [self.states[i] for i in indices]
        shuffled_dataset.policy_targets = [self.policy_targets[i] for i in indices]
        shuffled_dataset.value_targets = [self.value_targets[i] for i in indices]
        shuffled_dataset.stats = self.stats
        return shuffled_dataset

    @timeit
    @staticmethod
    def load(file_path: str | PathLike) -> SelfPlayDataset:
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
            from src.util.log import log, LogLevel

            log(f'Error loading dataset from {file_path}: {e}', level=LogLevel.WARNING)

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
        if suffix:
            file_path = Path(folder_path) / f'memory_{iteration}_{suffix}.hdf5'
        else:
            while True:
                file_path = Path(folder_path) / f'memory_{iteration}_{random_id()}.hdf5'
                if not file_path.exists():
                    break
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

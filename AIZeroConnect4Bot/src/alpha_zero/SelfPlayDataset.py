from __future__ import annotations

import h5py
import torch
import numpy as np
from os import PathLike
from pathlib import Path
from torch.utils.data import Dataset

from src.Encoding import decode_board_state, encode_board_state
from src.settings import CurrentGame
from src.util import random_id


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
        self.states: list[np.ndarray] = []
        self.policy_targets: list[np.ndarray] = []
        self.value_targets: list[float] = []

    def add_sample(self, state: np.ndarray, policy_target: np.ndarray, value_target: float) -> None:
        self.states.append(encode_board_state(state))
        self.policy_targets.append(policy_target)
        self.value_targets.append(value_target)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < len(self.states):
            decoded_state = decode_board_state(self.states[idx])
            return (
                torch.from_numpy(decoded_state),
                torch.from_numpy(self.policy_targets[idx]),
                torch.tensor(self.value_targets[idx], dtype=torch.float32),
            )

        idx -= len(self.states)
        decoded_state = decode_board_state(self.states[idx])
        return (
            torch.from_numpy(decoded_state),
            torch.from_numpy(self.policy_targets[idx]),
            torch.tensor(self.value_targets[idx], dtype=torch.float32),
        )

    def __add__(self, other: SelfPlayDataset) -> SelfPlayDataset:
        new_dataset = SelfPlayDataset()
        new_dataset.states = self.states + other.states
        new_dataset.policy_targets = self.policy_targets + other.policy_targets
        new_dataset.value_targets = self.value_targets + other.value_targets
        return new_dataset

    def deduplicate(self) -> None:
        """Deduplicate the data based on the board state by averaging the policy and value targets"""

        mp: dict[tuple[int, ...], tuple[int, tuple[np.ndarray, float]]] = {}

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

    @staticmethod
    def load(file_path: str | PathLike) -> SelfPlayDataset:
        with h5py.File(file_path, 'r') as file:
            metadata = dict(file.attrs)
            message = f'Invalid metadata. Expected {SelfPlayDataset._get_current_metadata()}, got {metadata}'
            assert metadata == SelfPlayDataset._get_current_metadata(), message

            dataset = SelfPlayDataset()
            dataset.states = [state for state in file['states']]  # type: ignore
            dataset.policy_targets = [policy_target for policy_target in file['policy_targets']]  # type: ignore
            dataset.value_targets = [value_target for value_target in file['value_targets']]  # type: ignore

        return dataset

    @staticmethod
    def get_files_to_load_for_iteration(folder_path: str | PathLike, iteration: int) -> list[str]:
        return [str(file_path) for file_path in Path(folder_path).glob(f'memory_{iteration}_*.hdf5')]

    @staticmethod
    def load_iteration(folder_path: str | PathLike, iteration: int) -> SelfPlayDataset:
        dataset = SelfPlayDataset()
        for file_path in SelfPlayDataset.get_files_to_load_for_iteration(folder_path, iteration):
            dataset += SelfPlayDataset.load(file_path)

        return dataset

    def save(self, folder_path: str | PathLike, iteration: int, suffix: str | None = None) -> None:
        if not suffix:
            suffix = random_id()
        file_path = Path(folder_path) / f'memory_{iteration}_{suffix}.hdf5'

        # write a h5py file with states, policy targets and value targets in it
        with h5py.File(file_path, 'w') as file:
            file.create_dataset('states', data=np.array(self.states))
            file.create_dataset('policy_targets', data=np.array(self.policy_targets))
            file.create_dataset('value_targets', data=np.array(self.value_targets))
            # write the metadata information about the current game, action size, representation shape, etc.
            file.attrs.update(SelfPlayDataset._get_current_metadata())

    @staticmethod
    def _get_current_metadata() -> dict[str, str]:
        # metadata information about current game, action size, representation shape, etc.
        return {
            'action_size': str(CurrentGame.action_size),
            'representation_shape': str(CurrentGame.representation_shape),
            'game': CurrentGame.__class__.__name__,
        }

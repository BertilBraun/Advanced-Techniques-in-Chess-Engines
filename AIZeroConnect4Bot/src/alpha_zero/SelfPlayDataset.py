from __future__ import annotations

import h5py
import torch
import numpy as np
from os import PathLike
from pathlib import Path
from torch.utils.data import Dataset

from src.settings import VALIDATE_DATASET, CurrentGame
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

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.states: list[np.ndarray] = []
        self.policy_targets: np.ndarray = np.empty((0, CurrentGame.action_size))
        self.value_targets: np.ndarray = np.empty(0)
        self.additional_policy_targets: list[np.ndarray] = []
        self.additional_value_targets: list[float] = []

    def add_sample(self, state: np.ndarray, policy_target: np.ndarray, value_target: float) -> None:
        self.states.append(self._encode_state(state))
        self.additional_policy_targets.append(policy_target)
        self.additional_value_targets.append(value_target)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        if idx < len(self.states):
            return (
                self._decode_state(self.states[idx]).to(self.device),
                torch.from_numpy(self.policy_targets[idx]).to(self.device),
                self.value_targets[idx],
            )

        idx -= len(self.states)
        return (
            self._decode_state(self.states[idx]).to(self.device),
            torch.from_numpy(self.additional_policy_targets[idx]).to(self.device),
            self.additional_value_targets[idx],
        )

    def __add__(self, other: SelfPlayDataset) -> SelfPlayDataset:
        self.collect()
        other.collect()

        new_dataset = SelfPlayDataset(self.device)
        new_dataset.states = self.states + other.states
        new_dataset.policy_targets = np.concatenate([self.policy_targets, other.policy_targets])
        new_dataset.value_targets = np.concatenate([self.value_targets, other.value_targets])
        return new_dataset

    def deduplicate(self) -> None:
        """Deduplicate the data based on the board state by averaging the policy and value targets"""
        assert (
            not self.additional_policy_targets and not self.additional_value_targets
        ), 'Additional data must be empty. Call collect() before deduplicating'

        mp: dict[tuple[int, ...], tuple[int, tuple[torch.Tensor, float]]] = {}
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

        self.additional_policy_targets = []
        self.additional_value_targets = []

        self.states = [np.array(state) for state in mp.keys()]
        self.policy_targets = np.stack([policy_target / count for count, (policy_target, _) in mp.values()])
        self.value_targets = np.array([value_target / count for count, (_, value_target) in mp.values()])

    @staticmethod
    def load(file_path: str | PathLike, device: torch.device) -> SelfPlayDataset:
        with h5py.File(file_path, 'r') as file:
            if VALIDATE_DATASET:
                metadata = dict(file.attrs)
                if metadata != SelfPlayDataset._get_current_metadata():
                    message = f'Invalid metadata. Expected {SelfPlayDataset._get_current_metadata()}, got {metadata}'
                    assert False, message

            dataset = SelfPlayDataset(device)
            dataset.states = [state for state in file['states']]  # type: ignore
            dataset.policy_targets = file['policy_targets'][:]  # type: ignore
            dataset.value_targets = file['value_targets'][:]  # type: ignore

        return dataset

    @staticmethod
    def get_files_to_load_for_iteration(folder_path: str | PathLike, iteration: int) -> list[str]:
        return [str(file_path) for file_path in Path(folder_path).glob(f'memory_{iteration}_*.hdf5')]

    @staticmethod
    def load_iteration(folder_path: str | PathLike, iteration: int, device: torch.device) -> SelfPlayDataset:
        dataset = SelfPlayDataset(device)
        for file_path in SelfPlayDataset.get_files_to_load_for_iteration(folder_path, iteration):
            dataset += SelfPlayDataset.load(file_path, device)

        return dataset

    def collect(self) -> None:
        """Collect the additional data into the main data tensors."""
        if not self.additional_policy_targets or not self.additional_value_targets:
            return

        policy_targets = np.stack(self.additional_policy_targets)
        self.policy_targets = np.concatenate([self.policy_targets, policy_targets])

        value_targets = np.array(self.additional_value_targets)
        self.value_targets = np.concatenate([self.value_targets, value_targets])

        self.additional_policy_targets = []
        self.additional_value_targets = []

    def save(self, folder_path: str | PathLike, iteration: int, suffix: str | None = None) -> None:
        self.collect()

        if not suffix:
            suffix = random_id()
        file_path = Path(folder_path) / f'memory_{iteration}_{suffix}.hdf5'

        # write a h5py file with states, policy targets and value targets in it
        with h5py.File(file_path, 'w') as file:
            file.create_dataset('states', data=np.array(self.states))
            file.create_dataset('policy_targets', data=self.policy_targets)
            file.create_dataset('value_targets', data=self.value_targets)
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

    def _encode_state(self, state: np.ndarray) -> np.ndarray:
        """Encode the state into a tuple of integers. Each integer represents a channel of the state. This assumes that the state is a binary state.

        The encoding is done by setting the i-th bit of the integer to the i-th bit of the flattened state.
        For example, if the state is:
        [[1, 0],
         [0, 1]]
        The encoding would be:
        >>> 1001
        """
        channels, height, width = state.shape
        n_bits = height * width
        assert n_bits <= 64, 'The state is too large to encode'

        # Prepare the bit masks: 1, 2, 4, ..., 2^(n_bits-1)
        bit_masks = 1 << np.arange(n_bits, dtype=np.uint64)  # Use uint64 to prevent overflow

        # Shape: (channels, height * width)
        flattened = state.reshape(channels, -1).astype(np.uint64)

        # Perform vectorized dot product to encode each channel
        encoded = (flattened * bit_masks).sum(axis=1)

        return encoded

    def _decode_state(self, state: np.ndarray) -> torch.Tensor:
        channels, height, width = CurrentGame.representation_shape
        n_bits = height * width

        # Prepare the bit masks: 1, 2, 4, ..., 2^(n_bits-1)
        bit_masks = 1 << np.arange(n_bits, dtype=np.uint64)  # Shape: (n_bits,)

        # Convert state tuple to a NumPy array for vectorized operations
        encoded_array = state.astype(np.uint64).reshape(channels, 1)

        # Extract bits for each channel
        bits = ((encoded_array & bit_masks) > 0).astype(np.int8).reshape(channels, height, width)

        return torch.from_numpy(bits).to(torch.int8)

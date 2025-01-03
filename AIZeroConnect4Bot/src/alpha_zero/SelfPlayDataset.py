from __future__ import annotations
from pathlib import Path

import torch
from os import PathLike
from torch.utils.data import Dataset

from src.settings import CurrentGame
from src.util.profiler import log_event


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
        self.additional_states: list[torch.Tensor] = []
        self.additional_policy_targets: list[torch.Tensor] = []
        self.additional_value_targets: list[float] = []
        self.states: torch.Tensor = torch.empty(0, device=device)
        self.policy_targets: torch.Tensor = torch.empty(0, device=device)
        self.value_targets: torch.Tensor = torch.empty(0, device=device)

    def add_sample(self, state: torch.Tensor, policy_target: torch.Tensor, value_target: float) -> None:
        self.additional_states.append(state.to(self.device))
        self.additional_policy_targets.append(policy_target.to(self.device))
        self.additional_value_targets.append(value_target)

    def __len__(self) -> int:
        return len(self.additional_states) + len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        if idx < len(self.states):
            return self.states[idx], self.policy_targets[idx], self.value_targets[idx]  # type: ignore

        idx -= len(self.states)
        return self.additional_states[idx], self.additional_policy_targets[idx], self.additional_value_targets[idx]

    def __add__(self, other: SelfPlayDataset) -> SelfPlayDataset:
        self.collect()
        other.collect()

        if self.device != other.device:
            print(
                f'Warning: Merging datasets with different devices: ({self.device} and {other.device}). Moving data to the device of the first dataset.'
            )

            other.states = other.states.to(self.device)
            other.policy_targets = other.policy_targets.to(self.device)
            other.value_targets = other.value_targets.to(self.device)

        new_dataset = SelfPlayDataset(self.device)
        new_dataset.states = torch.cat([self.states, other.states])
        new_dataset.policy_targets = torch.cat([self.policy_targets, other.policy_targets])
        new_dataset.value_targets = torch.cat([self.value_targets, other.value_targets])
        return new_dataset

    def deduplicate(self) -> None:
        """Deduplicate the data based on the board state by averaging the policy and value targets"""
        assert len(self.additional_states) == 0, 'Additional data must be empty. Call collect() before deduplicating'

        mp: dict[int, tuple[int, tuple[torch.Tensor, torch.Tensor, float]]] = {}
        hashes = CurrentGame.hash_boards(self.states)
        for i, h in enumerate(hashes):
            if h in mp:
                count, (state, policy_target, value_target) = mp[h]
                mp[h] = (  # type: ignore
                    count + 1,
                    (
                        state,
                        policy_target + self.policy_targets[i],
                        value_target + self.value_targets[i],
                    ),
                )
            else:
                mp[h] = (  # type: ignore
                    1,
                    (self.states[i], self.policy_targets[i], self.value_targets[i]),
                )

        self.additional_states = []
        self.additional_policy_targets = []
        self.additional_value_targets = []

        self.states = torch.stack([state for _, (state, _, _) in mp.values()])
        self.policy_targets = torch.stack([policy_target / count for count, (_, policy_target, _) in mp.values()])
        self.value_targets = torch.tensor(
            [value_target / count for count, (_, _, value_target) in mp.values()], device=self.device
        )

    @staticmethod
    def load(file_path: str | PathLike, device: torch.device) -> SelfPlayDataset:
        data: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = torch.load(
            file_path, weights_only=True, map_location=device
        )
        states, policy_targets, value_targets = data

        dataset = SelfPlayDataset(device)
        dataset.states = states
        dataset.policy_targets = policy_targets
        dataset.value_targets = value_targets
        return dataset

    @staticmethod
    def load_iteration(folder_path: str | PathLike, iteration: int, device: torch.device) -> SelfPlayDataset:
        dataset = SelfPlayDataset(device)
        for file_path in Path(folder_path).glob(f'memory_{iteration}_*.pt'):
            with log_event('dataset_loading'):
                dataset += SelfPlayDataset.load(file_path, device)

        return dataset

    def collect(self) -> None:
        """Collect the additional data into the main data tensors."""
        if len(self.additional_states) == 0:
            return

        additional_states = torch.stack(self.additional_states)
        self.states = torch.cat([self.states, additional_states])

        policy_targets = torch.stack(self.additional_policy_targets)
        self.policy_targets = torch.cat([self.policy_targets, policy_targets])

        value_targets = torch.tensor(self.additional_value_targets, device=self.device)
        self.value_targets = torch.cat([self.value_targets, value_targets])

        self.additional_states = []
        self.additional_policy_targets = []
        self.additional_value_targets = []

    def save(self, file_path: str | PathLike) -> None:
        self.collect()

        torch.save((self.states, self.policy_targets, self.value_targets), file_path)

from __future__ import annotations
from pathlib import Path

import torch
from os import PathLike
from torch.utils.data import Dataset

from src.settings import CurrentGame


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

    def __init__(self):
        self.states: list[torch.Tensor] = []
        self.policy_targets: list[torch.Tensor] = []
        self.value_targets: list[float] = []

    def add_sample(self, state: torch.Tensor, policy_target: torch.Tensor, value_target: float) -> None:
        self.states.append(state)
        self.policy_targets.append(policy_target)
        self.value_targets.append(value_target)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        return self.states[idx], self.policy_targets[idx], self.value_targets[idx]

    def __add__(self, other: SelfPlayDataset) -> SelfPlayDataset:
        new_dataset = SelfPlayDataset()
        new_dataset.states = self.states + other.states
        new_dataset.policy_targets = self.policy_targets + other.policy_targets
        new_dataset.value_targets = self.value_targets + other.value_targets
        return new_dataset

    def deduplicate(self) -> None:
        """Deduplicate the data based on the board state by averaging the policy and value targets"""
        mp: dict[int, tuple[int, tuple[torch.Tensor, torch.Tensor, float]]] = {}
        hashes = CurrentGame.hash_boards(torch.stack(self.states))
        for i, h in enumerate(hashes):
            if h in mp:
                count, (state, policy_target, value_target) = mp[h]
                mp[h] = (
                    count + 1,
                    (state, policy_target + self.policy_targets[i], value_target + self.value_targets[i]),
                )
            else:
                mp[h] = (1, (self.states[i], self.policy_targets[i], self.value_targets[i]))

        self.states = []
        self.policy_targets = []
        self.value_targets = []

        for count, (state, policy_target, value_target) in mp.values():
            self.add_sample(state, policy_target / count, value_target / count)

    @staticmethod
    def load(file_path: str | PathLike, device: torch.device | None) -> SelfPlayDataset:
        data: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = torch.load(
            file_path, weights_only=True, map_location=device
        )
        states, policy_targets, value_targets = data

        dataset = SelfPlayDataset()
        dataset.states = [state for state in states]
        dataset.policy_targets = [policy_target for policy_target in policy_targets]
        dataset.value_targets = value_targets.tolist()
        return dataset

    @staticmethod
    def load_iteration(folder_path: str | PathLike, iteration: int, device: torch.device | None) -> SelfPlayDataset:
        for file_path in Path(folder_path).glob(f'memory_{iteration}_*.pt'):
            return SelfPlayDataset.load(file_path, device)

        raise FileNotFoundError(f'No memory file found for iteration {iteration}')

    def save(self, file_path: str | PathLike) -> None:
        states = torch.stack(self.states)
        policy_targets = torch.stack(self.policy_targets)
        value_targets = torch.tensor(self.value_targets)

        torch.save((states, policy_targets, value_targets), file_path)

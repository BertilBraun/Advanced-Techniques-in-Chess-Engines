from __future__ import annotations
from collections.abc import Sequence

import h5py
import torch
import numpy as np
import numpy.typing as npt
from os import PathLike
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset

from src.Encoding import decode_board_state, decode_board_states, encode_board_state
from src.mcts.MCTS import action_probabilities
from src.settings import CurrentGame, USE_GPU
from src.util import random_id
from src.util.log import warn
from src.util.timing import timeit
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats


def training_batch_from_raw_samples(
    encoded_states: Sequence[bytes],
    visit_counts: Sequence[npt.NDArray[np.uint16]],
    value_targets: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(encoded_states)
    if len(visit_counts) != batch_size or len(value_targets) != batch_size:
        raise ValueError('Training batch inputs must contain the same number of samples.')

    states = torch.from_numpy(decode_board_states(encoded_states)).to(dtype=torch.float32)
    policies = np.zeros((batch_size, CurrentGame.action_size), dtype=np.float32)
    visit_lengths = np.fromiter((len(counts) for counts in visit_counts), dtype=np.int64, count=batch_size)
    if np.any(visit_lengths == 0):
        raise ValueError('Visit counts must not be empty.')

    concatenated_visits = np.concatenate(visit_counts)
    sample_indices = np.repeat(np.arange(batch_size), visit_lengths)
    policies[sample_indices, concatenated_visits[:, 0]] = concatenated_visits[:, 1]
    policy_totals = np.sum(policies, axis=1, keepdims=True)
    if np.any(policy_totals <= 0):
        raise ValueError('Visit counts must contain a positive total.')
    policies /= policy_totals

    return (
        states,
        torch.from_numpy(policies),
        torch.from_numpy(np.asarray(value_targets, dtype=np.float32)),
    )


TrainingBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def preserve_prebatched_samples(batch: TrainingBatch) -> TrainingBatch:
    return batch


class SelfPlayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
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
        self.encoded_states: list[bytes] = []
        self.visit_counts: list[npt.NDArray[np.uint16]] = []
        self.value_targets: list[float] = []
        self.stats = SelfPlayDatasetStats()

    def add_generation_stats(self, game_length: int, generation_time: float) -> None:
        self.stats += SelfPlayDatasetStats(
            num_games=1,
            game_lengths=[game_length],
            total_generation_time=generation_time,
        )

    def add_sample(self, state: npt.NDArray[np.int8], visit_counts: list[tuple[int, int]], value_target: float) -> None:
        assert len(visit_counts) > 0, 'Visit counts must not be empty'
        assert -1 - 1e-2 <= value_target <= 1 + 1e-2, f'Value target ({value_target}) must be in the range [-1, 1]'
        # if value target is nan or inf, we skip the sample
        if not (-1 - 1e-2 <= value_target <= 1 + 1e-2) or np.isnan(value_target) or np.isinf(value_target):
            warn('Skipping sample with invalid value target:', value_target)
            return

        self.encoded_states.append(encode_board_state(state))
        self.visit_counts.append(np.array(visit_counts, dtype=np.uint16))
        self.value_targets.append(value_target)
        self.stats += SelfPlayDatasetStats(num_samples=1)

    def __len__(self) -> int:
        return len(self.encoded_states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = decode_board_state(self.encoded_states[idx])
        probabilities = action_probabilities(self.visit_counts[idx])

        assert 1 - 1e-2 <= np.sum(probabilities) <= 1 + 1e-2, 'Probabilities must sum to 1'

        return (
            torch.from_numpy(state).to(dtype=torch.float32, non_blocking=USE_GPU),
            torch.from_numpy(probabilities).to(dtype=torch.float32, non_blocking=USE_GPU),
            torch.tensor(self.value_targets[idx], dtype=torch.float32),
        )

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return training_batch_from_raw_samples(
            [self.encoded_states[index] for index in indices],
            [self.visit_counts[index] for index in indices],
            [self.value_targets[index] for index in indices],
        )

    def raw_sample(self, idx: int) -> tuple[bytes, npt.NDArray[np.uint16], float]:
        return self.encoded_states[idx], self.visit_counts[idx], self.value_targets[idx]

    def __add__(self, other: SelfPlayDataset) -> SelfPlayDataset:
        new_dataset = SelfPlayDataset()
        new_dataset.encoded_states = self.encoded_states + other.encoded_states
        new_dataset.visit_counts = self.visit_counts + other.visit_counts
        new_dataset.value_targets = self.value_targets + other.value_targets
        new_dataset.stats = self.stats + other.stats
        return new_dataset

    @timeit
    def deduplicate(self) -> SelfPlayDataset:
        """Deduplicate the data based on the board state by averaging the policy and value targets"""
        mp: dict[bytes, tuple[int, tuple[npt.NDArray[np.uint16], float]]] = {}

        for state, visit_counts, value_target in zip(self.encoded_states, self.visit_counts, self.value_targets):
            if state in mp:
                count, (visit_count_sum, value_target_sum) = mp[state]
                new_visit_counts = []
                for move, visit_count in visit_counts:
                    # find the move in the existing visit counts and add the visit count
                    for existing_move, existing_visit_count in visit_count_sum:
                        if move == existing_move:
                            new_visit_counts.append((move, visit_count + existing_visit_count))
                            break
                    else:
                        new_visit_counts.append((move, visit_count))

                mp[state] = (
                    count + 1,
                    (np.array(new_visit_counts, dtype=np.uint16), value_target_sum + value_target),
                )
            else:
                mp[state] = (
                    1,
                    (visit_counts, value_target),
                )

        deduplicated_dataset = SelfPlayDataset()

        for state, (count, (visit_count_sum, value_target_sum)) in mp.items():
            deduplicated_dataset.encoded_states.append(state)
            deduplicated_dataset.visit_counts.append(visit_count_sum)
            deduplicated_dataset.value_targets.append(value_target_sum / count)

        deduplicated_dataset.stats = self.stats.overwrite(num_samples=len(mp))
        return deduplicated_dataset

    def shuffle(self) -> SelfPlayDataset:
        indices = np.arange(len(self))
        np.random.shuffle(indices)

        shuffled_dataset = SelfPlayDataset()
        shuffled_dataset.encoded_states = [self.encoded_states[i] for i in indices]
        shuffled_dataset.visit_counts = [self.visit_counts[i] for i in indices]
        shuffled_dataset.value_targets = [self.value_targets[i] for i in indices]
        shuffled_dataset.stats = self.stats
        return shuffled_dataset

    def sample(self, num_samples: int) -> SelfPlayDataset:
        indices = np.random.choice(len(self), num_samples, replace=False)

        sampled_dataset = SelfPlayDataset()
        sampled_dataset.encoded_states = [self.encoded_states[i] for i in indices]
        sampled_dataset.visit_counts = [self.visit_counts[i] for i in indices]
        sampled_dataset.value_targets = [self.value_targets[i] for i in indices]
        sampled_dataset.stats = self.stats.overwrite(num_samples=num_samples)
        return sampled_dataset

    @timeit
    @staticmethod
    def load(file_path: str | PathLike) -> SelfPlayDataset:
        try:
            return SelfPlayDataset.load_strict(file_path)
        except Exception as e:
            from src.util.log import log, LogLevel

            log(f'Error loading dataset from {file_path}: {e}', level=LogLevel.DEBUG)
            return SelfPlayDataset()

    @staticmethod
    def load_strict(file_path: str | PathLike) -> SelfPlayDataset:
        dataset = SelfPlayDataset()
        with h5py.File(file_path, 'r') as file:
            dataset.stats = SelfPlayDataset._load_stats_from_open_file(file)
            stored_states = np.asarray(file['states'][...])  # type: ignore
            stored_visit_counts = np.asarray(file['visit_counts'][...])  # type: ignore
            stored_value_targets = np.asarray(file['value_targets'][...])  # type: ignore

            dataset.encoded_states = stored_states.tolist()
            dataset.visit_counts = [
                visit_count[visit_count[:, 1] > 0].astype(np.uint16, copy=False) for visit_count in stored_visit_counts
            ]
            dataset.value_targets = stored_value_targets.tolist()
        return dataset

    @staticmethod
    def load_stats(file_path: str | PathLike) -> SelfPlayDatasetStats:
        try:
            with h5py.File(file_path, 'r') as file:
                return SelfPlayDataset._load_stats_from_open_file(file)
        except Exception as e:
            from src.util.log import log, LogLevel

            log(f'Error loading dataset stats from {file_path}: {e}', level=LogLevel.DEBUG)
            return SelfPlayDatasetStats()

    @staticmethod
    def _load_stats_from_open_file(file: h5py.File) -> SelfPlayDatasetStats:
        metadata: dict[str, Any] = eval(file.attrs['metadata'])  # type: ignore
        message = f'Invalid metadata. Expected {SelfPlayDataset._get_current_metadata()}, got {metadata}'
        assert metadata == SelfPlayDataset._get_current_metadata(), message

        stats: dict[str, Any] = eval(file.attrs['stats'])  # type: ignore
        return SelfPlayDatasetStats(**stats)

    @staticmethod
    def load_iteration(folder_path: str | PathLike, iteration: int) -> SelfPlayDataset:
        dataset = SelfPlayDataset()
        for file_path in SelfPlayDataset.get_files_to_load_for_iteration(folder_path, iteration):
            dataset += SelfPlayDataset.load(file_path)
        return dataset

    @staticmethod
    def load_iteration_stats(folder_path: str | PathLike, iteration: int) -> SelfPlayDatasetStats:
        stats = SelfPlayDatasetStats()
        for file_path in SelfPlayDataset.get_files_to_load_for_iteration(folder_path, iteration):
            stats += SelfPlayDataset.load_stats(file_path)
        return stats

    @staticmethod
    def get_files_to_load_for_iteration(folder_path: str | PathLike, iteration: int) -> list[Path]:
        old_save_format = list(Path(folder_path).glob(f'memory_{iteration}*.hdf5'))
        new_save_path = Path(folder_path) / f'memory_{iteration}'
        if new_save_path.exists():
            return list(new_save_path.glob('*.hdf5')) + list(old_save_format)
        return old_save_format

    def save_to_path(self, file_path: Path) -> bool:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_folder = file_path.parent.parent / 'tmp'
        tmp_folder.mkdir(parents=True, exist_ok=True)

        tmp_file_path = tmp_folder / file_path.name

        # write a h5py file with states, policy targets and value targets in it
        try:
            with h5py.File(tmp_file_path, 'w') as file:
                file.create_dataset('states', data=np.array(self.encoded_states))
                max_visit_num = max(len(visit_count) for visit_count in self.visit_counts)
                # padd all visit counts to the same length
                padded_visit_counts = np.zeros((len(self.visit_counts), max_visit_num, 2), dtype=np.int32)
                for i, visit_count in enumerate(self.visit_counts):
                    for j, (move, count) in enumerate(visit_count):
                        padded_visit_counts[i, j] = [move, count]
                file.create_dataset('visit_counts', data=padded_visit_counts)
                file.create_dataset('value_targets', data=np.array(self.value_targets))
                # write the metadata information about the current game, action size, representation shape, etc.
                file.attrs['metadata'] = str(SelfPlayDataset._get_current_metadata())
                # write the stats information about the dataset, num_games, total_generation_time
                file.attrs['stats'] = str(self.stats._asdict())

            # move the tmp file to the final location
            tmp_file_path.rename(file_path)

            # if we reach this point, we successfully saved the dataset
            return True
        except Exception as e:
            from src.util.log import log, LogLevel

            log(f'Error saving dataset to {file_path}: {e}', level=LogLevel.DEBUG)
            # if we fail to save, we delete the tmp file
            if tmp_file_path.exists():
                tmp_file_path.unlink()
            return False

    def save(self, folder_path: str | PathLike, iteration: int, suffix: str | None = None) -> Path:
        if suffix:
            file_path = Path(folder_path) / f'memory_{iteration}/{suffix}.hdf5'
            if not self.save_to_path(file_path):
                raise RuntimeError(f'Failed to save dataset to {file_path}')
        else:
            while True:
                file_path = Path(folder_path) / f'memory_{iteration}/{random_id()}.hdf5'
                if not file_path.exists() and self.save_to_path(file_path):
                    break

        return file_path

    def chunked_save(self, folder_path: str | PathLike, iteration: int, chunk_size: int) -> list[Path]:
        chunked_files = []
        for chunk_index, i in enumerate(range(0, len(self), chunk_size)):
            chunk = SelfPlayDataset()
            chunk.encoded_states = self.encoded_states[i : i + chunk_size]
            chunk.visit_counts = self.visit_counts[i : i + chunk_size]
            chunk.value_targets = self.value_targets[i : i + chunk_size]
            if chunk_index == 0:
                chunk.stats = self.stats.overwrite(num_samples=len(chunk))
            else:
                chunk.stats = SelfPlayDatasetStats(num_samples=len(chunk))

            chunked_files.append(chunk.save(folder_path, iteration, f'chunk_{i // chunk_size}_{random_id()}'))

        return chunked_files

    @staticmethod
    def _get_current_metadata() -> dict[str, Any]:
        # metadata information about current game, action size, representation shape, etc.
        return {
            'action_size': str(CurrentGame.action_size),
            'representation_shape': str(CurrentGame.representation_shape),
            'game': CurrentGame.__class__.__name__,
        }

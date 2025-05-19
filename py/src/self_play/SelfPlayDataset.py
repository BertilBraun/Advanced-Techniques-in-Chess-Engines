from __future__ import annotations
from math import ceil

import h5py
import torch
import numpy as np
import numpy.typing as npt
from os import PathLike
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset

from src.Encoding import decode_board_state, encode_board_state
from src.mcts.MCTS import action_probabilities
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
        self.encoded_states: list[bytes] = []
        self.visit_counts: list[npt.NDArray[np.uint16]] = []
        self.value_targets: list[float] = []
        self.stats = SelfPlayDatasetStats()

    def add_generation_stats(self, game_length: int, generation_time: float, resignation: bool) -> None:
        self.stats += SelfPlayDatasetStats(
            num_games=1,
            game_lengths=game_length,
            total_generation_time=generation_time,
            resignations=int(resignation),
        )

    def add_sample(self, state: npt.NDArray[np.int8], visit_counts: list[tuple[int, int]], value_target: float) -> None:
        self.encoded_states.append(encode_board_state(state))
        self.visit_counts.append(np.array(visit_counts, dtype=np.uint16))
        self.value_targets.append(value_target)
        self.stats += SelfPlayDatasetStats(num_samples=1)

    def __len__(self) -> int:
        return len(self.encoded_states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = decode_board_state(self.encoded_states[idx])
        probabilities = action_probabilities(self.visit_counts[idx])
        return (
            torch.from_numpy(state).to(dtype=TORCH_DTYPE, non_blocking=True),
            torch.from_numpy(probabilities).to(dtype=TORCH_DTYPE, non_blocking=True),
            torch.tensor(self.value_targets[idx], dtype=torch.float32),
        )

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

        deduplicated_dataset.stats = SelfPlayDatasetStats(
            num_samples=len(mp),
            num_games=self.stats.num_games,
            game_lengths=self.stats.game_lengths,
            total_generation_time=self.stats.total_generation_time,
            resignations=self.stats.resignations,
        )
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
        sampled_dataset.stats = SelfPlayDatasetStats(
            num_samples=num_samples,
            num_games=self.stats.num_games,
            game_lengths=self.stats.game_lengths,
            total_generation_time=self.stats.total_generation_time,
            resignations=self.stats.resignations,
        )
        return sampled_dataset

    @timeit
    @staticmethod
    def load(file_path: str | PathLike) -> SelfPlayDataset:
        dataset = SelfPlayDataset()
        dataset.stats = SelfPlayDataset.load_stats(file_path)

        try:
            with h5py.File(file_path, 'r') as file:
                dataset.encoded_states = [state for state in file['states']]  # type: ignore
                # parse out the visit counts but only the non-zero ones
                dataset.visit_counts = [
                    visit_count[visit_count[:, 1] > 0]
                    for visit_count in file['visit_counts']  # type: ignore
                ]
                dataset.value_targets = [value_target for value_target in file['value_targets']]  # type: ignore

                return dataset
        except Exception as e:
            from src.util.log import log, LogLevel

            log(f'Error loading dataset from {file_path}: {e}', level=LogLevel.DEBUG)
            return SelfPlayDataset()

    @staticmethod
    def load_stats(file_path: str | PathLike) -> SelfPlayDatasetStats:
        try:
            with h5py.File(file_path, 'r') as file:
                metadata: dict[str, Any] = eval(file.attrs['metadata'])  # type: ignore
                message = f'Invalid metadata. Expected {SelfPlayDataset._get_current_metadata()}, got {metadata}'
                assert metadata == SelfPlayDataset._get_current_metadata(), message

                stats: dict[str, Any] = eval(file.attrs['stats'])  # type: ignore
                return SelfPlayDatasetStats(**stats)
        except Exception as e:
            from src.util.log import log, LogLevel

            log(f'Error loading dataset stats from {file_path}: {e}', level=LogLevel.DEBUG)
            return SelfPlayDatasetStats()

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

    def save_to_path(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # write a h5py file with states, policy targets and value targets in it
        with h5py.File(file_path, 'w') as file:
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

    def save(self, folder_path: str | PathLike, iteration: int, suffix: str | None = None) -> Path:
        if suffix:
            file_path = Path(folder_path) / f'memory_{iteration}/{suffix}.hdf5'
        else:
            while True:
                file_path = Path(folder_path) / f'memory_{iteration}/{random_id()}.hdf5'
                if not file_path.exists():
                    break

        self.save_to_path(file_path)

        return file_path

    def chunked_save(self, folder_path: str | PathLike, iteration: int, chunk_size: int) -> list[Path]:
        chunked_files = []
        for i in range(0, len(self), chunk_size):
            chunk = SelfPlayDataset()
            chunk.encoded_states = self.encoded_states[i : i + chunk_size]
            chunk.visit_counts = self.visit_counts[i : i + chunk_size]
            chunk.value_targets = self.value_targets[i : i + chunk_size]
            fract_of_games = ceil(len(self) / chunk_size)
            chunk.stats = SelfPlayDatasetStats(
                num_samples=len(chunk),
                num_games=self.stats.num_games // fract_of_games,
                total_generation_time=self.stats.total_generation_time / fract_of_games,
                resignations=self.stats.resignations // fract_of_games,
            )

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

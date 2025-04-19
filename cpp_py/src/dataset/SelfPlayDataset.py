from __future__ import annotations
from math import ceil

import numpy as np
import numpy.typing as npt
from os import PathLike
from pathlib import Path

from src.dataset.SelfPlayDatasetIO import SelfPlaySample, load_selfplay_file, write_selfplay_file
from src.util import random_id
from src.dataset.SelfPlayDatasetStats import SelfPlayDatasetStats


class SelfPlayDataset:
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
        self.encoded_states: list[npt.NDArray[np.uint64]] = []
        self.visit_counts: list[npt.NDArray[np.uint32]] = []
        self.value_targets: list[float] = []
        self.stats = SelfPlayDatasetStats()

    def __len__(self) -> int:
        return len(self.encoded_states)

    def __add__(self, other: SelfPlayDataset) -> SelfPlayDataset:
        new_dataset = SelfPlayDataset()
        new_dataset.encoded_states = self.encoded_states + other.encoded_states
        new_dataset.visit_counts = self.visit_counts + other.visit_counts
        new_dataset.value_targets = self.value_targets + other.value_targets
        new_dataset.stats = self.stats + other.stats
        return new_dataset

    def add_sample(
        self,
        encoded_state: npt.NDArray[np.uint64],
        visit_counts: npt.NDArray[np.uint32],
        value_target: float,
    ) -> None:
        """Add a new sample to the dataset"""
        self.encoded_states.append(encoded_state)
        self.visit_counts.append(visit_counts)
        self.value_targets.append(value_target)
        self.stats += SelfPlayDatasetStats(num_samples=1)

    def deduplicate(self) -> None:
        """Deduplicate the data based on the board state by averaging the policy and value targets"""
        mp: dict[tuple[int, ...], tuple[int, tuple[npt.NDArray[np.uint32], float]]] = {}

        for state, visit_counts, value_target in zip(self.encoded_states, self.visit_counts, self.value_targets):
            state = tuple(state)
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
                    (np.array(new_visit_counts, dtype=np.uint32), value_target_sum + value_target),
                )
            else:
                mp[state] = (
                    1,
                    (visit_counts, value_target),
                )

        self.encoded_states = [np.array(state) for state in mp.keys()]
        self.visit_counts = [policy_target for _, (policy_target, _) in mp.values()]
        self.value_targets = [value_target / count for count, (_, value_target) in mp.values()]

    def shuffle(self) -> SelfPlayDataset:
        indices = np.arange(len(self))
        np.random.shuffle(indices)

        shuffled_dataset = SelfPlayDataset()
        shuffled_dataset.encoded_states = [self.encoded_states[i] for i in indices]
        shuffled_dataset.visit_counts = [self.visit_counts[i] for i in indices]
        shuffled_dataset.value_targets = [self.value_targets[i] for i in indices]
        shuffled_dataset.stats = self.stats
        return shuffled_dataset

    @staticmethod
    def load(file_path: str | PathLike) -> SelfPlayDataset:
        dataset = SelfPlayDataset()
        try:
            stats, samples = load_selfplay_file(str(file_path))

            dataset.stats = stats
            for board, visit_counts, result_score in samples:
                dataset.encoded_states.append(np.array(board, dtype=np.uint64))
                dataset.visit_counts.append(np.array(visit_counts, dtype=np.uint32))
                dataset.value_targets.append(result_score)
        except Exception as e:
            from src.util.log import log, LogLevel

            log(f'Error loading dataset from {file_path}: {e}', level=LogLevel.DEBUG)
        return dataset

    @staticmethod
    def load_stats(file_path: str | PathLike) -> SelfPlayDatasetStats:
        try:
            stats, _ = load_selfplay_file(str(file_path), load_samples=False)
            return stats
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
        save_path = Path(folder_path) / f'iteration_{iteration}'
        if not save_path.exists():
            return []
        all_files = list(save_path.glob('memory_*.json')) + list(save_path.glob('memory_*.csv'))
        # all files with the postfix after the last '_' removed
        files_with_removed_postfix: list[str] = []
        for file in all_files:
            file_name = file.name
            if '_' in file_name:
                file_name = file_name[: file_name.rindex('_')]
            files_with_removed_postfix.append(file_name)
        # remove duplicates
        files_with_removed_postfix = list(set(files_with_removed_postfix))

        return [save_path / file_name for file_name in files_with_removed_postfix]

    def save_to_path(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        samples: list[SelfPlaySample] = []
        for board_arr, visit_count_arr, value in zip(self.encoded_states, self.visit_counts, self.value_targets):
            board = board_arr.tolist()
            visit_counts = [tuple(pair) for pair in visit_count_arr.tolist()]
            samples.append((board, visit_counts, value))
        write_selfplay_file(str(file_path), self.stats, samples)

    def save(self, folder_path: str | PathLike, iteration: int, suffix: str | None = None) -> Path:
        if suffix:
            file_path = Path(folder_path) / f'iteration_{iteration}/{suffix}'
        else:
            while True:
                file_path = Path(folder_path) / f'iteration_{iteration}/{random_id()}'
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
                num_games=ceil(self.stats.num_games / fract_of_games),
                total_generation_time=self.stats.total_generation_time / fract_of_games,
                resignations=self.stats.resignations // fract_of_games,
            )

            chunked_files.append(chunk.save(folder_path, iteration, f'memory_{i // chunk_size}'))

        return chunked_files

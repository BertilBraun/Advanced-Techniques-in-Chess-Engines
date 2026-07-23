from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass

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
from src.util.timing import timeit
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.self_play.value_target import (
    REPLAY_SCHEMA_VERSION,
    FinalOutcome,
    ReplayValueTarget,
    TerminationReason,
)


class ReplaySchemaVersionError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingBatch:
    states: torch.Tensor
    policy_targets: torch.Tensor
    final_outcomes: torch.Tensor
    mcts_root_values: torch.Tensor
    outcome_target_eligible: torch.Tensor
    termination_reasons: torch.Tensor

    def pin_memory(self) -> TrainingBatch:
        return TrainingBatch(
            states=self.states.pin_memory(),
            policy_targets=self.policy_targets.pin_memory(),
            final_outcomes=self.final_outcomes.pin_memory(),
            mcts_root_values=self.mcts_root_values.pin_memory(),
            outcome_target_eligible=self.outcome_target_eligible.pin_memory(),
            termination_reasons=self.termination_reasons.pin_memory(),
        )


@dataclass(frozen=True)
class TrainingSample:
    state: torch.Tensor
    policy_target: torch.Tensor
    final_outcome: torch.Tensor
    mcts_root_value: torch.Tensor
    outcome_target_eligible: torch.Tensor
    termination_reason: torch.Tensor


def training_batch_from_raw_samples(
    encoded_states: Sequence[bytes],
    visit_counts: Sequence[npt.NDArray[np.uint16]],
    value_targets: Sequence[ReplayValueTarget],
) -> TrainingBatch:
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

    return TrainingBatch(
        states=states,
        policy_targets=torch.from_numpy(policies),
        final_outcomes=torch.from_numpy(
            np.fromiter((int(target.final_outcome) for target in value_targets), dtype=np.int64)
        ),
        mcts_root_values=torch.from_numpy(
            np.fromiter((target.mcts_root_value for target in value_targets), dtype=np.float32)
        ),
        outcome_target_eligible=torch.from_numpy(
            np.fromiter((target.outcome_target_eligible for target in value_targets), dtype=np.bool_)
        ),
        termination_reasons=torch.from_numpy(
            np.fromiter((int(target.termination_reason) for target in value_targets), dtype=np.int64)
        ),
    )


def preserve_prebatched_samples(batch: TrainingBatch) -> TrainingBatch:
    return batch


class SelfPlayDataset(Dataset[TrainingSample]):
    """Each sample is represented by:
    state: torch.Tensor
    policy_targets: torch.Tensor
    value_target: ReplayValueTarget

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
        self.value_targets: list[ReplayValueTarget] = []
        self.stats = SelfPlayDatasetStats()

    def add_generation_stats(self, game_length: int, generation_time: float) -> None:
        self.stats += SelfPlayDatasetStats(
            num_games=1,
            game_lengths=[game_length],
            total_generation_time=generation_time,
        )

    def add_sample(
        self,
        state: npt.NDArray[np.int8],
        visit_counts: list[tuple[int, int]],
        value_target: ReplayValueTarget,
    ) -> None:
        assert len(visit_counts) > 0, 'Visit counts must not be empty'

        self.encoded_states.append(encode_board_state(state))
        self.visit_counts.append(np.array(visit_counts, dtype=np.uint16))
        self.value_targets.append(value_target)
        self.stats += SelfPlayDatasetStats(num_samples=1)

    def __len__(self) -> int:
        return len(self.encoded_states)

    def __getitem__(self, idx: int) -> TrainingSample:
        state = decode_board_state(self.encoded_states[idx])
        probabilities = action_probabilities(self.visit_counts[idx])

        assert 1 - 1e-2 <= np.sum(probabilities) <= 1 + 1e-2, 'Probabilities must sum to 1'

        value_target = self.value_targets[idx]
        return TrainingSample(
            state=torch.from_numpy(state).to(dtype=torch.float32, non_blocking=USE_GPU),
            policy_target=torch.from_numpy(probabilities).to(dtype=torch.float32, non_blocking=USE_GPU),
            final_outcome=torch.tensor(int(value_target.final_outcome), dtype=torch.int64),
            mcts_root_value=torch.tensor(value_target.mcts_root_value, dtype=torch.float32),
            outcome_target_eligible=torch.tensor(value_target.outcome_target_eligible, dtype=torch.bool),
            termination_reason=torch.tensor(int(value_target.termination_reason), dtype=torch.int64),
        )

    def __getitems__(self, indices: list[int]) -> TrainingBatch:
        return training_batch_from_raw_samples(
            [self.encoded_states[index] for index in indices],
            [self.visit_counts[index] for index in indices],
            [self.value_targets[index] for index in indices],
        )

    def raw_sample(self, idx: int) -> tuple[bytes, npt.NDArray[np.uint16], ReplayValueTarget]:
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
        """Merge only samples whose board and complete value-target provenance agree."""
        merged_samples: dict[
            tuple[bytes, ReplayValueTarget],
            npt.NDArray[np.uint16],
        ] = {}

        for state, visit_counts, value_target in zip(self.encoded_states, self.visit_counts, self.value_targets):
            sample_key = (state, value_target)
            if sample_key in merged_samples:
                visit_count_sum = merged_samples[sample_key]
                counts_by_move = {int(move): int(count) for move, count in visit_count_sum}
                for move, count in visit_counts:
                    counts_by_move[int(move)] = counts_by_move.get(int(move), 0) + int(count)
                merged_samples[sample_key] = np.asarray(tuple(counts_by_move.items()), dtype=np.uint16)
            else:
                merged_samples[sample_key] = visit_counts

        deduplicated_dataset = SelfPlayDataset()

        for (state, value_target), visit_count_sum in merged_samples.items():
            deduplicated_dataset.encoded_states.append(state)
            deduplicated_dataset.visit_counts.append(visit_count_sum)
            deduplicated_dataset.value_targets.append(value_target)

        deduplicated_dataset.stats = self.stats.overwrite(num_samples=len(merged_samples))
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
        return SelfPlayDataset.load_strict(file_path)

    @staticmethod
    def load_strict(file_path: str | PathLike) -> SelfPlayDataset:
        dataset = SelfPlayDataset()
        with h5py.File(file_path, 'r') as file:
            SelfPlayDataset._require_current_schema(file, file_path)
            dataset.stats = SelfPlayDataset._load_stats_from_open_file(file)
            stored_states = np.asarray(file['states'][...])  # type: ignore
            stored_visit_counts = np.asarray(file['visit_counts'][...])  # type: ignore
            stored_final_outcomes = np.asarray(file['final_outcomes'][...], dtype=np.uint8)  # type: ignore
            stored_mcts_root_values = np.asarray(file['mcts_root_values'][...], dtype=np.float32)  # type: ignore
            stored_outcome_eligibility = np.asarray(
                file['outcome_target_eligible'][...],
                dtype=np.bool_,
            )  # type: ignore
            stored_termination_reasons = np.asarray(
                file['termination_reasons'][...],
                dtype=np.uint8,
            )  # type: ignore
            stored_lengths = {
                len(stored_states),
                len(stored_visit_counts),
                len(stored_final_outcomes),
                len(stored_mcts_root_values),
                len(stored_outcome_eligibility),
                len(stored_termination_reasons),
            }
            if len(stored_lengths) != 1:
                raise ValueError(f'Replay {file_path} has inconsistent sample-column lengths.')

            dataset.encoded_states = stored_states.tolist()
            dataset.visit_counts = [
                visit_count[visit_count[:, 1] > 0].astype(np.uint16, copy=False) for visit_count in stored_visit_counts
            ]
            dataset.value_targets = [
                ReplayValueTarget(
                    final_outcome=FinalOutcome(int(final_outcome)),
                    mcts_root_value=float(mcts_root_value),
                    termination_reason=TerminationReason(int(termination_reason)),
                    outcome_target_eligible=bool(outcome_target_eligible),
                )
                for final_outcome, mcts_root_value, termination_reason, outcome_target_eligible in zip(
                    stored_final_outcomes,
                    stored_mcts_root_values,
                    stored_termination_reasons,
                    stored_outcome_eligibility,
                )
            ]
        return dataset

    @staticmethod
    def load_stats(file_path: str | PathLike) -> SelfPlayDatasetStats:
        try:
            with h5py.File(file_path, 'r') as file:
                return SelfPlayDataset._load_stats_from_open_file(file)
        except ReplaySchemaVersionError:
            raise
        except Exception as e:
            from src.util.log import log, LogLevel

            log(f'Error loading dataset stats from {file_path}: {e}', level=LogLevel.DEBUG)
            return SelfPlayDatasetStats()

    @staticmethod
    def _load_stats_from_open_file(file: h5py.File) -> SelfPlayDatasetStats:
        SelfPlayDataset._require_current_schema(file, file.filename)
        metadata: dict[str, Any] = eval(file.attrs['metadata'])  # type: ignore
        message = f'Invalid metadata. Expected {SelfPlayDataset._get_current_metadata()}, got {metadata}'
        assert metadata == SelfPlayDataset._get_current_metadata(), message

        stats: dict[str, Any] = eval(file.attrs['stats'])  # type: ignore
        return SelfPlayDatasetStats(**stats)

    @staticmethod
    def _require_current_schema(file: h5py.File, file_path: str | PathLike) -> None:
        schema_version = file.attrs.get('replay_schema_version')
        if schema_version is None:
            raise ReplaySchemaVersionError(
                f'Replay {file_path} has no schema version and is legacy replay; '
                f'expected schema {REPLAY_SCHEMA_VERSION}. Legacy mixed scalar targets cannot be converted.'
            )
        if int(schema_version) != REPLAY_SCHEMA_VERSION:
            raise ReplaySchemaVersionError(
                f'Replay {file_path} uses schema {schema_version}; expected {REPLAY_SCHEMA_VERSION}.'
            )

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
        if len({len(self.encoded_states), len(self.visit_counts), len(self.value_targets)}) != 1:
            raise ValueError('Replay sample columns must contain the same number of entries.')
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
                file.create_dataset(
                    'final_outcomes',
                    data=np.fromiter(
                        (int(target.final_outcome) for target in self.value_targets),
                        dtype=np.uint8,
                    ),
                )
                file.create_dataset(
                    'mcts_root_values',
                    data=np.fromiter(
                        (target.mcts_root_value for target in self.value_targets),
                        dtype=np.float32,
                    ),
                )
                file.create_dataset(
                    'outcome_target_eligible',
                    data=np.fromiter(
                        (target.outcome_target_eligible for target in self.value_targets),
                        dtype=np.bool_,
                    ),
                )
                file.create_dataset(
                    'termination_reasons',
                    data=np.fromiter(
                        (int(target.termination_reason) for target in self.value_targets),
                        dtype=np.uint8,
                    ),
                )
                # write the metadata information about the current game, action size, representation shape, etc.
                file.attrs['replay_schema_version'] = REPLAY_SCHEMA_VERSION
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

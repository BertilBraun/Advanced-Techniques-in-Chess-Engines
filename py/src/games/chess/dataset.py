from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import json
from math import sqrt

import h5py
import torch
import numpy as np
import numpy.typing as npt
from os import PathLike, replace
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset

from src.games.chess.encoding import decode_board_state, decode_board_states, encode_board_state
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.packed_planes import PackedPlanePayload
from src.self_play.visit_policy import action_probabilities
from src.runtime import USE_GPU
from src.util import random_id
from src.util.timing import timeit
from src.games.chess.dataset_statistics import SelfPlayDatasetStats
from src.self_play.value_target import (
    REPLAY_SCHEMA_VERSION,
    FinalOutcome,
    ReplayValueTarget,
    TerminationReason,
)
from src.train.training_batch import ReplaySampleMetadata, TrainingBatch


class ReplaySchemaVersionError(ValueError):
    pass


EVALUATION_DATASET_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class TrainingSample:
    state: torch.Tensor
    policy_target: torch.Tensor
    final_outcome: torch.Tensor
    mcts_root_value: torch.Tensor
    outcome_target_eligible: torch.Tensor
    material_result_score: torch.Tensor
    material_target_eligible: torch.Tensor
    termination_reason: torch.Tensor
    ply: torch.Tensor
    current_player_piece_count: torch.Tensor
    opponent_piece_count: torch.Tensor
    occurrence_count: torch.Tensor
    sample_weight: torch.Tensor


@dataclass(frozen=True)
class DuplicateAggregationDiagnostics:
    raw_sample_count: int
    unique_sample_count: int
    conflicting_target_groups: int
    effective_multiplicity_weight: float

    @property
    def duplicate_factor(self) -> float:
        return self.raw_sample_count / self.unique_sample_count if self.unique_sample_count else 0.0


ReplayAggregationKey = tuple[PackedPlanePayload, FinalOutcome, TerminationReason, bool, bool]


def replay_aggregation_key(
    state: PackedPlanePayload,
    value_target: ReplayValueTarget,
) -> ReplayAggregationKey:
    return (
        state,
        value_target.final_outcome,
        value_target.termination_reason,
        value_target.outcome_target_eligible,
        value_target.material_target_eligible,
    )


def chess_sample_metadata(state: npt.NDArray[np.int8], ply: int) -> ReplaySampleMetadata:
    expected_shape = CHESS_STATE_CONTRACT.game.representation_shape
    if state.shape != expected_shape:
        raise ValueError(f'Expected chess state shape {expected_shape}, got {state.shape}.')
    return ReplaySampleMetadata(
        ply=ply,
        current_player_piece_count=int(np.count_nonzero(state[:6])),
        opponent_piece_count=int(np.count_nonzero(state[6:12])),
    )


def training_batch_from_raw_samples(
    encoded_states: Sequence[PackedPlanePayload],
    visit_counts: Sequence[npt.NDArray[np.uint16]],
    value_targets: Sequence[ReplayValueTarget],
    sample_metadata: Sequence[ReplaySampleMetadata],
) -> TrainingBatch:
    batch_size = len(encoded_states)
    if len(visit_counts) != batch_size or len(value_targets) != batch_size or len(sample_metadata) != batch_size:
        raise ValueError('Training batch inputs must contain the same number of samples.')

    states = torch.from_numpy(decode_board_states(encoded_states)).to(dtype=torch.float32)
    policies = np.zeros((batch_size, CHESS_STATE_CONTRACT.action_size), dtype=np.float32)
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
        material_result_scores=torch.from_numpy(
            np.fromiter((target.material_result_score for target in value_targets), dtype=np.float32)
        ),
        material_target_eligible=torch.from_numpy(
            np.fromiter((target.material_target_eligible for target in value_targets), dtype=np.bool_)
        ),
        termination_reasons=torch.from_numpy(
            np.fromiter((int(target.termination_reason) for target in value_targets), dtype=np.int64)
        ),
        plies=torch.from_numpy(np.fromiter((metadata.ply for metadata in sample_metadata), dtype=np.int32)),
        current_player_piece_counts=torch.from_numpy(
            np.fromiter((metadata.current_player_piece_count for metadata in sample_metadata), dtype=np.int8)
        ),
        opponent_piece_counts=torch.from_numpy(
            np.fromiter((metadata.opponent_piece_count for metadata in sample_metadata), dtype=np.int8)
        ),
        occurrence_counts=torch.from_numpy(
            np.fromiter((metadata.occurrence_count for metadata in sample_metadata), dtype=np.int32)
        ),
        sample_weights=torch.from_numpy(
            np.sqrt(np.fromiter((metadata.occurrence_count for metadata in sample_metadata), dtype=np.float32))
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
        self.encoded_states: list[PackedPlanePayload] = []
        self.visit_counts: list[npt.NDArray[np.uint16]] = []
        self.value_targets: list[ReplayValueTarget] = []
        self.sample_metadata: list[ReplaySampleMetadata] = []
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
        sample_metadata: ReplaySampleMetadata,
    ) -> None:
        assert len(visit_counts) > 0, 'Visit counts must not be empty'

        self.encoded_states.append(encode_board_state(state))
        self.visit_counts.append(np.array(visit_counts, dtype=np.uint16))
        self.value_targets.append(value_target)
        self.sample_metadata.append(sample_metadata)
        self.stats += SelfPlayDatasetStats(num_samples=1)

    def __len__(self) -> int:
        return len(self.encoded_states)

    def __getitem__(self, idx: int) -> TrainingSample:
        state = decode_board_state(self.encoded_states[idx])
        probabilities = action_probabilities(self.visit_counts[idx], CHESS_STATE_CONTRACT.action_size)

        assert 1 - 1e-2 <= np.sum(probabilities) <= 1 + 1e-2, 'Probabilities must sum to 1'

        value_target = self.value_targets[idx]
        return TrainingSample(
            state=torch.from_numpy(state).to(dtype=torch.float32, non_blocking=USE_GPU),
            policy_target=torch.from_numpy(probabilities).to(dtype=torch.float32, non_blocking=USE_GPU),
            final_outcome=torch.tensor(int(value_target.final_outcome), dtype=torch.int64),
            mcts_root_value=torch.tensor(value_target.mcts_root_value, dtype=torch.float32),
            outcome_target_eligible=torch.tensor(value_target.outcome_target_eligible, dtype=torch.bool),
            material_result_score=torch.tensor(value_target.material_result_score, dtype=torch.float32),
            material_target_eligible=torch.tensor(value_target.material_target_eligible, dtype=torch.bool),
            termination_reason=torch.tensor(int(value_target.termination_reason), dtype=torch.int64),
            ply=torch.tensor(self.sample_metadata[idx].ply, dtype=torch.int32),
            current_player_piece_count=torch.tensor(
                self.sample_metadata[idx].current_player_piece_count,
                dtype=torch.int8,
            ),
            opponent_piece_count=torch.tensor(self.sample_metadata[idx].opponent_piece_count, dtype=torch.int8),
            occurrence_count=torch.tensor(self.sample_metadata[idx].occurrence_count, dtype=torch.int32),
            sample_weight=torch.sqrt(torch.tensor(self.sample_metadata[idx].occurrence_count, dtype=torch.float32)),
        )

    def __getitems__(self, indices: list[int]) -> TrainingBatch:
        return training_batch_from_raw_samples(
            [self.encoded_states[index] for index in indices],
            [self.visit_counts[index] for index in indices],
            [self.value_targets[index] for index in indices],
            [self.sample_metadata[index] for index in indices],
        )

    def raw_sample(
        self,
        idx: int,
    ) -> tuple[PackedPlanePayload, npt.NDArray[np.uint16], ReplayValueTarget, ReplaySampleMetadata]:
        return (
            self.encoded_states[idx],
            self.visit_counts[idx],
            self.value_targets[idx],
            self.sample_metadata[idx],
        )

    def __add__(self, other: SelfPlayDataset) -> SelfPlayDataset:
        new_dataset = SelfPlayDataset()
        new_dataset.encoded_states = self.encoded_states + other.encoded_states
        new_dataset.visit_counts = self.visit_counts + other.visit_counts
        new_dataset.value_targets = self.value_targets + other.value_targets
        new_dataset.sample_metadata = self.sample_metadata + other.sample_metadata
        new_dataset.stats = self.stats + other.stats
        return new_dataset

    @timeit
    def deduplicate(self) -> SelfPlayDataset:
        dataset, _ = self.aggregate_duplicates()
        return dataset

    def aggregate_duplicates(self) -> tuple[SelfPlayDataset, DuplicateAggregationDiagnostics]:
        """Aggregate compatible duplicate targets and retain conflicting hard outcomes."""
        merged_samples: dict[
            ReplayAggregationKey,
            tuple[dict[int, int], float, float, int, ReplaySampleMetadata],
        ] = {}
        outcomes_by_state: dict[bytes, set[FinalOutcome]] = {}

        for state, visit_counts, value_target, metadata in zip(
            self.encoded_states,
            self.visit_counts,
            self.value_targets,
            self.sample_metadata,
        ):
            sample_key = replay_aggregation_key(state, value_target)
            outcomes_by_state.setdefault(state, set()).add(value_target.final_outcome)
            if sample_key in merged_samples:
                (
                    counts_by_move,
                    root_value_sum,
                    material_score_sum,
                    occurrence_count,
                    existing_metadata,
                ) = merged_samples[sample_key]
                for move, count in visit_counts:
                    counts_by_move[int(move)] = counts_by_move.get(int(move), 0) + int(count)
                if (
                    metadata.current_player_piece_count != existing_metadata.current_player_piece_count
                    or metadata.opponent_piece_count != existing_metadata.opponent_piece_count
                ):
                    raise ValueError('Duplicate replay positions disagree on piece-count metadata.')
                added_occurrences = metadata.occurrence_count
                combined_occurrences = occurrence_count + added_occurrences
                merged_samples[sample_key] = (
                    counts_by_move,
                    root_value_sum + value_target.mcts_root_value * added_occurrences,
                    material_score_sum + value_target.material_result_score * added_occurrences,
                    combined_occurrences,
                    ReplaySampleMetadata(
                        ply=round(
                            (existing_metadata.ply * occurrence_count + metadata.ply * added_occurrences)
                            / combined_occurrences
                        ),
                        current_player_piece_count=existing_metadata.current_player_piece_count,
                        opponent_piece_count=existing_metadata.opponent_piece_count,
                        occurrence_count=combined_occurrences,
                        starting_fen=existing_metadata.starting_fen,
                        moves_uci=existing_metadata.moves_uci,
                    ),
                )
            else:
                occurrence_count = metadata.occurrence_count
                merged_samples[sample_key] = (
                    {int(move): int(count) for move, count in visit_counts},
                    value_target.mcts_root_value * occurrence_count,
                    value_target.material_result_score * occurrence_count,
                    occurrence_count,
                    metadata,
                )

        deduplicated_dataset = SelfPlayDataset()

        for (
            state,
            final_outcome,
            termination_reason,
            outcome_target_eligible,
            material_target_eligible,
        ), (
            counts_by_move,
            root_value_sum,
            material_score_sum,
            occurrence_count,
            metadata,
        ) in merged_samples.items():
            maximum_count = max(counts_by_move.values())
            scale = min(1.0, np.iinfo(np.uint16).max / maximum_count)
            visit_count_sum = np.asarray(
                tuple((move, max(1, round(count * scale))) for move, count in sorted(counts_by_move.items())),
                dtype=np.uint16,
            )
            deduplicated_dataset.encoded_states.append(state)
            deduplicated_dataset.visit_counts.append(visit_count_sum)
            deduplicated_dataset.value_targets.append(
                ReplayValueTarget(
                    final_outcome=final_outcome,
                    mcts_root_value=root_value_sum / occurrence_count,
                    termination_reason=termination_reason,
                    outcome_target_eligible=outcome_target_eligible,
                    material_result_score=material_score_sum / occurrence_count,
                    material_target_eligible=material_target_eligible,
                )
            )
            deduplicated_dataset.sample_metadata.append(metadata)

        deduplicated_dataset.stats = self.stats.overwrite(num_samples=len(merged_samples))
        diagnostics = DuplicateAggregationDiagnostics(
            raw_sample_count=sum(metadata.occurrence_count for metadata in self.sample_metadata),
            unique_sample_count=len(merged_samples),
            conflicting_target_groups=sum(len(outcomes) > 1 for outcomes in outcomes_by_state.values()),
            effective_multiplicity_weight=sum(
                sqrt(metadata.occurrence_count) for metadata in deduplicated_dataset.sample_metadata
            ),
        )
        return deduplicated_dataset, diagnostics

    def shuffle(self) -> SelfPlayDataset:
        indices = np.arange(len(self))
        np.random.shuffle(indices)

        shuffled_dataset = SelfPlayDataset()
        shuffled_dataset.encoded_states = [self.encoded_states[i] for i in indices]
        shuffled_dataset.visit_counts = [self.visit_counts[i] for i in indices]
        shuffled_dataset.value_targets = [self.value_targets[i] for i in indices]
        shuffled_dataset.sample_metadata = [self.sample_metadata[i] for i in indices]
        shuffled_dataset.stats = self.stats
        return shuffled_dataset

    def sample(self, num_samples: int) -> SelfPlayDataset:
        indices = np.random.choice(len(self), num_samples, replace=False)

        sampled_dataset = SelfPlayDataset()
        sampled_dataset.encoded_states = [self.encoded_states[i] for i in indices]
        sampled_dataset.visit_counts = [self.visit_counts[i] for i in indices]
        sampled_dataset.value_targets = [self.value_targets[i] for i in indices]
        sampled_dataset.sample_metadata = [self.sample_metadata[i] for i in indices]
        sampled_dataset.stats = self.stats.overwrite(num_samples=num_samples)
        return sampled_dataset

    @timeit
    @staticmethod
    def load(file_path: str | PathLike) -> SelfPlayDataset:
        return SelfPlayDataset.load_strict(file_path)

    @staticmethod
    def load_strict(file_path: str | PathLike) -> SelfPlayDataset:
        return SelfPlayDataset._load_with_schema_versions(file_path, (REPLAY_SCHEMA_VERSION,))

    @staticmethod
    def load_evaluation(file_path: str | PathLike) -> SelfPlayDataset:
        return SelfPlayDataset._load_with_schema_versions(
            file_path,
            (EVALUATION_DATASET_SCHEMA_VERSION, REPLAY_SCHEMA_VERSION),
        )

    @staticmethod
    def _load_with_schema_versions(
        file_path: str | PathLike,
        accepted_schema_versions: tuple[int, ...],
    ) -> SelfPlayDataset:
        dataset = SelfPlayDataset()
        with h5py.File(file_path, 'r') as file:
            schema_version = SelfPlayDataset._require_schema(file, file_path, accepted_schema_versions)
            dataset.stats = SelfPlayDataset._load_stats_from_open_file(file, accepted_schema_versions)
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
            stored_plies = np.asarray(file['plies'][...], dtype=np.int32)  # type: ignore
            stored_current_player_piece_counts = np.asarray(
                file['current_player_piece_counts'][...],
                dtype=np.uint8,
            )  # type: ignore
            stored_opponent_piece_counts = np.asarray(
                file['opponent_piece_counts'][...],
                dtype=np.uint8,
            )  # type: ignore
            if schema_version == REPLAY_SCHEMA_VERSION:
                stored_material_result_scores = np.asarray(
                    file['material_result_scores'][...],
                    dtype=np.float32,
                )  # type: ignore
                stored_material_eligibility = np.asarray(
                    file['material_target_eligible'][...],
                    dtype=np.bool_,
                )  # type: ignore
                stored_occurrence_counts = np.asarray(file['occurrence_counts'][...], dtype=np.int32)  # type: ignore
                stored_starting_fens = np.asarray(file['position_starting_fens'].asstr()[...])
                stored_moves_uci = np.asarray(file['position_moves_uci'].asstr()[...])
            else:
                sample_count = len(stored_states)
                stored_material_result_scores = np.zeros(sample_count, dtype=np.float32)
                stored_material_eligibility = np.zeros(sample_count, dtype=np.bool_)
                stored_occurrence_counts = np.ones(sample_count, dtype=np.int32)
                stored_starting_fens = np.full(sample_count, '', dtype=np.str_)
                stored_moves_uci = np.asarray(['[]'] * sample_count, dtype=np.str_)
            stored_lengths = {
                len(stored_states),
                len(stored_visit_counts),
                len(stored_final_outcomes),
                len(stored_mcts_root_values),
                len(stored_outcome_eligibility),
                len(stored_termination_reasons),
                len(stored_material_result_scores),
                len(stored_material_eligibility),
                len(stored_plies),
                len(stored_current_player_piece_counts),
                len(stored_opponent_piece_counts),
                len(stored_occurrence_counts),
                len(stored_starting_fens),
                len(stored_moves_uci),
            }
            if len(stored_lengths) != 1:
                raise ValueError(f'Replay {file_path} has inconsistent sample-column lengths.')

            if stored_states.ndim == 2 and stored_states.dtype == np.uint8:
                dataset.encoded_states = [
                    CHESS_STATE_CONTRACT.representation.packed_planes.value(row.tobytes()) for row in stored_states
                ]
            else:
                dataset.encoded_states = [
                    CHESS_STATE_CONTRACT.representation.packed_planes.value(bytes(state))
                    for state in stored_states.tolist()
                ]
            dataset.visit_counts = [
                visit_count[visit_count[:, 1] > 0].astype(np.uint16, copy=False) for visit_count in stored_visit_counts
            ]
            dataset.value_targets = [
                ReplayValueTarget(
                    final_outcome=FinalOutcome(int(final_outcome)),
                    mcts_root_value=float(mcts_root_value),
                    termination_reason=TerminationReason(int(termination_reason)),
                    outcome_target_eligible=bool(outcome_target_eligible),
                    material_result_score=float(material_result_score),
                    material_target_eligible=bool(material_target_eligible),
                )
                for (
                    final_outcome,
                    mcts_root_value,
                    termination_reason,
                    outcome_target_eligible,
                    material_result_score,
                    material_target_eligible,
                ) in zip(
                    stored_final_outcomes,
                    stored_mcts_root_values,
                    stored_termination_reasons,
                    stored_outcome_eligibility,
                    stored_material_result_scores,
                    stored_material_eligibility,
                )
            ]
            dataset.sample_metadata = [
                ReplaySampleMetadata(
                    ply=int(ply),
                    current_player_piece_count=int(current_count),
                    opponent_piece_count=int(opponent_count),
                    occurrence_count=int(occurrence_count),
                    starting_fen=str(starting_fen) or None,
                    moves_uci=tuple(json.loads(str(moves_json))),
                )
                for ply, current_count, opponent_count, occurrence_count, starting_fen, moves_json in zip(
                    stored_plies,
                    stored_current_player_piece_counts,
                    stored_opponent_piece_counts,
                    stored_occurrence_counts,
                    stored_starting_fens,
                    stored_moves_uci,
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
    def _load_stats_from_open_file(
        file: h5py.File,
        accepted_schema_versions: tuple[int, ...] = (REPLAY_SCHEMA_VERSION,),
    ) -> SelfPlayDatasetStats:
        SelfPlayDataset._require_schema(file, file.filename, accepted_schema_versions)
        metadata: dict[str, Any] = eval(file.attrs['metadata'])  # type: ignore
        message = f'Invalid metadata. Expected {SelfPlayDataset._get_current_metadata()}, got {metadata}'
        assert metadata == SelfPlayDataset._get_current_metadata(), message

        stats: dict[str, Any] = eval(file.attrs['stats'])  # type: ignore
        return SelfPlayDatasetStats(**stats)

    @staticmethod
    def _require_current_schema(file: h5py.File, file_path: str | PathLike) -> None:
        SelfPlayDataset._require_schema(file, file_path, (REPLAY_SCHEMA_VERSION,))

    @staticmethod
    def _require_schema(
        file: h5py.File,
        file_path: str | PathLike,
        accepted_schema_versions: tuple[int, ...],
    ) -> int:
        schema_version = file.attrs.get('replay_schema_version')
        if schema_version is None:
            raise ReplaySchemaVersionError(
                f'Replay {file_path} has no schema version and is legacy replay; '
                f'expected schema {REPLAY_SCHEMA_VERSION}. Legacy mixed scalar targets cannot be converted.'
            )
        parsed_schema_version = int(schema_version)
        if parsed_schema_version not in accepted_schema_versions:
            raise ReplaySchemaVersionError(
                f'Replay {file_path} uses schema {schema_version}; expected {REPLAY_SCHEMA_VERSION}.'
            )
        return parsed_schema_version

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
        if (
            len(
                {
                    len(self.encoded_states),
                    len(self.visit_counts),
                    len(self.value_targets),
                    len(self.sample_metadata),
                }
            )
            != 1
        ):
            raise ValueError('Replay sample columns must contain the same number of entries.')
        file_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_folder = file_path.parent.parent / 'tmp'
        tmp_folder.mkdir(parents=True, exist_ok=True)

        tmp_file_path = tmp_folder / f'.{file_path.name}.{random_id()}.tmp'

        # write a h5py file with states, policy targets and value targets in it
        try:
            with h5py.File(tmp_file_path, 'w') as file:
                payload_bytes = CHESS_STATE_CONTRACT.representation.packed_planes.payload_bytes
                packed_states = np.frombuffer(
                    b''.join(state.payload for state in self.encoded_states),
                    dtype=np.uint8,
                ).reshape(len(self.encoded_states), payload_bytes)
                file.create_dataset('states', data=packed_states)
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
                file.create_dataset(
                    'material_result_scores',
                    data=np.fromiter(
                        (target.material_result_score for target in self.value_targets),
                        dtype=np.float32,
                    ),
                )
                file.create_dataset(
                    'material_target_eligible',
                    data=np.fromiter(
                        (target.material_target_eligible for target in self.value_targets),
                        dtype=np.bool_,
                    ),
                )
                file.create_dataset(
                    'plies',
                    data=np.fromiter((metadata.ply for metadata in self.sample_metadata), dtype=np.int32),
                )
                file.create_dataset(
                    'current_player_piece_counts',
                    data=np.fromiter(
                        (metadata.current_player_piece_count for metadata in self.sample_metadata),
                        dtype=np.uint8,
                    ),
                )
                file.create_dataset(
                    'opponent_piece_counts',
                    data=np.fromiter(
                        (metadata.opponent_piece_count for metadata in self.sample_metadata),
                        dtype=np.uint8,
                    ),
                )
                file.create_dataset(
                    'occurrence_counts',
                    data=np.fromiter(
                        (metadata.occurrence_count for metadata in self.sample_metadata),
                        dtype=np.int32,
                    ),
                )
                string_dtype = h5py.string_dtype(encoding='utf-8')
                file.create_dataset(
                    'position_starting_fens',
                    data=np.asarray(
                        tuple(metadata.starting_fen or '' for metadata in self.sample_metadata),
                        dtype=object,
                    ),
                    dtype=string_dtype,
                )
                file.create_dataset(
                    'position_moves_uci',
                    data=np.asarray(
                        tuple(json.dumps(metadata.moves_uci) for metadata in self.sample_metadata),
                        dtype=object,
                    ),
                    dtype=string_dtype,
                )
                # write the metadata information about the current game, action size, representation shape, etc.
                file.attrs['replay_schema_version'] = REPLAY_SCHEMA_VERSION
                file.attrs['metadata'] = str(SelfPlayDataset._get_current_metadata())
                # write the stats information about the dataset, num_games, total_generation_time
                file.attrs['stats'] = str(self.stats._asdict())

            # move the tmp file to the final location
            replace(tmp_file_path, file_path)

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
            chunk.sample_metadata = self.sample_metadata[i : i + chunk_size]
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
            'action_size': str(CHESS_STATE_CONTRACT.action_size),
            'representation_shape': str(CHESS_STATE_CONTRACT.game.representation_shape),
            'game': CHESS_STATE_CONTRACT.name,
        }

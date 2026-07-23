from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.Encoding import BINARY_CHANNELS, C, H, SCALAR_CHANNELS, W, decode_board_states, encode_board_state
from src.self_play.SelfPlayDataset import (
    ReplaySchemaVersionError,
    ReplaySampleMetadata,
    SelfPlayDataset,
    TrainingBatch,
    preserve_prebatched_samples,
)
from src.self_play.value_target import ReplayValueTarget, TerminationReason
from src.train.RollingSelfPlayBuffer import RollingSelfPlayBuffer
from src.train.Trainer import prefetch_training_batches


class FixedBatchLoader:
    def __init__(self, batches: tuple[TrainingBatch, ...]) -> None:
        self.batches = batches

    def __iter__(self) -> Iterator[TrainingBatch]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def assert_training_batches_equal(actual: TrainingBatch, expected: TrainingBatch) -> None:
    torch.testing.assert_close(actual.states, expected.states)
    torch.testing.assert_close(actual.policy_targets, expected.policy_targets)
    torch.testing.assert_close(actual.final_outcomes, expected.final_outcomes)
    torch.testing.assert_close(actual.mcts_root_values, expected.mcts_root_values)
    torch.testing.assert_close(actual.outcome_target_eligible, expected.outcome_target_eligible)
    torch.testing.assert_close(actual.termination_reasons, expected.termination_reasons)
    torch.testing.assert_close(actual.plies, expected.plies)
    torch.testing.assert_close(
        actual.current_player_piece_counts,
        expected.current_player_piece_counts,
    )
    torch.testing.assert_close(actual.opponent_piece_counts, expected.opponent_piece_counts)


def encoded_state(seed: int) -> bytes:
    generator = np.random.default_rng(seed)
    state = np.zeros((C, H, W), dtype=np.int8)
    state[list(BINARY_CHANNELS)] = generator.integers(
        0,
        2,
        size=(len(BINARY_CHANNELS), H, W),
        dtype=np.int8,
    )
    state[list(SCALAR_CHANNELS)] = generator.integers(
        -1,
        2,
        size=(len(SCALAR_CHANNELS), 1, 1),
        dtype=np.int8,
    )
    return encode_board_state(state)


def dataset() -> SelfPlayDataset:
    result = SelfPlayDataset()
    result.encoded_states = [encoded_state(seed) for seed in range(4)]
    result.visit_counts = [np.asarray(((seed, seed + 1), (seed + 10, seed + 2)), dtype=np.uint16) for seed in range(4)]
    result.value_targets = [
        ReplayValueTarget.from_scores(-1.0, -0.8, TerminationReason.NATURAL),
        ReplayValueTarget.from_scores(0.0, -0.25, TerminationReason.NATURAL),
        ReplayValueTarget.from_scores(1.0, 0.5, TerminationReason.PLY_CAP),
        ReplayValueTarget.from_scores(1.0, 1.0, TerminationReason.RESIGNATION),
    ]
    result.sample_metadata = [
        ReplaySampleMetadata(
            ply=seed,
            current_player_piece_count=16 - seed,
            opponent_piece_count=15 - seed,
        )
        for seed in range(4)
    ]
    return result


def test_vectorized_board_decode_matches_individual_decode() -> None:
    samples = dataset()

    individual_states = torch.stack([samples[index].state for index in range(len(samples))])
    batched_states = torch.from_numpy(decode_board_states(samples.encoded_states)).to(dtype=torch.float32)

    torch.testing.assert_close(batched_states, individual_states)


def test_prebatched_dataset_matches_individual_samples() -> None:
    samples = dataset()
    individual = [samples[index] for index in range(len(samples))]
    expected = TrainingBatch(
        states=torch.stack([sample.state for sample in individual]),
        policy_targets=torch.stack([sample.policy_target for sample in individual]),
        final_outcomes=torch.stack([sample.final_outcome for sample in individual]),
        mcts_root_values=torch.stack([sample.mcts_root_value for sample in individual]),
        outcome_target_eligible=torch.stack([sample.outcome_target_eligible for sample in individual]),
        termination_reasons=torch.stack([sample.termination_reason for sample in individual]),
        plies=torch.stack([sample.ply for sample in individual]),
        current_player_piece_counts=torch.stack([sample.current_player_piece_count for sample in individual]),
        opponent_piece_counts=torch.stack([sample.opponent_piece_count for sample in individual]),
    )

    batch = samples.__getitems__(list(range(len(samples))))

    assert_training_batches_equal(batch, expected)


def test_prebatched_dataset_dataloader_preserves_components() -> None:
    samples = dataset()
    loader = torch.utils.data.DataLoader(
        samples,
        batch_size=len(samples),
        shuffle=False,
        collate_fn=preserve_prebatched_samples,
    )

    batch = next(iter(loader))

    assert batch.states.shape == (len(samples), C, H, W)
    assert batch.policy_targets.shape == (len(samples), 1880)
    assert batch.final_outcomes.shape == (len(samples),)
    assert batch.mcts_root_values.shape == (len(samples),)
    assert batch.outcome_target_eligible.shape == (len(samples),)
    assert batch.termination_reasons.shape == (len(samples),)
    assert batch.plies.shape == (len(samples),)
    assert batch.current_player_piece_counts.shape == (len(samples),)
    assert batch.opponent_piece_counts.shape == (len(samples),)


def test_dataset_bulk_load_matches_legacy_row_iteration(tmp_path: Path) -> None:
    samples = dataset()
    memory_path = tmp_path / 'memory_0' / 'samples.hdf5'
    assert samples.save_to_path(memory_path)

    with h5py.File(memory_path, 'r') as file:
        expected_states = [bytes(state) for state in file['states']]
        expected_visit_counts = [visit_count[visit_count[:, 1] > 0] for visit_count in file['visit_counts']]
        expected_final_outcomes = [int(outcome) for outcome in file['final_outcomes']]
        expected_mcts_root_values = [float(value) for value in file['mcts_root_values']]
        expected_eligibility = [bool(value) for value in file['outcome_target_eligible']]
        expected_reasons = [int(reason) for reason in file['termination_reasons']]

    loaded_samples = SelfPlayDataset.load(memory_path)

    assert loaded_samples.encoded_states == expected_states
    assert [int(target.final_outcome) for target in loaded_samples.value_targets] == expected_final_outcomes
    assert [target.mcts_root_value for target in loaded_samples.value_targets] == expected_mcts_root_values
    assert [target.outcome_target_eligible for target in loaded_samples.value_targets] == expected_eligibility
    assert [int(target.termination_reason) for target in loaded_samples.value_targets] == expected_reasons
    assert len(loaded_samples.visit_counts) == len(expected_visit_counts)
    for loaded_visit_counts, expected_visit_counts in zip(
        loaded_samples.visit_counts,
        expected_visit_counts,
    ):
        assert loaded_visit_counts.dtype == np.uint16
        np.testing.assert_array_equal(loaded_visit_counts, expected_visit_counts)


def test_background_prefetch_preserves_batch_order_and_values() -> None:
    samples = dataset()
    first_batch = samples.__getitems__([2, 0])
    second_batch = samples.__getitems__([3, 1])
    loader = FixedBatchLoader((first_batch, second_batch))

    prefetched = list(prefetch_training_batches(loader))

    assert len(prefetched) == 2
    for actual_batch, expected_batch in zip(prefetched, (first_batch, second_batch)):
        assert_training_batches_equal(actual_batch, expected_batch)


def test_rolling_buffer_vectorizes_shuffled_indices_across_files(tmp_path: Path) -> None:
    samples = dataset()
    first_dataset = SelfPlayDataset()
    second_dataset = SelfPlayDataset()
    for target, indices in ((first_dataset, (0, 1)), (second_dataset, (2, 3))):
        target.encoded_states = [samples.encoded_states[index] for index in indices]
        target.visit_counts = [samples.visit_counts[index] for index in indices]
        target.value_targets = [samples.value_targets[index] for index in indices]
        target.sample_metadata = [samples.sample_metadata[index] for index in indices]

    first_path = tmp_path / 'memory_0' / 'first.hdf5'
    second_path = tmp_path / 'memory_0' / 'second.hdf5'
    assert first_dataset.save_to_path(first_path)
    assert second_dataset.save_to_path(second_path)

    rolling_buffer = RollingSelfPlayBuffer(max_buffer_samples=10)
    rolling_buffer.update(iteration=0, window_iter=1, files=[first_path, second_path])
    shuffled_indices = [3, 0, 2, 1]
    individual = [rolling_buffer[index] for index in shuffled_indices]
    expected = TrainingBatch(
        states=torch.stack([sample.state for sample in individual]),
        policy_targets=torch.stack([sample.policy_target for sample in individual]),
        final_outcomes=torch.stack([sample.final_outcome for sample in individual]),
        mcts_root_values=torch.stack([sample.mcts_root_value for sample in individual]),
        outcome_target_eligible=torch.stack([sample.outcome_target_eligible for sample in individual]),
        termination_reasons=torch.stack([sample.termination_reason for sample in individual]),
        plies=torch.stack([sample.ply for sample in individual]),
        current_player_piece_counts=torch.stack([sample.current_player_piece_count for sample in individual]),
        opponent_piece_counts=torch.stack([sample.opponent_piece_count for sample in individual]),
    )

    batch = rolling_buffer.__getitems__(shuffled_indices)

    assert_training_batches_equal(batch, expected)


def test_rolling_buffer_replay_updates_are_idempotent(tmp_path: Path) -> None:
    samples = dataset()
    memory_path = tmp_path / 'memory_0' / 'samples.hdf5'
    assert samples.save_to_path(memory_path)
    rolling_buffer = RollingSelfPlayBuffer(max_buffer_samples=10)

    rolling_buffer.update(iteration=0, window_iter=1, files=[memory_path])
    rolling_buffer.update(iteration=0, window_iter=1, files=[memory_path])

    assert len(rolling_buffer) == len(samples)


def test_legacy_mixed_scalar_replay_is_rejected(tmp_path: Path) -> None:
    legacy_path = tmp_path / 'legacy.hdf5'
    with h5py.File(legacy_path, 'w') as file:
        file.create_dataset('states', data=np.asarray((encoded_state(0),)))
        file.create_dataset('visit_counts', data=np.asarray(((((0, 1),)),)))
        file.create_dataset('value_targets', data=np.asarray((0.25,), dtype=np.float32))

    with pytest.raises(ReplaySchemaVersionError, match='Legacy mixed scalar targets cannot be converted'):
        SelfPlayDataset.load_strict(legacy_path)


def test_deduplicate_preserves_conflicting_hard_targets_and_provenance() -> None:
    samples = dataset()
    samples.encoded_states[1] = samples.encoded_states[0]
    samples.visit_counts[1] = samples.visit_counts[0]

    deduplicated = samples.deduplicate()

    assert len(deduplicated) == len(samples)
    assert deduplicated.value_targets[0] != deduplicated.value_targets[1]

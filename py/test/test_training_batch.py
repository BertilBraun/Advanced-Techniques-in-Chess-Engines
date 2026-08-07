from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.games.chess.encoding import (
    BINARY_CHANNELS,
    C,
    H,
    SCALAR_CHANNELS,
    W,
    decode_board_states,
    encode_board_state,
)
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.self_play.SelfPlayDataset import (
    ReplaySchemaVersionError,
    ReplaySampleMetadata,
    SelfPlayDataset,
    TrainingBatch,
    preserve_prebatched_samples,
)
from src.self_play.value_target import ReplayValueTarget, TerminationReason
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
    torch.testing.assert_close(actual.material_result_scores, expected.material_result_scores)
    torch.testing.assert_close(actual.material_target_eligible, expected.material_target_eligible)
    torch.testing.assert_close(actual.termination_reasons, expected.termination_reasons)
    torch.testing.assert_close(actual.plies, expected.plies)
    torch.testing.assert_close(
        actual.current_player_piece_counts,
        expected.current_player_piece_counts,
    )
    torch.testing.assert_close(actual.opponent_piece_counts, expected.opponent_piece_counts)
    torch.testing.assert_close(actual.occurrence_counts, expected.occurrence_counts)


def encoded_state(seed: int):
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
        material_result_scores=torch.stack([sample.material_result_score for sample in individual]),
        material_target_eligible=torch.stack([sample.material_target_eligible for sample in individual]),
        termination_reasons=torch.stack([sample.termination_reason for sample in individual]),
        plies=torch.stack([sample.ply for sample in individual]),
        current_player_piece_counts=torch.stack([sample.current_player_piece_count for sample in individual]),
        opponent_piece_counts=torch.stack([sample.opponent_piece_count for sample in individual]),
        occurrence_counts=torch.stack([sample.occurrence_count for sample in individual]),
        sample_weights=torch.stack([sample.sample_weight for sample in individual]),
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
        expected_states = [
            CHESS_STATE_CONTRACT.representation.packed_planes.value(state.tobytes()) for state in file['states']
        ]
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


def test_legacy_mixed_scalar_replay_is_rejected(tmp_path: Path) -> None:
    legacy_path = tmp_path / 'legacy.hdf5'
    with h5py.File(legacy_path, 'w') as file:
        file.create_dataset(
            'states',
            data=np.frombuffer(encoded_state(0).payload, dtype=np.uint8).reshape(1, -1),
        )
        file.create_dataset('visit_counts', data=np.asarray(((((0, 1),)),)))
        file.create_dataset('value_targets', data=np.asarray((0.25,), dtype=np.float32))

    with pytest.raises(ReplaySchemaVersionError, match='Legacy mixed scalar targets cannot be converted'):
        SelfPlayDataset.load_strict(legacy_path)


def test_schema_three_evaluation_dataset_loads_without_relaxing_replay_loading(tmp_path: Path) -> None:
    samples = dataset()
    evaluation_path = tmp_path / 'evaluation-schema-3.hdf5'
    assert samples.save_to_path(evaluation_path)
    with h5py.File(evaluation_path, 'r+') as file:
        file.attrs['replay_schema_version'] = 3
        for column in (
            'material_result_scores',
            'material_target_eligible',
            'occurrence_counts',
            'position_starting_fens',
            'position_moves_uci',
        ):
            del file[column]

    with pytest.raises(ReplaySchemaVersionError, match='uses schema 3; expected 6'):
        SelfPlayDataset.load_strict(evaluation_path)

    loaded = SelfPlayDataset.load_evaluation(evaluation_path)

    assert len(loaded) == len(samples)
    assert all(not target.material_target_eligible for target in loaded.value_targets)
    assert all(target.material_result_score == 0.0 for target in loaded.value_targets)
    assert all(metadata.occurrence_count == 1 for metadata in loaded.sample_metadata)
    assert all(metadata.starting_fen is None for metadata in loaded.sample_metadata)
    assert all(not metadata.moves_uci for metadata in loaded.sample_metadata)


def test_deduplicate_preserves_conflicting_hard_targets_and_provenance() -> None:
    samples = dataset()
    samples.encoded_states[1] = samples.encoded_states[0]
    samples.visit_counts[1] = samples.visit_counts[0]

    deduplicated = samples.deduplicate()

    assert len(deduplicated) == len(samples)
    assert deduplicated.value_targets[0] != deduplicated.value_targets[1]
    assert all(metadata.occurrence_count == 1 for metadata in deduplicated.sample_metadata)


def test_duplicate_aggregation_averages_compatible_targets_and_tracks_multiplicity() -> None:
    samples = dataset()
    samples.encoded_states[1] = samples.encoded_states[0]
    samples.visit_counts[1] = np.asarray(((1, 1), (2, 3)), dtype=np.uint16)
    samples.value_targets[1] = ReplayValueTarget.from_scores(
        -1.0,
        0.4,
        TerminationReason.NATURAL,
    )
    samples.sample_metadata[1] = ReplaySampleMetadata(
        ply=20,
        current_player_piece_count=samples.sample_metadata[0].current_player_piece_count,
        opponent_piece_count=samples.sample_metadata[0].opponent_piece_count,
    )

    aggregated, diagnostics = samples.aggregate_duplicates()

    assert len(aggregated) == len(samples) - 1
    assert aggregated.sample_metadata[0].occurrence_count == 2
    assert aggregated.sample_metadata[0].ply == 10
    assert aggregated.value_targets[0].mcts_root_value == pytest.approx(-0.2)
    assert diagnostics.raw_sample_count == len(samples)
    assert diagnostics.unique_sample_count == len(samples) - 1
    assert diagnostics.duplicate_factor == pytest.approx(len(samples) / (len(samples) - 1))

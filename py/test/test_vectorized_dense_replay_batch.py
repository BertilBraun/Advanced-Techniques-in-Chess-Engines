from __future__ import annotations

import numpy as np
import pytest
import torch
from src.games.chess.contract import ChessStateContract
from src.games.go.contract import GoStateContract
from src.games.representation import decode_packed_planes, encode_packed_planes
from src.replay.batch_loader import build_dense_targets, build_training_batch_from_columns, decode_augmented_states
from src.replay.columnar import (
    ReplayColumnViews,
    ReplayLegalMovesColumnViews,
    ReplayNextPolicyColumnViews,
    ReplayPolicyColumnViews,
    ReplayScalarColumnViews,
    ReplaySearchCorrectionColumnViews,
)
from src.replay.layout import ReplayLayout
from src.training.targets import (
    FutureSearchValueHeadLayout,
    IrreversibleProgressHeadLayout,
    LegalMovesHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    SearchCorrectionHeadLayout,
    TrainingTargetLayout,
)


class _SyntheticChessState(ChessStateContract):
    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        return action_id if augmentation_index == 0 else self.action_size - 1 - action_id


class _InvalidIdentityChessState(_SyntheticChessState):
    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        del augmentation_index
        return self.action_size - 1 - action_id


class _NonBijectiveChessState(_SyntheticChessState):
    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        return action_id if augmentation_index == 0 else 0


def _policy() -> ReplayPolicyColumnViews:
    return ReplayPolicyColumnViews(
        entry_count=np.asarray((2, 1), dtype=np.uint8),
        action_ids=np.asarray(((1, 3), (2, 0)), dtype=np.uint16),
        visit_counts=np.asarray(((3, 1), (5, 0)), dtype=np.uint16),
        legal_count=np.asarray((3, 2), dtype=np.uint8),
        legal_action_ids=np.asarray(((1, 3, 4), (2, 5, 0)), dtype=np.uint16),
    )


def test_vectorized_dense_batch_preserves_every_auxiliary_variant() -> None:
    state = _SyntheticChessState()
    heads = (
        NextPolicyHeadLayout(kind='next_policy', action_size=state.action_size, ply_offset=1),
        RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=100.0),
        FutureSearchValueHeadLayout(kind='future_search_value', ply_offset=1, smooth_l1_beta=0.1),
        IrreversibleProgressHeadLayout(kind='irreversible_progress', horizon_plies=8),
        LegalMovesHeadLayout(kind='legal_moves', action_size=state.action_size),
        SearchCorrectionHeadLayout(kind='search_correction'),
    )
    layout = ReplayLayout(
        packed_planes=state.packed_plane_layout,
        targets=TrainingTargetLayout(action_size=state.action_size, wdl_size=3, auxiliary_heads=heads),
        maximum_policy_entries=2,
        maximum_legal_actions=3,
    )
    decoded = np.zeros((2, 29, 8, 8), dtype=np.int8)
    decoded[0, 0, 0, 1] = 1
    decoded[1, 1, 2, 3] = 1
    decoded[:, 22] = np.asarray((2, -3), dtype=np.int8)[:, np.newaxis, np.newaxis]
    encoded = np.stack(
        tuple(
            np.frombuffer(
                bytes(
                    encode_packed_planes(
                        row,
                        state.packed_plane_layout,
                        state.representation.binary_channels,
                        state.representation.scalar_channels,
                    )
                ),
                dtype=np.uint8,
            )
            for row in decoded
        )
    )
    policy = _policy()
    columns = ReplayColumnViews(
        encoded_state=encoded,
        policy=policy,
        wdl_target=np.asarray(((1.0, 0.0, 0.0), (0.0, 0.5, 0.5)), dtype=np.float32),
        root_value=np.asarray((0.25, -0.5), dtype=np.float32),
        auxiliary=(
            ReplayNextPolicyColumnViews(kind='next_policy', policy=policy, eligible=np.asarray((1, 0), dtype=np.uint8)),
            ReplayScalarColumnViews(
                kind='remaining_game_length',
                value=np.asarray((0.5, 0.25), dtype=np.float32),
                eligible=np.asarray((1, 0), dtype=np.uint8),
            ),
            ReplayScalarColumnViews(
                kind='future_search_value',
                value=np.asarray((-0.2, 0.4), dtype=np.float32),
                eligible=np.asarray((0, 1), dtype=np.uint8),
            ),
            ReplayScalarColumnViews(
                kind='irreversible_progress',
                value=np.asarray((0.1, 0.9), dtype=np.float32),
                eligible=np.asarray((1, 1), dtype=np.uint8),
            ),
            ReplayLegalMovesColumnViews(kind='legal_moves'),
            ReplaySearchCorrectionColumnViews(kind='search_correction', value=np.asarray((0.2, 0.3), dtype=np.float32)),
        ),
        sample_weight=np.asarray((1.0, 2.0), dtype=np.float32),
        source_model_generation=np.asarray((7, 8), dtype=np.uint32),
        source_timestamp=np.asarray((10.0, 11.0), dtype=np.float64),
    )

    batch = build_training_batch_from_columns(columns, layout, state, np.asarray((0, 1), dtype=np.int64))

    expected_second = decode_packed_planes(
        state.packed_plane_layout.value(encoded[1].tobytes()),
        state.packed_plane_layout,
        state.representation.binary_channels,
        state.representation.scalar_channels,
    ).astype(np.float32)[np.newaxis, ...]
    state.transform_decoded_states(expected_second, np.asarray((1,), dtype=np.int64))
    np.testing.assert_array_equal(batch.states[0].numpy(), decoded[0].astype(np.float32))
    np.testing.assert_array_equal(batch.states[1].numpy(), expected_second[0])
    assert batch.policy_targets[0, 1] == 0.75
    assert batch.policy_targets[0, 3] == 0.25
    assert batch.policy_targets[1, state.action_size - 1 - 2] == 1.0
    assert torch.equal(
        batch.policy_legal_action_ids,
        torch.tensor(((1, 3, 4), (state.action_size - 1 - 2, state.action_size - 1 - 5, -1))),
    )
    assert tuple(mask.tolist() for mask in batch.auxiliary_eligibility) == (
        [True, False],
        [True, False],
        [False, True],
        [True, True],
        [True, True],
        [True, True],
    )
    assert batch.auxiliary_targets[0][1].sum() == 0.0
    assert batch.auxiliary_targets[1][1, 0] == 0.0
    assert batch.auxiliary_targets[4][0, 4] == 1.0
    assert batch.auxiliary_targets[4][1, state.action_size - 1 - 5] == 1.0
    assert batch.source_model_generations.dtype is torch.int64
    assert batch.source_created_at_seconds.dtype is torch.float64
    with pytest.raises(ValueError, match='outside the game contract'):
        build_dense_targets(columns, layout, state, np.asarray((0, state.augmentation_count), dtype=np.int64))


@pytest.mark.parametrize(
    'augmentation_indices',
    (
        [0, 1],
        np.asarray((0, 1), dtype=np.int32),
        np.asarray(((0, 1),), dtype=np.int64),
    ),
)
def test_dense_batch_requires_one_dimensional_int64_augmentation_indices(
    augmentation_indices: object,
) -> None:
    state = _SyntheticChessState()
    with pytest.raises(ValueError, match='one-dimensional int64'):
        decode_augmented_states(
            np.zeros((2, state.packed_plane_layout.payload_bytes), dtype=np.uint8),
            state,
            augmentation_indices,  # type: ignore[arg-type]
        )


def test_action_permutations_are_cached_validated_and_read_only() -> None:
    state = _SyntheticChessState()

    permutations = state.action_permutations

    assert permutations is state.action_permutations
    assert permutations.dtype == np.uint16
    assert permutations.shape == (2, state.action_size)
    assert not permutations.flags.writeable
    with pytest.raises(ValueError, match='Identity augmentation'):
        _InvalidIdentityChessState().action_permutations
    with pytest.raises(ValueError, match='bijection'):
        _NonBijectiveChessState().action_permutations


@pytest.mark.parametrize('board_size', (7, 9))
def test_vectorized_go_states_match_every_scalar_transform(board_size: int) -> None:
    state = GoStateContract(board_size=board_size)
    decoded = np.zeros((8, state.channels, board_size, board_size), dtype=np.int8)
    generator = np.random.default_rng(7 + board_size)
    decoded[:, : len(state.binary_channels)] = generator.integers(
        0, 2, size=(8, len(state.binary_channels), board_size, board_size), dtype=np.int8
    )
    decoded[:, state.scalar_channels[0]] = np.arange(8, dtype=np.int8)[:, np.newaxis, np.newaxis]
    transformed = decoded.astype(np.float32)
    augmentations = np.arange(8, dtype=np.int64)

    state.transform_decoded_states(transformed, augmentations)

    for row, augmentation in enumerate(augmentations):
        expected = decoded[row].copy()
        binary = expected[: len(state.binary_channels)]
        if augmentation >= 4:
            binary = np.flip(binary, axis=2)
        rotation = int(augmentation) % 4
        if rotation:
            binary = np.rot90(binary, k=-rotation, axes=(1, 2))
        expected[: len(state.binary_channels)] = binary
        np.testing.assert_array_equal(transformed[row], expected.astype(np.float32))


@pytest.mark.parametrize('board_size', (7, 9))
def test_vectorized_go_policy_transforms_every_symmetry_and_preserves_pass(board_size: int) -> None:
    state = GoStateContract(board_size=board_size)
    row_count = state.augmentation_count
    pass_action = state.pass_action
    policy = ReplayPolicyColumnViews(
        entry_count=np.full(row_count, 2, dtype=np.uint8),
        action_ids=np.tile(np.asarray((0, pass_action), dtype=np.uint16), (row_count, 1)),
        visit_counts=np.tile(np.asarray((3, 1), dtype=np.uint16), (row_count, 1)),
        legal_count=np.full(row_count, 3, dtype=np.uint8),
        legal_action_ids=np.tile(np.asarray((0, 1, pass_action), dtype=np.uint16), (row_count, 1)),
    )
    layout = ReplayLayout(
        packed_planes=state.packed_planes,
        targets=TrainingTargetLayout(action_size=state.action_size, wdl_size=3, auxiliary_heads=()),
        maximum_policy_entries=2,
        maximum_legal_actions=3,
    )
    columns = ReplayColumnViews(
        encoded_state=np.zeros((row_count, state.packed_planes.payload_bytes), dtype=np.uint8),
        policy=policy,
        wdl_target=np.tile(np.asarray((0.0, 1.0, 0.0), dtype=np.float32), (row_count, 1)),
        root_value=np.zeros(row_count, dtype=np.float32),
        auxiliary=(),
        sample_weight=np.ones(row_count, dtype=np.float32),
        source_model_generation=np.zeros(row_count, dtype=np.uint32),
        source_timestamp=np.zeros(row_count, dtype=np.float64),
    )
    augmentations = np.arange(row_count, dtype=np.int64)

    batch = build_training_batch_from_columns(columns, layout, state, augmentations)

    for row, augmentation in enumerate(augmentations):
        transformed_zero = state.action_permutations[augmentation, 0]
        assert batch.policy_targets[row, transformed_zero] == 0.75
        assert batch.policy_targets[row, pass_action] == 0.25
        assert batch.policy_legal_action_ids[row, 2] == pass_action

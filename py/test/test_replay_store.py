from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import src.replay.store as replay_store_module
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.contracts import WdlTarget
from src.games.go.contract import GoStateContract
from src.games.representation import PackedPlaneLayout
from src.replay.columnar import (
    ReplayColumnArray,
    ReplayColumnViews,
    ReplayNextPolicyColumnViews,
    ReplayScalarColumnViews,
    build_column_views,
    flatten_column_views,
)
from src.replay.contracts import (
    EligibleLegalMovesTarget,
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    EligibleScalarAuxiliaryTarget,
    IneligibleNextPolicyTarget,
    IneligibleRemainingGameLengthTarget,
    IneligibleScalarAuxiliaryTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.encoding import encode_replay_columns
from src.replay.layout import (
    ReplayColumnDescriptor,
    ReplayColumnKey,
    ReplayColumnKind,
    ReplayElementType,
    ReplayLayout,
)
from src.replay.store import (
    ReplayAppendTransaction,
    ReplayStore,
    encode_replay_rows,
    plan_replay_append_chain,
)
from src.self_play.completed_game import SearchVisitCounts
from src.training.targets import (
    FutureSearchValueHeadLayout,
    IrreversibleProgressHeadLayout,
    LegalMovesHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    SearchCorrectionHeadLayout,
    TrainingTargetLayout,
)


def _layout(ply_offset: int = 1) -> ReplayLayout:
    return ReplayLayout(
        packed_planes=PackedPlaneLayout(board_size=3, binary_plane_count=2, scalar_count=1),
        targets=TrainingTargetLayout(
            action_size=10,
            wdl_size=3,
            auxiliary_heads=(
                NextPolicyHeadLayout(kind='next_policy', action_size=10, ply_offset=ply_offset),
                RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=20.0),
            ),
        ),
        maximum_policy_entries=4,
        maximum_legal_actions=10,
    )


def _sample(layout: ReplayLayout, action_id: int, generation: int, auxiliary_eligible: bool = True) -> ReplaySample:
    policy = SparsePolicyTarget(
        visits=SearchVisitCounts(
            action_ids=(action_id, (action_id + 1) % 10),
            visit_counts=(7, 3),
        ),
        legal_action_ids=tuple(range(10)),
    )
    auxiliary = EligibleNextPolicyTarget(policy=policy) if auxiliary_eligible else IneligibleNextPolicyTarget()
    return ReplaySample(
        encoded_state=layout.packed_planes.value(bytes([action_id]) * layout.packed_planes.payload_bytes),
        policy=policy,
        wdl_target=WdlTarget(win=0.5, draw=0.25, loss=0.25),
        root_value=0.125,
        auxiliary_targets=(
            auxiliary,
            EligibleRemainingGameLengthTarget(normalized_length=(action_id + 1) / 20),
        ),
        sample_weight=1.5,
        source_model_generation=generation,
        source_created_at_seconds=123.0 + generation,
    )


def test_replay_store_persists_fixed_rows_and_fifo_state(tmp_path: Path) -> None:
    layout = _layout()
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, layout, maximum_capacity=5, logical_capacity=3)
    assert store.allocated_file_size == ReplayStore.projected_file_size(layout, 5)
    samples = tuple(
        _sample(layout, action_id, action_id, auxiliary_eligible=action_id % 2 == 0) for action_id in range(5)
    )

    for sample in samples:
        store.append(sample)
    store.flush()

    assert store.state.size == 3
    assert store.state.evicted_rows == 2
    assert tuple(store.sample_at(index).source_model_generation for index in range(3)) == (2, 3, 4)
    assert store.sample_at(0).policy == samples[2].policy
    stored_auxiliary = store.sample_at(0).auxiliary_targets
    assert stored_auxiliary[0] == samples[2].auxiliary_targets[0]
    assert isinstance(stored_auxiliary[1], EligibleRemainingGameLengthTarget)
    assert stored_auxiliary[1].normalized_length == pytest.approx(0.15)
    store.close()

    reopened = ReplayStore.open(path, layout)
    assert reopened.state.size == 3
    assert tuple(reopened.sample_at(index).source_model_generation for index in range(3)) == (2, 3, 4)
    reopened.set_logical_capacity(2)
    assert reopened.state.size == 2
    assert reopened.state.evicted_rows == 3
    assert tuple(reopened.sample_at(index).source_model_generation for index in range(2)) == (3, 4)
    reopened.close()


def test_replay_store_rejects_layout_with_same_width_but_different_target_semantics(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, _layout(ply_offset=1), maximum_capacity=2, logical_capacity=2)
    store.close()

    with pytest.raises(ValueError, match='layout'):
        ReplayStore.open(path, _layout(ply_offset=2))


def test_replay_store_bulk_extend_matches_individual_appends_across_wraparound(tmp_path: Path) -> None:
    layout = _layout()
    samples = tuple(_sample(layout, action_id % 9, action_id) for action_id in range(8))
    appended = ReplayStore.create(tmp_path / 'appended.bin', layout, maximum_capacity=5, logical_capacity=3)
    extended = ReplayStore.create(tmp_path / 'extended.bin', layout, maximum_capacity=5, logical_capacity=3)

    for sample in samples:
        appended.append(sample)
    extended.extend(samples)

    assert extended.state.maximum_capacity == appended.state.maximum_capacity
    assert extended.state.logical_capacity == appended.state.logical_capacity
    assert extended.state.head == appended.state.head
    assert extended.state.size == appended.state.size
    assert extended.state.evicted_rows == appended.state.evicted_rows
    assert extended.state.total_appended_rows == appended.state.total_appended_rows
    assert tuple(extended.sample_at(index) for index in range(3)) == tuple(
        appended.sample_at(index) for index in range(3)
    )
    appended.close()
    extended.close()


def test_replay_store_rejects_sparse_policy_beyond_fixed_width(tmp_path: Path) -> None:
    layout = _layout()
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=2, logical_capacity=2)
    sample = _sample(layout, 0, 0)
    oversized_policy = SparsePolicyTarget(
        visits=SearchVisitCounts(action_ids=tuple(range(5)), visit_counts=(1,) * 5),
        legal_action_ids=tuple(range(10)),
    )
    oversized = ReplaySample(
        encoded_state=sample.encoded_state,
        policy=oversized_policy,
        wdl_target=sample.wdl_target,
        root_value=sample.root_value,
        auxiliary_targets=sample.auxiliary_targets,
        sample_weight=sample.sample_weight,
        source_model_generation=sample.source_model_generation,
        source_created_at_seconds=sample.source_created_at_seconds,
    )

    with pytest.raises(ValueError, match='retained-entry'):
        store.append(oversized)
    store.close()


def _all_auxiliary_layout() -> ReplayLayout:
    return ReplayLayout(
        packed_planes=PackedPlaneLayout(board_size=3, binary_plane_count=2, scalar_count=1),
        targets=TrainingTargetLayout(
            action_size=10,
            wdl_size=3,
            auxiliary_heads=(
                NextPolicyHeadLayout(kind='next_policy', action_size=10, ply_offset=1),
                RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=8.0),
                FutureSearchValueHeadLayout(kind='future_search_value', ply_offset=2, smooth_l1_beta=0.1),
                IrreversibleProgressHeadLayout(kind='irreversible_progress', horizon_plies=4),
                LegalMovesHeadLayout(kind='legal_moves', action_size=10),
                SearchCorrectionHeadLayout(kind='search_correction'),
            ),
        ),
        maximum_policy_entries=4,
        maximum_legal_actions=10,
    )


def _all_auxiliary_sample(layout: ReplayLayout, generation: int, eligible: bool) -> ReplaySample:
    policy = SparsePolicyTarget(
        visits=SearchVisitCounts(action_ids=(2, 4), visit_counts=(6, 2)),
        legal_action_ids=(1, 2, 4, 7),
    )
    return ReplaySample(
        encoded_state=layout.packed_planes.value(bytes([generation + 1]) * layout.packed_planes.payload_bytes),
        policy=policy,
        wdl_target=WdlTarget(win=0.5, draw=0.25, loss=0.25),
        root_value=-0.5,
        auxiliary_targets=(
            EligibleNextPolicyTarget(policy=policy) if eligible else IneligibleNextPolicyTarget(),
            (
                EligibleRemainingGameLengthTarget(normalized_length=0.5)
                if eligible
                else IneligibleRemainingGameLengthTarget()
            ),
            (
                EligibleScalarAuxiliaryTarget(kind='future_search_value', value=-0.25)
                if eligible
                else IneligibleScalarAuxiliaryTarget(kind='future_search_value')
            ),
            (
                EligibleScalarAuxiliaryTarget(kind='irreversible_progress', value=0.75)
                if eligible
                else IneligibleScalarAuxiliaryTarget(kind='irreversible_progress')
            ),
            EligibleLegalMovesTarget(),
            EligibleScalarAuxiliaryTarget(kind='search_correction', value=0.25),
        ),
        sample_weight=1.5,
        source_model_generation=generation,
        source_created_at_seconds=128.0 + generation,
    )


def test_columnar_store_uses_aligned_nonoverlapping_slabs_and_capacity_independent_digest(tmp_path: Path) -> None:
    layout = _all_auxiliary_layout()
    first = ReplayStore.create(tmp_path / 'first.bin', layout, maximum_capacity=3, logical_capacity=2)
    second = ReplayStore.create(tmp_path / 'second.bin', layout, maximum_capacity=11, logical_capacity=7)

    assert first.layout.digest == second.layout.digest
    assert first.path.read_bytes()[:8] == b'AZRPLY02'
    previous_stop = 65_536
    for physical in first.physical_columns:
        assert physical.offset % 4_096 == 0
        assert physical.slab_bytes % 4_096 == 0
        assert physical.offset >= previous_stop
        previous_stop = physical.offset + physical.slab_bytes
    assert first.allocated_file_size == previous_stop
    first.close()
    second.close()


def test_old_row_encoding_matches_new_typed_columns_for_every_auxiliary_variant(tmp_path: Path) -> None:
    layout = _all_auxiliary_layout()
    samples = (
        _all_auxiliary_sample(layout, generation=0, eligible=True),
        _all_auxiliary_sample(layout, generation=1, eligible=False),
    )
    encoded_rows = encode_replay_rows(layout, samples)
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=4, logical_capacity=4)

    store.extend_rows(encoded_rows, transaction_identity='equivalence')
    gathered = store.gather_logical(np.arange(len(samples), dtype=np.int64))

    for column in flatten_column_views(layout, gathered):
        np.testing.assert_array_equal(column.values, encoded_rows[column.descriptor.key.name])
    assert store.sample_at(0) == samples[0]
    assert store.sample_at(1) == samples[1]
    assert store.state.append_sequence == 1
    assert store.state.last_transaction_identity == 'equivalence'
    store.close()


def test_direct_column_encoding_matches_transitional_rows_and_zeroes_inactive_padding() -> None:
    layout = _all_auxiliary_layout()
    samples = (
        _all_auxiliary_sample(layout, generation=0, eligible=True),
        _all_auxiliary_sample(layout, generation=1, eligible=False),
    )

    columns = encode_replay_columns(layout, samples)
    encoded_rows = encode_replay_rows(layout, samples)

    assert hashlib.sha256(encoded_rows.tobytes()).hexdigest() == (
        '241dd39eb9a985add6a52282c37e73056c2fd3e53665047b2fe9ffc70c326b58'
    )
    for column in flatten_column_views(layout, columns):
        np.testing.assert_array_equal(column.values, encoded_rows[column.descriptor.key.name])
        assert column.values.dtype == column.descriptor.element_type.numpy_dtype
        assert column.values.shape == (len(samples), *column.descriptor.trailing_shape)
    assert np.all(columns.policy.action_ids[:, 2:] == 0)
    assert np.all(columns.policy.visit_counts[:, 2:] == 0)
    assert np.all(columns.policy.legal_action_ids[:, 4:] == 0)
    next_policy = columns.auxiliary[0]
    remaining_length = columns.auxiliary[1]
    future_value = columns.auxiliary[2]
    irreversible_progress = columns.auxiliary[3]
    assert isinstance(next_policy, ReplayNextPolicyColumnViews)
    assert isinstance(remaining_length, ReplayScalarColumnViews)
    assert isinstance(future_value, ReplayScalarColumnViews)
    assert isinstance(irreversible_progress, ReplayScalarColumnViews)
    assert np.all(next_policy.policy.entry_count[1:] == 0)
    assert np.all(next_policy.policy.action_ids[1:] == 0)
    assert np.all(next_policy.policy.visit_counts[1:] == 0)
    assert np.all(next_policy.policy.legal_count[1:] == 0)
    assert np.all(next_policy.policy.legal_action_ids[1:] == 0)
    assert remaining_length.value[1] == 0.0
    assert future_value.value[1] == 0.0
    assert irreversible_progress.value[1] == 0.0


def test_direct_column_encoding_empty_samples_has_exact_canonical_arrays() -> None:
    layout = _all_auxiliary_layout()

    columns = encode_replay_columns(layout, ())
    encoded_rows = encode_replay_rows(layout, ())

    assert columns.row_count == 0
    assert encoded_rows.shape == (0,)
    assert encoded_rows.dtype == layout.row_dtype
    for column in flatten_column_views(layout, columns):
        assert column.values.shape == (0, *column.descriptor.trailing_shape)
        assert column.values.dtype == column.descriptor.element_type.numpy_dtype


@pytest.mark.parametrize('game', ('chess', 'go'))
def test_direct_column_encoding_supports_game_specific_chess_and_go_layouts(game: str) -> None:
    state = CHESS_STATE_CONTRACT if game == 'chess' else GoStateContract(board_size=7)
    layout = ReplayLayout(
        packed_planes=state.packed_plane_layout,
        targets=TrainingTargetLayout(action_size=state.action_size, wdl_size=3, auxiliary_heads=()),
        maximum_policy_entries=4,
        maximum_legal_actions=state.maximum_legal_action_count,
    )
    policy = SparsePolicyTarget(
        visits=SearchVisitCounts(action_ids=(1, 2), visit_counts=(7, 3)),
        legal_action_ids=(0, 1, 2, 3),
    )
    sample = ReplaySample(
        encoded_state=layout.packed_planes.value(bytes([5]) * layout.packed_planes.payload_bytes),
        policy=policy,
        wdl_target=WdlTarget(win=0.5, draw=0.25, loss=0.25),
        root_value=-0.125,
        auxiliary_targets=(),
        sample_weight=1.25,
        source_model_generation=9,
        source_created_at_seconds=123.5,
    )

    columns = encode_replay_columns(layout, (sample,))
    encoded_rows = encode_replay_rows(layout, (sample,))

    for column in flatten_column_views(layout, columns):
        np.testing.assert_array_equal(column.values, encoded_rows[column.descriptor.key.name])
    assert columns.encoded_state[0].tobytes() == bytes(sample.encoded_state)


def test_vectorized_gather_handles_wrap_duplicates_and_append_larger_than_capacity(tmp_path: Path) -> None:
    layout = _layout()
    samples = tuple(_sample(layout, action_id % 9, action_id) for action_id in range(12))
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=5, logical_capacity=3)

    store.extend(samples[:4])
    store.extend(samples[4:])

    assert store.state.size == 3
    assert store.state.total_appended_rows == 12
    assert store.state.evicted_rows == 9
    assert tuple(store.sample_at(index).source_model_generation for index in range(3)) == (9, 10, 11)
    gathered = store.gather_logical(np.asarray([2, 0, 2, 1], dtype=np.int64))
    assert gathered.source_model_generation.tolist() == [11, 9, 11, 10]
    physical = store.logical_to_physical(np.asarray([0, 1, 2], dtype=np.int64))
    assert len(set(physical.tolist())) == 3
    store.close()


def test_capacity_changes_preserve_total_append_count_and_fifo_order(tmp_path: Path) -> None:
    layout = _layout()
    samples = tuple(_sample(layout, action_id, action_id) for action_id in range(7))
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=6, logical_capacity=5)
    store.extend(samples[:5])

    store.set_logical_capacity(2)
    assert store.state.total_appended_rows == 5
    assert tuple(store.sample_at(index).source_model_generation for index in range(2)) == (3, 4)
    store.set_logical_capacity(4)
    store.extend(samples[5:])

    assert store.state.total_appended_rows == 7
    assert tuple(store.sample_at(index).source_model_generation for index in range(4)) == (3, 4, 5, 6)
    store.close()


def test_schema_three_magic_is_rejected_without_runtime_fallback(tmp_path: Path) -> None:
    layout = _layout()
    path = tmp_path / 'old.bin'
    path.write_bytes(b'AZRPLY01' + bytes(65_528))

    with pytest.raises(ValueError, match='unsupported schema'):
        ReplayStore.open(path, layout)


def _column_views(layout: ReplayLayout, samples: tuple[ReplaySample, ...]) -> ReplayColumnViews:
    rows = encode_replay_rows(layout, samples)
    return build_column_views(
        layout,
        tuple(
            ReplayColumnArray(descriptor, np.array(rows[descriptor.key.name], copy=True))
            for descriptor in layout.columns.columns
        ),
    )


@pytest.mark.parametrize(
    'invalid_case',
    (
        'empty_visits',
        'too_many_visits',
        'empty_legal',
        'too_many_legal',
        'duplicate_visits',
        'duplicate_legal',
        'visit_out_of_range',
        'legal_out_of_range',
        'zero_visit_count',
        'visited_action_not_legal',
        'wdl_nonfinite',
        'wdl_negative',
        'wdl_wrong_sum',
        'root_nonfinite',
        'root_out_of_range',
        'weight_nonfinite',
        'weight_not_positive',
        'timestamp_nonfinite',
        'timestamp_negative',
        'eligibility_not_boolean',
        'next_policy_invalid',
        'remaining_length_negative',
        'remaining_length_nonfinite',
        'future_value_out_of_range',
        'future_value_nonfinite',
        'irreversible_progress_out_of_range',
        'irreversible_progress_nonfinite',
        'search_correction_out_of_range',
        'search_correction_nonfinite',
    ),
)
def test_append_columns_rejects_invalid_semantics(tmp_path: Path, invalid_case: str) -> None:
    layout = _all_auxiliary_layout()
    columns = _column_views(layout, (_all_auxiliary_sample(layout, generation=0, eligible=True),))
    next_policy = columns.auxiliary[0]
    remaining_length = columns.auxiliary[1]
    future_value = columns.auxiliary[2]
    irreversible_progress = columns.auxiliary[3]
    search_correction = columns.auxiliary[5]
    assert isinstance(next_policy, ReplayNextPolicyColumnViews)
    assert isinstance(remaining_length, ReplayScalarColumnViews)
    assert isinstance(future_value, ReplayScalarColumnViews)
    assert isinstance(irreversible_progress, ReplayScalarColumnViews)
    match invalid_case:
        case 'empty_visits':
            columns.policy.entry_count[0] = 0
        case 'too_many_visits':
            columns.policy.entry_count[0] = layout.maximum_policy_entries + 1
        case 'empty_legal':
            columns.policy.legal_count[0] = 0
        case 'too_many_legal':
            columns.policy.legal_count[0] = layout.maximum_legal_actions + 1
        case 'duplicate_visits':
            columns.policy.action_ids[0, 1] = columns.policy.action_ids[0, 0]
        case 'duplicate_legal':
            columns.policy.legal_action_ids[0, 1] = columns.policy.legal_action_ids[0, 0]
        case 'visit_out_of_range':
            columns.policy.action_ids[0, 0] = layout.targets.action_size
        case 'legal_out_of_range':
            columns.policy.legal_action_ids[0, 0] = layout.targets.action_size
        case 'zero_visit_count':
            columns.policy.visit_counts[0, 0] = 0
        case 'visited_action_not_legal':
            columns.policy.legal_action_ids[0, :4] = (0, 1, 3, 5)
        case 'wdl_nonfinite':
            columns.wdl_target[0, 0] = np.nan
        case 'wdl_negative':
            columns.wdl_target[0] = (-0.25, 0.5, 0.75)
        case 'wdl_wrong_sum':
            columns.wdl_target[0] = (0.25, 0.25, 0.25)
        case 'root_nonfinite':
            columns.root_value[0] = np.inf
        case 'root_out_of_range':
            columns.root_value[0] = 1.25
        case 'weight_nonfinite':
            columns.sample_weight[0] = np.nan
        case 'weight_not_positive':
            columns.sample_weight[0] = 0.0
        case 'timestamp_nonfinite':
            columns.source_timestamp[0] = np.inf
        case 'timestamp_negative':
            columns.source_timestamp[0] = -1.0
        case 'eligibility_not_boolean':
            next_policy.eligible[0] = 2
        case 'next_policy_invalid':
            next_policy.policy.entry_count[0] = 0
        case 'remaining_length_negative':
            remaining_length.value[0] = -0.25
        case 'remaining_length_nonfinite':
            remaining_length.value[0] = np.nan
        case 'future_value_out_of_range':
            future_value.value[0] = 1.25
        case 'future_value_nonfinite':
            future_value.value[0] = np.inf
        case 'irreversible_progress_out_of_range':
            irreversible_progress.value[0] = 1.25
        case 'irreversible_progress_nonfinite':
            irreversible_progress.value[0] = np.nan
        case 'search_correction_out_of_range':
            search_correction.value[0] = -0.25
        case 'search_correction_nonfinite':
            search_correction.value[0] = np.inf
        case _:
            raise AssertionError(f'Unhandled invalid replay case: {invalid_case}')
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=2, logical_capacity=2)

    with pytest.raises(ValueError):
        store.append_columns(columns)
    store.close()


def test_append_columns_ignores_inactive_auxiliary_cells(tmp_path: Path) -> None:
    layout = _all_auxiliary_layout()
    columns = _column_views(layout, (_all_auxiliary_sample(layout, generation=0, eligible=False),))
    next_policy = columns.auxiliary[0]
    remaining_length = columns.auxiliary[1]
    future_value = columns.auxiliary[2]
    irreversible_progress = columns.auxiliary[3]
    assert isinstance(next_policy, ReplayNextPolicyColumnViews)
    assert isinstance(remaining_length, ReplayScalarColumnViews)
    assert isinstance(future_value, ReplayScalarColumnViews)
    assert isinstance(irreversible_progress, ReplayScalarColumnViews)
    next_policy.policy.entry_count[0] = 255
    next_policy.policy.legal_count[0] = 0
    next_policy.policy.action_ids[0] = 65_535
    next_policy.policy.visit_counts[0] = 0
    remaining_length.value[0] = np.nan
    future_value.value[0] = np.inf
    irreversible_progress.value[0] = -1.0
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=2, logical_capacity=2)

    store.append_columns(columns)

    assert store.state.size == 1
    store.close()


def test_transaction_identity_is_idempotent_and_zero_row_transactions_commit(tmp_path: Path) -> None:
    layout = _layout()
    rows = encode_replay_rows(layout, (_sample(layout, 0, 0),))
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=3, logical_capacity=3)
    store.extend_rows(rows, transaction_identity='rows-1')
    committed = store.state

    store.extend_rows(rows, transaction_identity='rows-1')

    assert store.state == committed
    with pytest.raises(ValueError, match='different row count'):
        store.extend_rows(np.empty((0,), dtype=layout.row_dtype), transaction_identity='rows-1')
    empty_rows = np.empty((0,), dtype=layout.row_dtype)
    store.extend_rows(empty_rows, transaction_identity='empty-2')
    assert store.state.size == 1
    assert store.state.total_appended_rows == 1
    assert store.state.append_sequence == committed.append_sequence + 1
    assert store.state.last_transaction_identity == 'empty-2'
    empty_committed = store.state
    store.extend_rows(empty_rows, transaction_identity='empty-2')
    assert store.state == empty_committed
    store.close()


def test_append_plan_can_be_reapplied_only_to_its_exact_before_or_after_state(tmp_path: Path) -> None:
    layout = _layout()
    columns = _column_views(layout, (_sample(layout, 0, 0),))
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=3, logical_capacity=3)
    plan = store.plan_append(1, 'planned-1')

    store.apply_append_plan(columns, plan)
    committed = store.state
    store.apply_append_plan(columns, plan)

    assert store.state == committed
    store.append(_sample(layout, 1, 1))
    with pytest.raises(ValueError, match='current store state'):
        store.apply_append_plan(columns, plan)
    with pytest.raises(ValueError, match='ambiguous store state'):
        store.reapply_append_plan(columns, plan)
    store.close()


def test_append_chain_is_pure_and_includes_zero_row_transactions_after_capacity_change(tmp_path: Path) -> None:
    layout = _layout()
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=6, logical_capacity=5)
    store.extend(tuple(_sample(layout, generation, generation) for generation in range(5)))
    store.set_logical_capacity(2)
    store.set_logical_capacity(4)
    starting_state = store.state

    plans = plan_replay_append_chain(
        starting_state,
        (
            ReplayAppendTransaction(row_count=2, transaction_identity='shard-1'),
            ReplayAppendTransaction(row_count=0, transaction_identity='shard-empty'),
            ReplayAppendTransaction(row_count=5, transaction_identity='shard-2'),
        ),
    )

    assert store.state == starting_state
    assert tuple(plan.before for plan in plans) == (starting_state, plans[0].after, plans[1].after)
    assert tuple(plan.after.append_sequence for plan in plans) == tuple(
        starting_state.append_sequence + offset for offset in (1, 2, 3)
    )
    assert plans[1].after.total_appended_rows == plans[1].before.total_appended_rows
    assert plans[1].after.last_transaction_identity == 'shard-empty'
    assert plans[2].after.logical_capacity == 4
    assert plans[2].after.maximum_capacity == 6
    store.close()


def test_append_chain_rejects_duplicate_nonempty_transaction_identities(tmp_path: Path) -> None:
    layout = _layout()
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=3, logical_capacity=3)
    store.extend_rows(
        encode_replay_rows(layout, (_sample(layout, 0, 0),)),
        transaction_identity='committed',
    )

    with pytest.raises(ValueError, match='already present'):
        plan_replay_append_chain(
            store.state,
            (ReplayAppendTransaction(row_count=1, transaction_identity='committed'),),
        )
    with pytest.raises(ValueError, match='already present'):
        plan_replay_append_chain(
            store.state,
            (
                ReplayAppendTransaction(row_count=0, transaction_identity='repeated'),
                ReplayAppendTransaction(row_count=0, transaction_identity='repeated'),
            ),
        )
    store.close()


@pytest.mark.parametrize(
    ('maximum_capacity', 'logical_capacity', 'initial_count', 'slice_sizes'),
    (
        (8, 8, 1, (2, 2)),
        (6, 5, 4, (2, 2)),
        (5, 3, 2, (2, 5)),
    ),
)
def test_multi_slice_append_matches_single_batch_bytes_and_fifo(
    tmp_path: Path,
    maximum_capacity: int,
    logical_capacity: int,
    initial_count: int,
    slice_sizes: tuple[int, ...],
) -> None:
    layout = _layout()
    expected_path = tmp_path / 'expected.bin'
    actual_path = tmp_path / 'actual.bin'
    expected = ReplayStore.create(expected_path, layout, maximum_capacity, logical_capacity)
    actual = ReplayStore.create(actual_path, layout, maximum_capacity, logical_capacity)
    initial = tuple(_sample(layout, generation % 9, generation) for generation in range(initial_count))
    expected.extend_rows(encode_replay_rows(layout, initial), transaction_identity='initial')
    actual.extend_rows(encode_replay_rows(layout, initial), transaction_identity='initial')
    addition_count = sum(slice_sizes)
    additions = tuple(
        _sample(layout, generation % 9, generation)
        for generation in range(initial_count, initial_count + addition_count)
    )
    column_slices = []
    start = 0
    for slice_size in slice_sizes:
        column_slices.append(_column_views(layout, additions[start : start + slice_size]))
        start += slice_size
    expected_plan = expected.plan_append(addition_count, 'multi')
    actual_plan = actual.plan_append(addition_count, 'multi')
    written_states = []
    write_state = actual._write_state

    def record_written_state(state: replay_store_module.ReplayStoreState) -> None:
        written_states.append(state)
        write_state(state)

    actual._write_state = record_written_state

    expected.apply_append_plan(_column_views(layout, additions), expected_plan)
    actual.apply_append_plan_slices(tuple(column_slices), actual_plan)
    actual.apply_append_plan_slices(tuple(column_slices), actual_plan)

    assert actual.state == expected.state
    assert written_states == [actual_plan.after]
    assert tuple(actual.sample_at(index) for index in range(actual.state.size)) == tuple(
        expected.sample_at(index) for index in range(expected.state.size)
    )
    actual._write_state = write_state
    expected.close()
    actual.close()
    assert actual_path.read_bytes() == expected_path.read_bytes()


def test_multi_slice_zero_row_plan_updates_header_once_without_columns(tmp_path: Path) -> None:
    layout = _layout()
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=3, logical_capacity=3)
    before = store.state
    (plan,) = plan_replay_append_chain(
        before,
        (ReplayAppendTransaction(row_count=0, transaction_identity='empty-shard'),),
    )

    store.apply_append_plan_slices((), plan)

    assert store.state == plan.after
    assert store.state.size == before.size
    assert store.state.append_sequence == before.append_sequence + 1
    store.apply_append_plan_slices((), plan)
    assert store.state == plan.after
    store.close()


def test_multi_slice_append_requires_total_rows_to_match_plan(tmp_path: Path) -> None:
    layout = _layout()
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=3, logical_capacity=3)
    columns = _column_views(layout, (_sample(layout, 0, 0),))
    plan = store.plan_append(2, 'wrong-total')

    with pytest.raises(ValueError, match='row count'):
        store.apply_append_plan_slices((columns,), plan)

    assert store.state == plan.before
    store.close()


def test_multi_slice_semantic_validation_rejects_all_slices_before_mutation(tmp_path: Path) -> None:
    layout = _layout()
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, layout, maximum_capacity=4, logical_capacity=4)
    first = _column_views(layout, (_sample(layout, 0, 0),))
    invalid = _column_views(layout, (_sample(layout, 1, 1),))
    invalid.sample_weight[0] = np.nan
    plan = store.plan_append(2, 'invalid-second-slice')
    before_state = store.state
    before_bytes = path.read_bytes()

    with pytest.raises(ValueError, match='sample weights'):
        store.apply_append_plan_slices((first, invalid), plan)

    assert store.state == before_state
    assert path.read_bytes() == before_bytes
    store.close()


def test_multi_slice_reapply_recovers_second_plan_from_first_plan_boundary(tmp_path: Path) -> None:
    layout = _layout()
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, layout, maximum_capacity=4, logical_capacity=3)
    plans = plan_replay_append_chain(
        store.state,
        (
            ReplayAppendTransaction(row_count=2, transaction_identity='shard-1'),
            ReplayAppendTransaction(row_count=3, transaction_identity='shard-2'),
        ),
    )
    first_columns = _column_views(layout, (_sample(layout, 0, 0), _sample(layout, 1, 1)))
    second_slices = (
        _column_views(layout, (_sample(layout, 2, 2),)),
        _column_views(layout, (_sample(layout, 3, 3), _sample(layout, 4, 4))),
    )
    store.apply_append_plan_slices((first_columns,), plans[0])
    store.close()
    header = np.memmap(path, mode='r+', dtype=replay_store_module._HEADER_DTYPE, shape=(1,))
    header[0]['head'] = plans[1].after.head
    header[0]['size'] = plans[1].after.size
    header[0]['evicted_rows'] = plans[1].after.evicted_rows
    header.flush()
    del header
    recovering = ReplayStore.open_for_recovery(path, layout)

    recovering.reapply_append_plan_slices(second_slices, plans[1])

    assert recovering.state == plans[1].after
    assert tuple(recovering.sample_at(index).source_model_generation for index in range(3)) == (2, 3, 4)
    recovering.reapply_append_plan_slices(second_slices, plans[1])
    assert recovering.state == plans[1].after
    recovering.close()


def test_interrupted_header_can_be_opened_and_reapplied_from_exact_plan(tmp_path: Path) -> None:
    layout = _layout()
    path = tmp_path / 'replay.bin'
    columns = _column_views(layout, (_sample(layout, 0, 0), _sample(layout, 1, 1)))
    store = ReplayStore.create(path, layout, maximum_capacity=3, logical_capacity=2)
    plan = store.plan_append(2, 'recovery-1')
    store.close()
    header = np.memmap(path, mode='r+', dtype=replay_store_module._HEADER_DTYPE, shape=(1,))
    header[0]['head'] = plan.after.head
    header[0]['size'] = plan.after.size
    header[0]['evicted_rows'] = plan.after.evicted_rows
    header[0]['total_appended_rows'] = plan.after.total_appended_rows
    header.flush()
    del header

    with pytest.raises(ValueError, match='transaction counters'):
        ReplayStore.open(path, layout)
    recovering = ReplayStore.open_for_recovery(path, layout)
    recovering.reapply_append_plan(columns, plan)

    assert recovering.state == plan.after
    assert tuple(recovering.sample_at(index).source_model_generation for index in range(2)) == (0, 1)
    recovering.close()


def test_read_only_store_exposes_only_detached_gathers_and_rejects_mutation(tmp_path: Path) -> None:
    layout = _layout()
    path = tmp_path / 'replay.bin'
    writable = ReplayStore.create(path, layout, maximum_capacity=3, logical_capacity=3)
    writable.append(_sample(layout, 0, 0))
    writable.close()
    read_only = ReplayStore.open(path, layout, writable=False)
    gathered = read_only.gather_logical(np.asarray([0], dtype=np.int64))
    gathered.source_model_generation[0] = 999

    assert read_only.sample_at(0).source_model_generation == 0
    with pytest.raises(RuntimeError, match='read-only'):
        read_only.append(_sample(layout, 1, 1))
    read_only.close()
    read_only.close()


@pytest.mark.parametrize(
    'corruption',
    (
        'magic',
        'schema',
        'endian',
        'container',
        'header_bytes',
        'descriptor_count',
        'descriptor_bytes',
        'layout_digest_non_ascii',
        'descriptor_digest',
        'transaction_non_ascii',
        'logical_capacity',
        'head',
        'size',
        'append_counters',
        'file_size',
    ),
)
def test_open_rejects_header_file_and_fifo_corruption(tmp_path: Path, corruption: str) -> None:
    layout = _layout()
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, layout, maximum_capacity=3, logical_capacity=2)
    store.close()
    if corruption == 'file_size':
        with path.open('r+b') as file:
            file.truncate(path.stat().st_size - 1)
    else:
        header = np.memmap(path, mode='r+', dtype=replay_store_module._HEADER_DTYPE, shape=(1,))
        match corruption:
            case 'magic':
                header[0]['magic'] = b'INVALID!'
            case 'schema':
                header[0]['schema_version'] = 99
            case 'endian':
                header[0]['endian_marker'] = 0
            case 'container':
                header[0]['container_kind'] = 99
            case 'header_bytes':
                header[0]['header_bytes'] = 1
            case 'descriptor_count':
                header[0]['descriptor_count'] = 0
            case 'descriptor_bytes':
                header[0]['descriptor_bytes'] = 0
            case 'layout_digest_non_ascii':
                header[0]['layout_digest'] = b'\xff' * 64
            case 'descriptor_digest':
                header[0]['descriptor_digest'] = b'0' * 64
            case 'transaction_non_ascii':
                header[0]['last_transaction_identity'] = b'\xff'
            case 'logical_capacity':
                header[0]['logical_capacity'] = 0
            case 'head':
                header[0]['head'] = 3
            case 'size':
                header[0]['size'] = 3
            case 'append_counters':
                header[0]['total_appended_rows'] = 1
            case _:
                raise AssertionError(f'Unhandled corruption case: {corruption}')
        header.flush()
        del header

    with pytest.raises(ValueError):
        ReplayStore.open(path, layout)
    path.unlink()
    assert not path.exists()


@pytest.mark.parametrize('corruption', ('name_non_ascii', 'rank', 'offset'))
def test_open_rejects_explicit_column_descriptor_corruption(tmp_path: Path, corruption: str) -> None:
    layout = _layout()
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, layout, maximum_capacity=3, logical_capacity=2)
    store.close()
    header = np.memmap(path, mode='r+', dtype=replay_store_module._HEADER_DTYPE, shape=(1,))
    descriptors = np.memmap(
        path,
        mode='r+',
        dtype=replay_store_module._COLUMN_DESCRIPTOR_DTYPE,
        offset=replay_store_module._DESCRIPTOR_TABLE_OFFSET,
        shape=(replay_store_module._MAXIMUM_COLUMN_COUNT,),
    )
    match corruption:
        case 'name_non_ascii':
            descriptors[0]['name'] = b'\xff'
        case 'rank':
            descriptors[0]['rank'] = 5
        case 'offset':
            descriptors[0]['offset'] = int(descriptors[0]['offset']) + 1
    count = int(header[0]['descriptor_count'])
    header[0]['descriptor_digest'] = hashlib.sha256(descriptors[:count].tobytes()).hexdigest().encode('ascii')
    descriptors.flush()
    header.flush()
    del descriptors
    del header

    with pytest.raises(ValueError):
        ReplayStore.open(path, layout)
    path.unlink()


def test_column_descriptor_rejects_unrepresentable_name_and_rank() -> None:
    with pytest.raises(ValueError, match='names'):
        ReplayColumnDescriptor(
            ReplayColumnKey(ReplayColumnKind.AUXILIARY_VALUE, 10**64),
            ReplayElementType.FLOAT32,
        )
    with pytest.raises(ValueError, match='four trailing'):
        ReplayColumnDescriptor(
            ReplayColumnKey(ReplayColumnKind.ENCODED_STATE),
            ReplayElementType.UINT8,
            (1, 1, 1, 1, 1),
        )

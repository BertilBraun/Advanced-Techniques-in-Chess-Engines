from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
import src.replay.shard as shard_module
from pydantic import ValidationError
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.contracts import WdlTarget
from src.games.go.contract import GoStateContract
from src.replay.columnar import ReplayColumnArray, ReplayColumnViews, build_column_views, flatten_column_views
from src.replay.contracts import (
    EligibleLegalMovesTarget,
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    EligibleScalarAuxiliaryTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.layout import ReplayLayout
from src.replay.shard import (
    ReplayShardGameMetadata,
    ReplayShardReader,
    ReplayShardSourceGame,
    SealedReplayShardManifest,
    projected_replay_shard_size,
    replay_shard_identity,
    replay_shard_manifest_path,
    replay_shard_physical_columns,
    sealed_replay_shard_manifest_paths,
    write_replay_shard,
)
from src.replay.store import encode_replay_rows
from src.self_play.completed_game import (
    GameIdentity,
    SearchCheckpointObservation,
    SearchObservation,
    SearchStopReason,
    SearchVisitCounts,
    TerminationReason,
)
from src.training.targets import (
    FutureSearchValueHeadLayout,
    IrreversibleProgressHeadLayout,
    LegalMovesHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    SearchCorrectionHeadLayout,
    TrainingTargetLayout,
)


def _layout(game: str) -> ReplayLayout:
    if game == 'chess':
        return ReplayLayout(
            packed_planes=CHESS_STATE_CONTRACT.packed_plane_layout,
            targets=TrainingTargetLayout(
                action_size=CHESS_STATE_CONTRACT.action_size,
                wdl_size=3,
                auxiliary_heads=(),
            ),
            maximum_policy_entries=8,
            maximum_legal_actions=CHESS_STATE_CONTRACT.maximum_legal_action_count,
        )
    state = GoStateContract(board_size=7)
    return ReplayLayout(
        packed_planes=state.packed_plane_layout,
        targets=TrainingTargetLayout(
            action_size=state.action_size,
            wdl_size=3,
            auxiliary_heads=(
                NextPolicyHeadLayout(kind='next_policy', action_size=state.action_size, ply_offset=1),
                RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=100.0),
                FutureSearchValueHeadLayout(kind='future_search_value', ply_offset=2, smooth_l1_beta=0.1),
                IrreversibleProgressHeadLayout(kind='irreversible_progress', horizon_plies=4),
                LegalMovesHeadLayout(kind='legal_moves', action_size=state.action_size),
                SearchCorrectionHeadLayout(kind='search_correction'),
            ),
        ),
        maximum_policy_entries=8,
        maximum_legal_actions=state.maximum_legal_action_count,
    )


def _sample(layout: ReplayLayout, generation: int) -> ReplaySample:
    policy = SparsePolicyTarget(
        visits=SearchVisitCounts(action_ids=(1, 2), visit_counts=(7, 3)),
        legal_action_ids=(0, 1, 2, 3),
    )
    auxiliary = []
    for head in layout.targets.auxiliary_heads:
        match head:
            case NextPolicyHeadLayout():
                auxiliary.append(EligibleNextPolicyTarget(policy=policy))
            case RemainingGameLengthHeadLayout():
                auxiliary.append(EligibleRemainingGameLengthTarget(normalized_length=0.5))
            case FutureSearchValueHeadLayout():
                auxiliary.append(EligibleScalarAuxiliaryTarget(kind='future_search_value', value=-0.25))
            case IrreversibleProgressHeadLayout():
                auxiliary.append(EligibleScalarAuxiliaryTarget(kind='irreversible_progress', value=0.75))
            case LegalMovesHeadLayout():
                auxiliary.append(EligibleLegalMovesTarget())
            case SearchCorrectionHeadLayout():
                auxiliary.append(EligibleScalarAuxiliaryTarget(kind='search_correction', value=0.4))
    return ReplaySample(
        encoded_state=layout.packed_planes.value(bytes([generation + 1]) * layout.packed_planes.payload_bytes),
        policy=policy,
        wdl_target=WdlTarget(win=0.5, draw=0.25, loss=0.25),
        root_value=0.125,
        auxiliary_targets=tuple(auxiliary),
        sample_weight=1.0,
        source_model_generation=generation,
        source_created_at_seconds=100.0 + generation,
    )


def _columns(layout: ReplayLayout, samples: tuple[ReplaySample, ...]) -> ReplayColumnViews:
    rows = encode_replay_rows(layout, samples)
    return build_column_views(
        layout,
        tuple(
            ReplayColumnArray(descriptor, np.array(rows[descriptor.key.name], copy=True))
            for descriptor in layout.columns.columns
        ),
    )


WORKER_INDEX = 5


def _source(game_number: int) -> ReplayShardSourceGame:
    identity = GameIdentity(
        worker_id=3,
        process_instance_id=UUID('38c8809f-a49d-4d98-8da5-034614893665'),
        game_number=game_number,
    )
    return ReplayShardSourceGame(identity=identity, counter=game_number)


def _observation() -> SearchObservation:
    return SearchObservation(
        ply=0,
        model_generation=4,
        policy_target_visits=SearchVisitCounts(action_ids=(1, 2), visit_counts=(7, 3)),
        root_value=0.2,
        highest_visited_child_action_id=1,
        highest_visited_child_visit_count=7,
        highest_visited_child_q=0.1,
        selected_action_id=1,
        full_search=True,
        sample_weight=1.0,
        search_budget=16,
        network_root_value=0.15,
        policy_correction=0.1,
        value_correction=0.2,
        search_correction_target=0.2,
        predicted_search_correction=0.25,
        starting_visits=2,
        final_visits=16,
        stop_reason=SearchStopReason.MAXIMUM,
        learned_gate_evaluated=True,
        checkpoints=(
            SearchCheckpointObservation(
                visits=8,
                leader_action_id=1,
                most_visited_action_id=1,
                top_visit_share=0.7,
                top_two_margin=0.4,
                root_value=0.2,
                root_value_delta=0.05,
                leader_stable=True,
            ),
        ),
    )


def _game_metadata(source: ReplayShardSourceGame, row_start: int, row_count: int) -> ReplayShardGameMetadata:
    return ReplayShardGameMetadata(
        source=source,
        row_start=row_start,
        row_count=row_count,
        length_plies=max(1, row_count),
        termination_reason=TerminationReason.NATURAL,
        is_resignation_continuation=False,
        final_wdl=WdlTarget(win=1.0, draw=0.0, loss=0.0),
        observations=(_observation(),),
        policies_truncated=0,
        retained_visit_mass=10 * row_count,
        discarded_visit_mass=0,
    )


def _write_valid_shard(
    path: Path,
    layout: ReplayLayout,
    samples: tuple[ReplaySample, ...] | None = None,
) -> tuple[SealedReplayShardManifest, Path]:
    shard_samples = samples if samples is not None else (_sample(layout, 0), _sample(layout, 1))
    sources = (_source(0), _source(1))
    games = (
        _game_metadata(sources[0], 0, 1),
        _game_metadata(sources[1], 1, len(shard_samples) - 1),
    )
    manifest = write_replay_shard(path, layout, WORKER_INDEX, 0, 1, _columns(layout, shard_samples), games)
    return manifest, replay_shard_manifest_path(path, manifest.shard_identity)


@pytest.mark.parametrize('game', ('chess', 'go'))
def test_shard_round_trip_matches_old_rows_and_canonical_columns(tmp_path: Path, game: str) -> None:
    layout = _layout(game)
    samples = (_sample(layout, 0), _sample(layout, 1))
    old_rows = encode_replay_rows(layout, samples)
    manifest, manifest_path = _write_valid_shard(tmp_path, layout, samples)

    assert sealed_replay_shard_manifest_paths(tmp_path) == (manifest_path,)
    with ReplayShardReader.open(manifest_path, layout) as reader:
        assert reader.manifest == manifest
        columns = reader.columns
        for column in flatten_column_views(layout, columns):
            np.testing.assert_array_equal(column.values, old_rows[column.descriptor.key.name])
            assert not column.values.flags.writeable
        with pytest.raises(ValueError, match='read-only'):
            columns.source_model_generation[0] = 99
        del columns
    with pytest.raises(RuntimeError, match='closed'):
        reader.columns
    reader.close()


def test_zero_row_game_shard_is_sealed_and_readable(tmp_path: Path) -> None:
    layout = _layout('go')
    sources = (_source(0),)
    games = tuple(_game_metadata(source, 0, 0) for source in sources)

    manifest = write_replay_shard(tmp_path, layout, WORKER_INDEX, 0, 0, _columns(layout, ()), games)

    assert manifest.row_count == 0
    assert manifest.data_size == projected_replay_shard_size(layout, 0) == 4_096
    with ReplayShardReader.open(replay_shard_manifest_path(tmp_path, manifest.shard_identity), layout) as reader:
        assert reader.columns.row_count == 0


def test_shard_slabs_are_derived_from_canonical_descriptors_and_aligned() -> None:
    layout = _layout('go')
    physical = replay_shard_physical_columns(layout, 7)

    assert tuple(column.descriptor for column in physical) == layout.columns.columns
    assert all(column.offset % 4_096 == 0 for column in physical)
    assert all(column.slab_bytes == 7 * column.descriptor.row_bytes for column in physical)
    assert projected_replay_shard_size(layout, 7) == physical[-1].offset + physical[-1].slab_bytes


def test_shard_identity_is_stable_and_changes_with_its_worker_counter_span() -> None:
    layout = _layout('chess')

    first = replay_shard_identity(layout.digest, WORKER_INDEX, 0, 1)

    assert first == replay_shard_identity(layout.digest, WORKER_INDEX, 0, 1)
    assert first != replay_shard_identity(layout.digest, WORKER_INDEX + 1, 0, 1)
    assert first != replay_shard_identity(layout.digest, WORKER_INDEX, 0, 2)
    assert first != replay_shard_identity(layout.digest, WORKER_INDEX, 1, 1)


def test_resealing_the_same_counter_span_adopts_the_existing_shard(tmp_path: Path) -> None:
    layout = _layout('go')
    manifest, _ = _write_valid_shard(tmp_path, layout)
    source = _source(0)

    repeated = write_replay_shard(
        tmp_path,
        layout,
        WORKER_INDEX,
        0,
        1,
        _columns(layout, (_sample(layout, 0),)),
        (_game_metadata(source, 0, 1),),
    )

    assert repeated == manifest


def test_writer_rejects_noncanonical_column_dtype_before_creating_files(tmp_path: Path) -> None:
    layout = _layout('chess')
    source = _source(0)
    identity = replay_shard_identity(layout.digest, WORKER_INDEX, 0, 0)
    columns = _columns(layout, (_sample(layout, 0),))
    invalid = replace(columns, source_model_generation=columns.source_model_generation.astype(np.uint16))

    with pytest.raises(ValueError, match='dtype'):
        write_replay_shard(tmp_path, layout, WORKER_INDEX, 0, 0, invalid, (_game_metadata(source, 0, 1),))

    assert not replay_shard_manifest_path(tmp_path, identity).exists()
    assert not (tmp_path / shard_module.replay_shard_data_name(identity)).exists()


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    (
        ('magic', b'BADMAGIC', 'magic'),
        ('schema', 2, 'schema'),
        ('layout_digest', b'0' * 64, 'layout'),
        ('descriptor_digest', b'0' * 64, 'descriptors'),
        ('row_count', 99, 'row count'),
    ),
)
def test_reader_rejects_corrupt_header(
    tmp_path: Path,
    field: str,
    value: bytes | int,
    message: str,
) -> None:
    layout = _layout('chess')
    manifest, manifest_path = _write_valid_shard(tmp_path, layout)
    data_path = tmp_path / manifest.data_file
    header = list(shard_module._HEADER.unpack(data_path.read_bytes()[: shard_module._HEADER.size]))
    indices = {'magic': 0, 'schema': 1, 'layout_digest': 4, 'descriptor_digest': 5, 'row_count': 6}
    header[indices[field]] = value
    payload = bytearray(data_path.read_bytes())
    payload[: shard_module._HEADER.size] = shard_module._HEADER.pack(*header)
    data_path.write_bytes(payload)
    changed = manifest.model_copy(update={'data_sha256': hashlib.sha256(payload).hexdigest()})
    manifest_path.write_text(changed.model_dump_json() + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match=message):
        ReplayShardReader.open(manifest_path, layout)


def test_reader_rejects_hash_size_and_derived_slab_corruption(tmp_path: Path) -> None:
    layout = _layout('go')
    manifest, manifest_path = _write_valid_shard(tmp_path, layout)
    data_path = tmp_path / manifest.data_file
    payload = bytearray(data_path.read_bytes())
    payload[-1] ^= 1
    data_path.write_bytes(payload)
    with pytest.raises(ValueError, match='hash'):
        ReplayShardReader.open(manifest_path, layout)

    payload.append(0)
    data_path.write_bytes(payload)
    changed = manifest.model_copy(
        update={'data_size': len(payload), 'data_sha256': hashlib.sha256(payload).hexdigest()}
    )
    manifest_path.write_text(changed.model_dump_json() + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='derived slabs'):
        ReplayShardReader.open(manifest_path, layout)

    data_path.write_bytes(payload[:-2])
    with pytest.raises(ValueError, match='size'):
        ReplayShardReader.open(manifest_path, layout)


def test_reader_can_explicitly_skip_whole_file_hash_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout('go')
    manifest, manifest_path = _write_valid_shard(tmp_path, layout)
    data_path = tmp_path / manifest.data_file
    payload = bytearray(data_path.read_bytes())
    payload[-1] ^= 1
    data_path.write_bytes(payload)

    def unexpected_hash_scan(path: Path) -> str:
        raise AssertionError(f'Unexpected replay shard hash scan: {path}')

    monkeypatch.setattr(shard_module, '_file_sha256', unexpected_hash_scan)
    with ReplayShardReader.open(manifest_path, layout, verify_data_hash=False) as reader:
        assert reader.columns.row_count == manifest.row_count


@pytest.mark.parametrize('corruption', ('size', 'header', 'layout'))
def test_reader_without_hash_still_rejects_structural_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    layout = _layout('go')
    manifest, manifest_path = _write_valid_shard(tmp_path, layout)
    data_path = tmp_path / manifest.data_file
    expected_message = corruption
    opened_layout = layout
    match corruption:
        case 'size':
            with data_path.open('ab') as file:
                file.write(b'\x00')
        case 'header':
            payload = bytearray(data_path.read_bytes())
            payload[:8] = b'BADMAGIC'
            data_path.write_bytes(payload)
            expected_message = 'magic'
        case 'layout':
            opened_layout = _layout('chess')
        case _:
            raise AssertionError(f'Unhandled replay shard corruption: {corruption}')

    def unexpected_hash_scan(path: Path) -> str:
        raise AssertionError(f'Unexpected replay shard hash scan: {path}')

    monkeypatch.setattr(shard_module, '_file_sha256', unexpected_hash_scan)
    with pytest.raises(ValueError, match=expected_message):
        ReplayShardReader.open(manifest_path, opened_layout, verify_data_hash=False)


def test_reader_rejects_layout_mismatch(tmp_path: Path) -> None:
    manifest, manifest_path = _write_valid_shard(tmp_path, _layout('chess'))
    assert manifest.layout_digest != _layout('go').digest

    with pytest.raises(ValueError, match='layout'):
        ReplayShardReader.open(manifest_path, _layout('go'))


def test_typed_manifests_reject_bad_spans_order_and_duplicate_games() -> None:
    layout = _layout('chess')
    sources = (_source(0), _source(1))
    identity = replay_shard_identity(layout.digest, WORKER_INDEX, 0, 1)
    games = (_game_metadata(sources[0], 0, 1), _game_metadata(sources[1], 1, 1))
    payload = {
        'shard_identity': identity,
        'layout_digest': layout.digest,
        'worker_index': WORKER_INDEX,
        'first_counter': 0,
        'last_counter': 1,
        'data_file': shard_module.replay_shard_data_name(identity),
        'data_size': projected_replay_shard_size(layout, 2),
        'data_sha256': '0' * 64,
        'row_count': 2,
        'games': tuple(game.model_dump(mode='json') for game in games),
    }
    assert SealedReplayShardManifest.model_validate(payload).shard_identity == identity

    bad_span = payload | {
        'games': (games[0].model_dump(mode='json'), games[1].model_dump(mode='json') | {'row_start': 2})
    }
    with pytest.raises(ValidationError, match='contiguous'):
        SealedReplayShardManifest.model_validate(bad_span)

    reordered = payload | {'games': tuple(reversed(payload['games']))}
    with pytest.raises(ValidationError, match='increasing worker counters'):
        SealedReplayShardManifest.model_validate(reordered)

    with pytest.raises(ValidationError, match='at least 1'):
        SealedReplayShardManifest.model_validate(payload | {'games': ()})

    with pytest.raises(ValidationError, match='inside its identity counter span'):
        SealedReplayShardManifest.model_validate(payload | {'first_counter': 1})


def test_game_metadata_allows_primary_and_auxiliary_policy_truncations_per_row() -> None:
    source = _source(0)
    game = _game_metadata(source, 0, 1)

    validated = ReplayShardGameMetadata.model_validate(game.model_dump() | {'policies_truncated': 2})

    assert validated.policies_truncated == 2


def test_data_and_temporary_files_without_manifest_are_not_sealed(tmp_path: Path) -> None:
    identity = '0' * 64
    data_path = tmp_path / shard_module.replay_shard_data_name(identity)
    temporary_path = tmp_path / f'.{data_path.name}.interrupted.tmp'
    data_path.write_bytes(bytes(4_096))
    temporary_path.write_bytes(b'partial')
    manifest_path = replay_shard_manifest_path(tmp_path, identity)

    assert not manifest_path.exists()
    assert sealed_replay_shard_manifest_paths(tmp_path) == ()
    with pytest.raises(ValueError, match='manifest'):
        ReplayShardReader.open(manifest_path, _layout('chess'))

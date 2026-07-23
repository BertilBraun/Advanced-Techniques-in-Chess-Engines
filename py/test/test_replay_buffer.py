from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from src.Encoding import C, H, W
from src.games.chess.ChessGame import ChessGame
from src.self_play.SelfPlayDataset import ReplaySampleMetadata, SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.self_play.value_target import REPLAY_SCHEMA_VERSION, ReplayValueTarget, TerminationReason
from src.train.RollingReplayBuffer import (
    ActiveCompactionPlan,
    CHESS_MIRROR_ACTION_MAP,
    COMPACTION_TARGET_POSITIONS,
    CompactionStepStatus,
    DEFAULT_REPLAY_CAPACITY,
    ReplayPayloadKind,
    ReplayQuantumRequest,
    ReplayShardManifest,
    RollingReplayBuffer,
    RollingReplayIndexState,
    TerminationCounts,
    _sample_is_mirrored,
    commit_replay_shard,
    decode_rank_quantum,
    file_sha256,
    prefetch_rank_quanta,
)
from tools.benchmark_replay_loader import Arguments, benchmark_coordinated_processes
from tools.production_ddp_fixture import write_replay_fixture


def replay_dataset(sample_count: int, game_count: int = 20) -> SelfPlayDataset:
    dataset = SelfPlayDataset()
    for sample_index in range(sample_count):
        state = np.zeros((C, H, W), dtype=np.int8)
        state[0, sample_index % H, sample_index % W] = 1
        state[6, (sample_index + 1) % H, (sample_index + 2) % W] = 1
        dataset.add_sample(
            state=state,
            visit_counts=[(sample_index % 100, 2), ((sample_index + 1) % 100, 1)],
            value_target=ReplayValueTarget.from_scores(
                final_score=float((sample_index % 3) - 1),
                mcts_root_value=float((sample_index % 5) - 2) / 2,
                termination_reason=TerminationReason.NATURAL,
            ),
            sample_metadata=ReplaySampleMetadata(
                ply=sample_index,
                current_player_piece_count=1,
                opponent_piece_count=1,
            ),
        )
    dataset.stats = SelfPlayDatasetStats(num_samples=sample_count, num_games=game_count)
    return dataset


def commit(
    replay_inbox: Path,
    shard_id: str,
    sample_count: int,
    creation_offset: float = 0.0,
) -> None:
    manifest = commit_replay_shard(
        replay_dataset(sample_count),
        replay_inbox,
        producing_worker=3,
        minimum_model_version=10,
        maximum_model_version=12,
        shard_id=shard_id,
    )
    if creation_offset:
        manifest_path = replay_inbox / manifest.manifest_file_name
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        payload['creation_timestamp_seconds'] += creation_offset
        manifest_path.write_text(json.dumps(payload), encoding='utf-8')


def commit_large_shard(
    replay_inbox: Path,
    shard_id: str,
    sample_count: int,
    creation_timestamp_seconds: float,
) -> ReplayShardManifest:
    hdf5_path = replay_inbox / f'{shard_id}.hdf5'
    write_replay_fixture(hdf5_path, sample_count=sample_count, seed=sample_count)
    manifest = ReplayShardManifest(
        schema_version=REPLAY_SCHEMA_VERSION,
        shard_id=shard_id,
        game_count=max(20, sample_count // 150),
        unique_sample_count=sample_count,
        producing_worker=1,
        minimum_model_version=10,
        maximum_model_version=12,
        termination_counts=TerminationCounts(
            natural=sample_count,
            resignation=0,
            ply_cap=0,
            material_adjudication=0,
            diagnostic=0,
        ),
        content_sha256=file_sha256(hdf5_path),
        creation_timestamp_seconds=creation_timestamp_seconds,
        hdf5_file_name=hdf5_path.name,
    )
    (replay_inbox / manifest.manifest_file_name).write_text(
        manifest.model_dump_json(),
        encoding='utf-8',
    )
    return manifest


def partition_signature(
    buffer: RollingReplayBuffer,
    global_step: int,
    sample_count: int,
) -> tuple[tuple[str, int], ...]:
    with buffer.lease_quantum(global_step, sample_count, world_size=1) as lease:
        return tuple((reference.segment_id, reference.sample_index) for reference in lease.partitions[0].references)


def test_unmanifested_and_temporary_shards_are_invisible(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    replay_inbox.mkdir()
    assert replay_dataset(4).save_to_path(replay_inbox / 'unmanifested.hdf5')
    (replay_inbox / '.partial.manifest.json.tmp').write_text('{}', encoding='utf-8')
    buffer = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json')

    result = buffer.discover_committed_shards()

    assert result.unique_samples == 0
    assert buffer.unique_sample_count == 0


def test_manifest_is_typed_hashed_and_issues_four_credits_once(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'shard-a', 7)
    buffer = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json')

    first = buffer.discover_committed_shards()
    second = buffer.discover_committed_shards()

    assert first.unique_samples == 7
    assert first.presentation_credits == 28
    assert second.unique_samples == 0
    assert second.presentation_credits == 0
    assert buffer.unique_sample_count == 7


def test_default_capacity_is_two_and_a_half_million_disk_backed_positions(tmp_path: Path) -> None:
    buffer = RollingReplayBuffer(tmp_path / 'inbox', tmp_path / 'index.json')

    assert buffer.capacity == DEFAULT_REPLAY_CAPACITY == 2_500_000
    assert buffer.unique_sample_count == 0
    assert buffer.metadata_memory_bytes <= 256


def test_sampler_is_reproducible_without_replacement_and_ddp_partitions_are_disjoint(
    tmp_path: Path,
) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'shard-a', 40)
    buffer = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=73)
    buffer.discover_committed_shards()

    with buffer.lease_quantum(global_step=11, global_sample_count=24, world_size=4) as lease:
        all_references = [
            (reference.segment_id, reference.sample_index)
            for partition in lease.partitions
            for reference in partition.references
        ]
        assert all(len(partition.references) == 6 for partition in lease.partitions)
        assert len(all_references) == len(set(all_references))
        first_signature = tuple(all_references)

    restarted = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=999)
    with restarted.lease_quantum(global_step=11, global_sample_count=24, world_size=4) as lease:
        restarted_signature = tuple(
            (reference.segment_id, reference.sample_index)
            for partition in lease.partitions
            for reference in partition.references
        )
    assert restarted_signature == first_signature


def test_small_buffer_reuses_across_quantum_but_not_within_global_batches(
    tmp_path: Path,
) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'shard-a', 1_500)
    index_path = tmp_path / 'index.json'
    buffer = RollingReplayBuffer(replay_inbox, index_path, sampler_seed=91)
    buffer.discover_committed_shards()

    with buffer.lease_quantum(
        global_step=4,
        global_sample_count=5_120,
        world_size=4,
        global_batch_size=1_024,
    ) as lease:
        references = sorted(
            (reference for partition in lease.partitions for reference in partition.references),
            key=lambda reference: reference.global_batch_position,
        )
        signature = tuple((reference.segment_id, reference.sample_index) for reference in references)
        for offset in range(0, len(references), 1_024):
            batch = references[offset : offset + 1_024]
            identities = {(reference.segment_id, reference.sample_index) for reference in batch}
            assert len(identities) == 1_024
        assert len(set(signature)) < len(signature)

    restarted = RollingReplayBuffer(replay_inbox, index_path)
    with restarted.lease_quantum(
        global_step=4,
        global_sample_count=5_120,
        world_size=4,
        global_batch_size=1_024,
    ) as lease:
        restarted_signature = tuple(
            (reference.segment_id, reference.sample_index)
            for reference in sorted(
                (reference for partition in lease.partitions for reference in partition.references),
                key=lambda reference: reference.global_batch_position,
            )
        )
    assert restarted_signature == signature


def test_buffer_smaller_than_global_batch_fails_clearly(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'shard-a', 100)
    buffer = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json')
    buffer.discover_committed_shards()

    with pytest.raises(ValueError, match='duplicate-free global batch requires 1024'):
        buffer.lease_quantum(
            global_step=0,
            global_sample_count=5_120,
            world_size=4,
            global_batch_size=1_024,
        )


def test_sampling_is_uniform_across_unequal_shards(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'small', 5)
    commit(replay_inbox, 'large', 45, creation_offset=1.0)
    buffer = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=19)
    buffer.discover_committed_shards()

    small_samples = 0
    total_samples = 4_000
    for global_step in range(total_samples):
        signature = partition_signature(buffer, global_step, 1)
        small_samples += signature[0][0] == 'small'

    assert 0.08 <= small_samples / total_samples <= 0.12


def test_mirror_action_map_is_an_involution_and_decode_is_restart_reproducible(
    tmp_path: Path,
) -> None:
    np.testing.assert_array_equal(
        CHESS_MIRROR_ACTION_MAP[CHESS_MIRROR_ACTION_MAP],
        np.arange(len(CHESS_MIRROR_ACTION_MAP)),
    )
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'shard-a', 16)
    buffer = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=31)
    buffer.discover_committed_shards()

    with buffer.lease_quantum(global_step=8, global_sample_count=8, world_size=2) as lease:
        first = buffer.decode_partition(lease.partitions[1], global_step=8)
    restarted = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json')
    with restarted.lease_quantum(global_step=8, global_sample_count=8, world_size=2) as lease:
        second = restarted.decode_partition(lease.partitions[1], global_step=8)

    torch.testing.assert_close(first.states, second.states)
    torch.testing.assert_close(first.policy_targets, second.policy_targets)
    torch.testing.assert_close(first.plies, second.plies)
    torch.testing.assert_close(first.current_player_piece_counts, second.current_player_piece_counts)
    torch.testing.assert_close(first.opponent_piece_counts, second.opponent_piece_counts)
    torch.testing.assert_close(first.policy_targets.sum(dim=1), torch.ones(4))


def test_mirrored_decode_transforms_board_and_legal_policy_move_together(tmp_path: Path) -> None:
    game = ChessGame()
    board = game.get_initial_board()
    state = game.get_canonical_board(board).astype(np.int8)
    original_action = game.encode_move(chess.Move.from_uci('e2e4'), board)
    mirrored_action = game.encode_move(chess.Move.from_uci('d2d4'), board)
    assert CHESS_MIRROR_ACTION_MAP[original_action] == mirrored_action
    assert game.decode_move(mirrored_action, board) in board.get_valid_moves()

    dataset = SelfPlayDataset()
    dataset.add_sample(
        state=state,
        visit_counts=[(original_action, 7)],
        value_target=ReplayValueTarget.from_scores(0.0, 0.0, TerminationReason.NATURAL),
        sample_metadata=ReplaySampleMetadata(
            ply=0,
            current_player_piece_count=16,
            opponent_piece_count=16,
        ),
    )
    dataset.stats = SelfPlayDatasetStats(num_samples=1, num_games=1)
    replay_inbox = tmp_path / 'inbox'
    commit_replay_shard(dataset, replay_inbox, 0, 0, 0, 'opening')
    sampler_seed = 31
    global_step = next(
        step for step in range(100) if _sample_is_mirrored(sampler_seed, step, rank=0, global_batch_position=0)
    )
    buffer = RollingReplayBuffer(
        replay_inbox,
        tmp_path / 'index.json',
        sampler_seed=sampler_seed,
    )
    buffer.discover_committed_shards()

    with buffer.lease_quantum(global_step, global_sample_count=1, world_size=1) as lease:
        batch = buffer.decode_partition(lease.partitions[0], global_step)

    expected_state = torch.from_numpy(np.flip(state, axis=2).copy()).to(dtype=torch.float32)
    torch.testing.assert_close(batch.states[0], expected_state)
    assert int(torch.argmax(batch.policy_targets[0]).item()) == mirrored_action


def test_fifo_eviction_waits_for_active_lease_then_reuses_capacity(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'old', 4)
    buffer = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json', capacity=4)
    buffer.discover_committed_shards()
    lease = buffer.lease_quantum(global_step=0, global_sample_count=2, world_size=1)
    commit(replay_inbox, 'new', 4, creation_offset=1.0)

    buffer.discover_committed_shards()

    assert buffer.shard_count == 2
    assert (replay_inbox / 'old.hdf5').exists()
    lease.release()
    assert buffer.shard_count == 1
    assert not (replay_inbox / 'old.hdf5').exists()
    assert (replay_inbox / 'new.hdf5').exists()


def test_recovery_finishes_recorded_eviction_and_preserves_live_sample_set(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'old', 3)
    commit(replay_inbox, 'live', 5, creation_offset=1.0)
    index_path = tmp_path / 'index.json'
    buffer = RollingReplayBuffer(replay_inbox, index_path, capacity=10, sampler_seed=44)
    buffer.discover_committed_shards()
    state = RollingReplayIndexState.model_validate_json(index_path.read_text(encoding='utf-8'))
    live_segment = next(segment for segment in state.live_segments if segment.segment_id == 'live')
    interrupted_state = RollingReplayIndexState(
        schema_version=state.schema_version,
        sampler_seed=state.sampler_seed,
        live_segments=(live_segment,),
        physical_payloads=state.physical_payloads,
        retired_payload_ids=('old',),
        active_compaction=None,
    )
    index_path.write_text(interrupted_state.model_dump_json(), encoding='utf-8')

    recovered = RollingReplayBuffer(replay_inbox, index_path, capacity=10)

    assert recovered.unique_sample_count == 5
    assert recovered.shard_count == 1
    assert not (replay_inbox / 'old.hdf5').exists()
    assert not (replay_inbox / 'old.manifest.json').exists()
    recovered_state = RollingReplayIndexState.model_validate_json(index_path.read_text(encoding='utf-8'))
    assert recovered_state.retired_payload_ids == ()


def test_index_metadata_memory_scales_with_shards_not_position_payloads(tmp_path: Path) -> None:
    small_inbox = tmp_path / 'small'
    large_inbox = tmp_path / 'large'
    commit(small_inbox, 'small', 1)
    commit(large_inbox, 'large', 2_000)
    small = RollingReplayBuffer(small_inbox, tmp_path / 'small-index.json')
    large = RollingReplayBuffer(large_inbox, tmp_path / 'large-index.json')
    small.discover_committed_shards()
    large.discover_committed_shards()

    assert small.shard_count == large.shard_count == 1
    assert abs(large.metadata_memory_bytes - small.metadata_memory_bytes) < 128
    assert (large_inbox / 'large.hdf5').stat().st_size > (small_inbox / 'small.hdf5').stat().st_size * 10


def test_prefetch_preserves_request_order_and_values(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'shard-a', 32)
    buffer = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=81)
    buffer.discover_committed_shards()
    requests = tuple(
        ReplayQuantumRequest(
            global_step=global_step * 2,
            optimizer_steps=2,
            global_batch_size=4,
            world_size=2,
            rank=1,
        )
        for global_step in range(4)
    )
    expected = tuple(decode_rank_quantum(buffer, request) for request in requests)

    prefetched = tuple(prefetch_rank_quanta(buffer, requests))

    assert len(prefetched) == len(requests)
    for quantum, expected_quantum in zip(prefetched, expected):
        assert quantum.global_step == expected_quantum.global_step
        torch.testing.assert_close(
            quantum.full_batch.plies,
            expected_quantum.full_batch.plies,
        )


def test_compaction_uses_whole_chronological_shards_and_preserves_sampling(
    tmp_path: Path,
) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit_large_shard(replay_inbox, 'source-a', 40_000, 1.0)
    commit_large_shard(replay_inbox, 'source-b', 40_000, 2.0)
    index_path = tmp_path / 'index.json'
    buffer = RollingReplayBuffer(replay_inbox, index_path, sampler_seed=117)
    initial_ingest = buffer.discover_committed_shards()

    assert initial_ingest.presentation_credits == 80_000 * 4
    assert buffer.compact_one_idle_container().status is CompactionStepStatus.WAITING_FOR_MORE_SHARDS

    commit_large_shard(replay_inbox, 'source-c', 25_000, 3.0)
    final_ingest = buffer.discover_committed_shards()
    signature_before = partition_signature(buffer, global_step=17, sample_count=2_048)
    compaction = buffer.compact_one_idle_container()
    signature_after = partition_signature(buffer, global_step=17, sample_count=2_048)

    assert final_ingest.presentation_credits == 25_000 * 4
    assert compaction.status is CompactionStepStatus.COMMITTED_CONTAINER
    assert compaction.compacted_source_shards == 3
    assert compaction.compacted_unique_positions == 105_000
    assert compaction.container_id is not None
    assert signature_after == signature_before
    state = RollingReplayIndexState.model_validate_json(index_path.read_text(encoding='utf-8'))
    compacted_segments = state.live_segments[:3]
    assert tuple(segment.physical_offset for segment in compacted_segments) == (0, 40_000, 80_000)
    assert {segment.physical_payload_id for segment in compacted_segments} == {compaction.container_id}
    assert all(not (replay_inbox / f'source-{suffix}.hdf5').exists() for suffix in ('a', 'b', 'c'))

    with buffer.lease_quantum(19, 4_096, world_size=1) as lease:
        buffer.decode_partition(lease.partitions[0], global_step=19)
    statistics = buffer.last_decode_statistics
    assert statistics.payload_open_count == 1
    assert statistics.selected_rows == 4_096
    assert statistics.rows_read <= 105_000
    assert statistics.bytes_read >= statistics.selected_bytes

    first_container_path = replay_inbox / next(
        payload.hdf5_file_name
        for payload in state.physical_payloads
        if payload.kind is ReplayPayloadKind.COMPACTED_CONTAINER
    )
    commit_large_shard(replay_inbox, 'source-d', 60_000, 4.0)
    commit_large_shard(replay_inbox, 'source-e', 40_000, 5.0)
    second_ingest = buffer.discover_committed_shards()
    second_compaction = buffer.compact_one_idle_container()

    assert second_ingest.presentation_credits == 100_000 * 4
    assert second_compaction.status is CompactionStepStatus.COMMITTED_CONTAINER
    assert second_compaction.container_id != compaction.container_id
    assert buffer.compacted_container_count == 2
    assert first_container_path.exists()
    assert buffer.discover_committed_shards().presentation_credits == 0


def test_compaction_keeps_old_lease_payloads_and_fifo_retires_logical_ranges(
    tmp_path: Path,
) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit_large_shard(replay_inbox, 'source-a', 40_000, 1.0)
    commit_large_shard(replay_inbox, 'source-b', 40_000, 2.0)
    commit_large_shard(replay_inbox, 'source-c', 25_000, 3.0)
    index_path = tmp_path / 'index.json'
    buffer = RollingReplayBuffer(replay_inbox, index_path, sampler_seed=19)
    buffer.discover_committed_shards()
    old_lease = buffer.lease_quantum(0, 105_000, world_size=1)
    old_partition = old_lease.partitions[0]

    result = buffer.compact_one_idle_container()

    assert result.container_id is not None
    assert all((replay_inbox / f'source-{suffix}.hdf5').exists() for suffix in ('a', 'b', 'c'))
    with buffer.lease_quantum(1, 1_024, world_size=1) as new_lease:
        assert {reference.physical_payload_id for reference in new_lease.partitions[0].references} == {
            result.container_id
        }
    old_subset = type(old_partition)(rank=0, references=old_partition.references[:8])
    assert len(buffer.decode_partition(old_subset, global_step=0)) == 8

    old_lease.release()

    assert all(not (replay_inbox / f'source-{suffix}.hdf5').exists() for suffix in ('a', 'b', 'c'))
    with pytest.raises(RuntimeError, match='Stale replay reference'):
        buffer.decode_partition(old_subset, global_step=0)
    container_path = replay_inbox / f'containers/{result.container_id}.hdf5'
    assert container_path.exists()
    reduced = RollingReplayBuffer(replay_inbox, index_path, capacity=65_000)
    assert reduced.unique_sample_count == 65_000
    assert tuple(segment.segment_id for segment in reduced._state.live_segments) == ('source-b', 'source-c')
    assert container_path.exists()
    smallest = RollingReplayBuffer(replay_inbox, index_path, capacity=25_000)
    assert smallest.unique_sample_count == 25_000
    assert tuple(segment.segment_id for segment in smallest._state.live_segments) == ('source-c',)
    assert container_path.exists()
    empty = RollingReplayBuffer(replay_inbox, index_path, capacity=1)
    assert empty.unique_sample_count == 0
    assert not container_path.exists()


def test_interrupted_compaction_is_rolled_back_without_losing_sources(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit_large_shard(replay_inbox, 'source-a', COMPACTION_TARGET_POSITIONS, 1.0)
    index_path = tmp_path / 'index.json'
    buffer = RollingReplayBuffer(replay_inbox, index_path)
    buffer.discover_committed_shards()
    state = RollingReplayIndexState.model_validate_json(index_path.read_text(encoding='utf-8'))
    plan = ActiveCompactionPlan(
        container_id='container-interrupted',
        source_segment_ids=('source-a',),
        total_rows=COMPACTION_TARGET_POSITIONS,
        temporary_hdf5_file_name='tmp/.container-interrupted.hdf5.tmp',
        final_hdf5_file_name='containers/container-interrupted.hdf5',
        manifest_file_name='containers/container-interrupted.container.json',
    )
    interrupted_state = RollingReplayIndexState(
        schema_version=state.schema_version,
        sampler_seed=state.sampler_seed,
        live_segments=state.live_segments,
        physical_payloads=state.physical_payloads,
        retired_payload_ids=state.retired_payload_ids,
        active_compaction=plan,
    )
    index_path.write_text(interrupted_state.model_dump_json(), encoding='utf-8')
    for relative_path in (
        plan.temporary_hdf5_file_name,
        plan.final_hdf5_file_name,
        plan.manifest_file_name,
    ):
        path = replay_inbox / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'interrupted')

    recovered = RollingReplayBuffer(replay_inbox, index_path)

    recovered_state = RollingReplayIndexState.model_validate_json(index_path.read_text(encoding='utf-8'))
    assert recovered_state.active_compaction is None
    assert recovered.unique_sample_count == COMPACTION_TARGET_POSITIONS
    assert (replay_inbox / 'source-a.hdf5').exists()
    assert all(
        not (replay_inbox / relative_path).exists()
        for relative_path in (
            plan.temporary_hdf5_file_name,
            plan.final_hdf5_file_name,
            plan.manifest_file_name,
        )
    )


def test_rank_quantum_batches_are_ordered_zero_copy_views(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'source-a', 64)
    buffer = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=41)
    buffer.discover_committed_shards()
    request = ReplayQuantumRequest(
        global_step=8,
        optimizer_steps=4,
        global_batch_size=8,
        world_size=2,
        rank=1,
    )

    quantum = decode_rank_quantum(buffer, request)
    batches = tuple(quantum.optimizer_batches())

    assert len(quantum.full_batch) == 16
    assert len(batches) == 4
    assert all(len(batch) == 4 for batch in batches)
    full_storage = quantum.full_batch.states.untyped_storage().data_ptr()
    assert all(batch.states.untyped_storage().data_ptr() == full_storage for batch in batches)
    torch.testing.assert_close(
        torch.cat(tuple(batch.plies for batch in batches)),
        quantum.full_batch.plies,
    )
    assert buffer.last_decode_statistics.payload_open_count == 1


def test_read_ranks_refresh_only_between_quanta(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='does not exist for a read-only rank'):
        RollingReplayBuffer(
            tmp_path / 'missing-inbox',
            tmp_path / 'missing-index.json',
            read_only=True,
        )

    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'source-a', 16)
    index_path = tmp_path / 'index.json'
    writer = RollingReplayBuffer(replay_inbox, index_path)
    writer.discover_committed_shards()
    reader = RollingReplayBuffer(replay_inbox, index_path, read_only=True)
    lease = reader.lease_quantum(0, 8, world_size=1)
    commit(replay_inbox, 'source-b', 16, creation_offset=1.0)
    writer.discover_committed_shards()

    with pytest.raises(RuntimeError, match='during an active quantum'):
        reader.refresh_index_for_read()
    lease.release()
    reader.refresh_index_for_read()

    assert reader.unique_sample_count == 32
    with pytest.raises(RuntimeError, match='Read-only replay ranks cannot'):
        reader.compact_one_idle_container()


def test_coordinated_four_process_benchmark_decodes_all_rank_quanta(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'source-a', 512)
    replay_buffer = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=9)
    replay_buffer.discover_committed_shards()
    arguments = Arguments(
        workspace=tmp_path,
        output=tmp_path / 'result.json',
        shard_count=1,
        samples_per_shard=512,
        global_batch_size=64,
        world_size=4,
        optimizer_steps=2,
        quantum_count=2,
        trainer_consumption_samples_per_second=1.0,
        sampler_seed=9,
    )

    result = benchmark_coordinated_processes(replay_buffer, arguments)

    assert result.decoded_samples == 256
    assert tuple(process.rank for process in result.process_results) == (0, 1, 2, 3)
    assert all(process.decoded_samples == 64 for process in result.process_results)
    assert result.selected_rows == 256
    assert result.rows_read >= result.selected_rows
    assert result.bytes_read >= result.selected_bytes
    assert result.row_read_amplification > 1.0
    assert result.byte_read_amplification > 1.0
    json.dumps(asdict(result))

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from src.Encoding import C, H, W
from src.games.chess.ChessGame import ChessGame
from src.self_play.SelfPlayDataset import ReplaySampleMetadata, SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.self_play.value_target import ReplayValueTarget, TerminationReason
from src.train.ReplayBuffer import (
    CHESS_MIRROR_ACTION_MAP,
    DEFAULT_REPLAY_CAPACITY,
    DiskReplayBuffer,
    ReplayBatchRequest,
    ReplayIndexState,
    _sample_is_mirrored,
    commit_replay_shard,
    prefetch_replay_batches,
)


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


def partition_signature(buffer: DiskReplayBuffer, global_step: int, sample_count: int) -> tuple[tuple[str, int], ...]:
    with buffer.lease_quantum(global_step, sample_count, world_size=1) as lease:
        return tuple((reference.shard_id, reference.sample_index) for reference in lease.partitions[0].references)


def test_unmanifested_and_temporary_shards_are_invisible(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    replay_inbox.mkdir()
    assert replay_dataset(4).save_to_path(replay_inbox / 'unmanifested.hdf5')
    (replay_inbox / '.partial.manifest.json.tmp').write_text('{}', encoding='utf-8')
    buffer = DiskReplayBuffer(replay_inbox, tmp_path / 'index.json')

    result = buffer.discover_committed_shards()

    assert result.unique_samples == 0
    assert buffer.unique_sample_count == 0


def test_manifest_is_typed_hashed_and_issues_four_credits_once(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'shard-a', 7)
    buffer = DiskReplayBuffer(replay_inbox, tmp_path / 'index.json')

    first = buffer.discover_committed_shards()
    second = buffer.discover_committed_shards()

    assert first.unique_samples == 7
    assert first.presentation_credits == 28
    assert second.unique_samples == 0
    assert second.presentation_credits == 0
    assert buffer.unique_sample_count == 7


def test_default_capacity_is_two_and_a_half_million_disk_backed_positions(tmp_path: Path) -> None:
    buffer = DiskReplayBuffer(tmp_path / 'inbox', tmp_path / 'index.json')

    assert buffer.capacity == DEFAULT_REPLAY_CAPACITY == 2_500_000
    assert buffer.unique_sample_count == 0
    assert buffer.metadata_memory_bytes <= 16


def test_sampler_is_reproducible_without_replacement_and_ddp_partitions_are_disjoint(
    tmp_path: Path,
) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'shard-a', 40)
    buffer = DiskReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=73)
    buffer.discover_committed_shards()

    with buffer.lease_quantum(global_step=11, global_sample_count=24, world_size=4) as lease:
        all_references = [
            (reference.shard_id, reference.sample_index)
            for partition in lease.partitions
            for reference in partition.references
        ]
        assert all(len(partition.references) == 6 for partition in lease.partitions)
        assert len(all_references) == len(set(all_references))
        first_signature = tuple(all_references)

    restarted = DiskReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=999)
    with restarted.lease_quantum(global_step=11, global_sample_count=24, world_size=4) as lease:
        restarted_signature = tuple(
            (reference.shard_id, reference.sample_index)
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
    buffer = DiskReplayBuffer(replay_inbox, index_path, sampler_seed=91)
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
        signature = tuple((reference.shard_id, reference.sample_index) for reference in references)
        for offset in range(0, len(references), 1_024):
            batch = references[offset : offset + 1_024]
            identities = {(reference.shard_id, reference.sample_index) for reference in batch}
            assert len(identities) == 1_024
        assert len(set(signature)) < len(signature)

    restarted = DiskReplayBuffer(replay_inbox, index_path)
    with restarted.lease_quantum(
        global_step=4,
        global_sample_count=5_120,
        world_size=4,
        global_batch_size=1_024,
    ) as lease:
        restarted_signature = tuple(
            (reference.shard_id, reference.sample_index)
            for reference in sorted(
                (reference for partition in lease.partitions for reference in partition.references),
                key=lambda reference: reference.global_batch_position,
            )
        )
    assert restarted_signature == signature


def test_buffer_smaller_than_global_batch_fails_clearly(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'shard-a', 100)
    buffer = DiskReplayBuffer(replay_inbox, tmp_path / 'index.json')
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
    buffer = DiskReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=19)
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
    buffer = DiskReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=31)
    buffer.discover_committed_shards()

    with buffer.lease_quantum(global_step=8, global_sample_count=8, world_size=2) as lease:
        first = buffer.decode_partition(lease.partitions[1], global_step=8)
    restarted = DiskReplayBuffer(replay_inbox, tmp_path / 'index.json')
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
    buffer = DiskReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=sampler_seed)
    buffer.discover_committed_shards()

    with buffer.lease_quantum(global_step, global_sample_count=1, world_size=1) as lease:
        batch = buffer.decode_partition(lease.partitions[0], global_step)

    expected_state = torch.from_numpy(np.flip(state, axis=2).copy()).to(dtype=torch.float32)
    torch.testing.assert_close(batch.states[0], expected_state)
    assert int(torch.argmax(batch.policy_targets[0]).item()) == mirrored_action


def test_fifo_eviction_waits_for_active_lease_then_reuses_capacity(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'old', 4)
    buffer = DiskReplayBuffer(replay_inbox, tmp_path / 'index.json', capacity=4)
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
    buffer = DiskReplayBuffer(replay_inbox, index_path, capacity=10, sampler_seed=44)
    buffer.discover_committed_shards()
    state = ReplayIndexState.model_validate_json(index_path.read_text(encoding='utf-8'))
    live_manifest = next(manifest for manifest in state.live_shards if manifest.shard_id == 'live')
    interrupted_state = ReplayIndexState(
        schema_version=state.schema_version,
        sampler_seed=state.sampler_seed,
        live_shards=(live_manifest,),
        evicted_shard_ids=('old',),
    )
    index_path.write_text(interrupted_state.model_dump_json(), encoding='utf-8')

    recovered = DiskReplayBuffer(replay_inbox, index_path, capacity=10)

    assert recovered.unique_sample_count == 5
    assert recovered.shard_count == 1
    assert not (replay_inbox / 'old.hdf5').exists()
    assert not (replay_inbox / 'old.manifest.json').exists()
    recovered_state = ReplayIndexState.model_validate_json(index_path.read_text(encoding='utf-8'))
    assert recovered_state.evicted_shard_ids == ()


def test_index_metadata_memory_scales_with_shards_not_position_payloads(tmp_path: Path) -> None:
    small_inbox = tmp_path / 'small'
    large_inbox = tmp_path / 'large'
    commit(small_inbox, 'small', 1)
    commit(large_inbox, 'large', 2_000)
    small = DiskReplayBuffer(small_inbox, tmp_path / 'small-index.json')
    large = DiskReplayBuffer(large_inbox, tmp_path / 'large-index.json')
    small.discover_committed_shards()
    large.discover_committed_shards()

    assert small.shard_count == large.shard_count == 1
    assert abs(large.metadata_memory_bytes - small.metadata_memory_bytes) < 128
    assert (large_inbox / 'large.hdf5').stat().st_size > (small_inbox / 'small.hdf5').stat().st_size * 10


def test_prefetch_preserves_request_order_and_values(tmp_path: Path) -> None:
    replay_inbox = tmp_path / 'inbox'
    commit(replay_inbox, 'shard-a', 32)
    buffer = DiskReplayBuffer(replay_inbox, tmp_path / 'index.json', sampler_seed=81)
    buffer.discover_committed_shards()
    requests = tuple(
        ReplayBatchRequest(
            global_step=global_step,
            global_sample_count=8,
            world_size=2,
            rank=1,
        )
        for global_step in range(4)
    )
    expected_plies = []
    for request in requests:
        with buffer.lease_quantum(
            request.global_step,
            request.global_sample_count,
            request.world_size,
        ) as lease:
            expected_plies.append(
                buffer.decode_partition(
                    lease.partitions[request.rank],
                    request.global_step,
                ).plies
            )

    batches = tuple(prefetch_replay_batches(buffer, requests, maximum_prefetched_batches=2))

    assert len(batches) == len(requests)
    for batch, expected in zip(batches, expected_plies):
        torch.testing.assert_close(batch.plies, expected)

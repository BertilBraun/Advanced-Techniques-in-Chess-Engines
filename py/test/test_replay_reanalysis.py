from pathlib import Path

import numpy as np
import pytest

from src.Encoding import C, H, W
from src.self_play.SelfPlayDataset import ReplaySampleMetadata, SelfPlayDataset
from src.self_play.value_target import ReplayValueTarget, TerminationReason
from src.train.ReplayReanalysis import (
    ReanalysisTarget,
    latest_reanalysis_overrides,
    write_reanalysis_sidecar,
)
from src.train.RollingReplayBuffer import RollingReplayBuffer, commit_replay_shard, file_sha256


def target(row_index: int, move: int, root_value: float) -> ReanalysisTarget:
    return ReanalysisTarget(
        row_index=row_index,
        visit_counts=np.asarray(((move, 600),), dtype=np.uint16),
        mcts_root_value=root_value,
    )


def test_reanalysis_sidecars_fold_disjoint_rows_and_newest_repeated_row(tmp_path: Path) -> None:
    source = tmp_path / 'source.hdf5'
    source.write_bytes(b'immutable replay payload')
    source_hash = file_sha256(source)

    write_reanalysis_sidecar(source, 10, (target(2, 20, 0.2), target(4, 40, 0.4)))
    write_reanalysis_sidecar(source, 11, (target(3, 30, 0.3), target(4, 41, 0.5)))

    overrides = latest_reanalysis_overrides(source, source_hash)

    assert set(overrides) == {2, 3, 4}
    assert int(overrides[4][0][0, 0]) == 41
    assert overrides[4][1] == pytest.approx(0.5)


def test_reanalysis_sidecar_rejects_different_source_identity(tmp_path: Path) -> None:
    source = tmp_path / 'source.hdf5'
    source.write_bytes(b'first payload')
    write_reanalysis_sidecar(source, 10, (target(0, 1, 0.1),))
    source.write_bytes(b'replaced payload')

    with pytest.raises(ValueError, match='source identity mismatch'):
        latest_reanalysis_overrides(source, file_sha256(source))


def test_reanalysis_sidecar_updates_sampled_search_targets_without_outcome_or_credits(
    tmp_path: Path,
) -> None:
    replay_inbox = tmp_path / 'replay'
    dataset = SelfPlayDataset()
    dataset.add_sample(
        state=np.zeros((C, H, W), dtype=np.int8),
        visit_counts=[(1, 600)],
        value_target=ReplayValueTarget.from_scores(1.0, 0.2, TerminationReason.NATURAL),
        sample_metadata=ReplaySampleMetadata(
            ply=2,
            current_player_piece_count=16,
            opponent_piece_count=16,
        ),
    )
    manifest = commit_replay_shard(dataset, replay_inbox, 0, 1, 1, shard_id='source')
    source = replay_inbox / manifest.hdf5_file_name
    write_reanalysis_sidecar(source, 2, (target(0, 7, -0.4),))
    replay = RollingReplayBuffer(replay_inbox, tmp_path / 'index.json')

    ingest = replay.discover_committed_shards()
    with replay.lease_quantum(0, 1, 1) as lease:
        batch = replay.decode_partition(lease.partitions[0], global_step=0)

    assert ingest.presentation_credits == 4
    assert int(batch.final_outcomes[0]) == 0
    assert float(batch.mcts_root_values[0]) == pytest.approx(-0.4)
    assert int(np.argmax(batch.policy_targets[0].numpy())) == 7

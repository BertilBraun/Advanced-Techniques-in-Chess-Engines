from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from src.az.experiment.commit_journal import ReplayCommitJournal
from src.az.replay.envelope import ReplayRecord, derive_self_play_seed_lineage
from src.az.self_play.scheduling import LogicalWorkerGameScheduler
from test.unit.go_stage5_helpers import envelope


def _record(index: int) -> ReplayRecord:
    return ReplayRecord(
        envelope=envelope(index).model_copy(
            update={
                'sample_id': UUID(int=1_000 + index),
                'replay_credit_id': UUID(int=2_000 + index),
            }
        ),
        payload=b'payload',
    )


def test_replay_commit_journal_repairs_torn_tail(tmp_path: Path) -> None:
    path = (tmp_path / 'commits.azc').resolve()
    journal = ReplayCommitJournal(path)
    journal.commit((_record(1),))
    first_size = path.stat().st_size
    journal.commit((_record(2),))
    with path.open('r+b') as stream:
        stream.truncate(path.stat().st_size - 3)

    recovered = ReplayCommitJournal(path)

    assert recovered.sample_ids == frozenset({UUID(int=1_001)})
    assert path.stat().st_size == first_size


def test_replay_commit_journal_rejects_committed_frame_corruption(tmp_path: Path) -> None:
    path = (tmp_path / 'commits.azc').resolve()
    ReplayCommitJournal(path).commit((_record(1),))
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)

    with pytest.raises(ValueError, match='checksum'):
        ReplayCommitJournal(path)


def test_commit_history_resumes_each_logical_worker_without_identity_reuse(tmp_path: Path) -> None:
    path = (tmp_path / 'commits.azc').resolve()
    records = tuple(
        ReplayRecord(
            envelope=_record(worker_index + 1).envelope.model_copy(
                update={
                    'seed_lineage': derive_self_play_seed_lineage(
                        root_seed=123,
                        process_index=0,
                        worker_index=worker_index,
                        game_index=game_index,
                        ply=worker_index + 1,
                    )
                }
            ),
            payload=b'payload',
        )
        for worker_index, game_index in ((0, 4), (1, 8))
    )
    journal = ReplayCommitJournal(path)
    journal.commit(records)

    scheduler = LogicalWorkerGameScheduler(0, 2, journal.next_game_indices(2))
    resumed = tuple(scheduler.next_game() for _ in range(2))

    assert tuple((game.logical_worker_index, game.game_index) for game in resumed) == (
        (0, 5),
        (1, 9),
    )

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from src.az.replay.credits import ReplayCreditJournal, ReplayCreditState
from src.az.replay.envelope import ReplayEnvelope, ReplayRecord
from src.az.replay.sampling import DeterministicReplaySampler, ReplaySamplerState
from test.unit.go_stage5_helpers import envelope


def _records(count: int) -> tuple[ReplayRecord, ...]:
    return tuple(ReplayRecord(envelope=record_envelope, payload=b'payload') for record_envelope in _envelopes(count))


def _envelopes(count: int) -> Iterator[ReplayEnvelope]:
    for index in range(1, count + 1):
        original = envelope(index)
        yield original.model_copy(
            update={
                'sample_id': UUID(int=100 + index),
                'replay_credit_id': UUID(int=200 + index),
            }
        )


def test_fractional_replay_credits_are_exact_and_carry_forward(tmp_path: Path) -> None:
    journal = ReplayCreditJournal(tmp_path / 'credit-identities.bin')
    journal.credit_shard(0, tuple(UUID(int=index) for index in range(1, 6)))
    state = ReplayCreditState.initial().reconcile(journal.snapshot, Decimal('1.5'))
    published = state.prepare_training_quantum(optimizer_steps=1, global_batch_size=4, maximum_optimizer_steps=10)

    assert published.earned_position_credits == Decimal('7.5')
    assert published.consumed_position_credits == Decimal(4)
    assert published.available_position_credits == Decimal('3.5')


def test_credit_journal_counts_unique_identities_across_restart(tmp_path: Path) -> None:
    path = tmp_path / 'credit-identities.bin'
    journal = ReplayCreditJournal(path)

    assert journal.credit_shard(2, (UUID(int=2), UUID(int=1))) == 2
    restarted = ReplayCreditJournal(path)
    assert restarted.credit_shard(2, (UUID(int=2), UUID(int=1))) == 2
    assert restarted.credit_shard(3, (UUID(int=3),)) == 3
    assert ReplayCreditJournal(path).credited_unique_positions == 3


def test_credit_journal_recovers_a_torn_final_record(tmp_path: Path) -> None:
    path = tmp_path / 'credit-identities.bin'
    ReplayCreditJournal(path).credit_shard(1, (UUID(int=1),))
    complete_size = path.stat().st_size
    with path.open('ab') as stream:
        stream.write(b'torn')

    recovered = ReplayCreditJournal(path)

    assert recovered.credited_unique_positions == 1
    assert path.stat().st_size == complete_size


def test_credit_journal_rejects_corrupt_complete_record(tmp_path: Path) -> None:
    path = tmp_path / 'credit-identities.bin'
    ReplayCreditJournal(path).credit_shard(1, (UUID(int=1),))
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)

    with pytest.raises(ValueError, match='checksum'):
        ReplayCreditJournal(path)


def test_credit_snapshot_rejects_a_different_same_sized_identity_prefix(tmp_path: Path) -> None:
    first = ReplayCreditJournal(tmp_path / 'first.bin')
    second = ReplayCreditJournal(tmp_path / 'second.bin')
    first.credit_shard(0, (UUID(int=1), UUID(int=2)))
    second.credit_shard(0, (UUID(int=1), UUID(int=3)))

    with pytest.raises(ValueError, match='prefix'):
        second.verify_snapshot(first.snapshot)

    checkpoint_snapshot = first.snapshot
    first.credit_shard(1, (UUID(int=4),))
    first.verify_snapshot(checkpoint_snapshot)


def test_shard_credit_history_rejects_conflicts_and_cross_shard_identity_reuse(tmp_path: Path) -> None:
    journal = ReplayCreditJournal(tmp_path / 'credit-identities.bin')
    original = (UUID(int=1), UUID(int=2))
    journal.credit_shard(4, original)

    assert journal.credit_shard(4, original) == 2
    with pytest.raises(ValueError, match='conflicts'):
        journal.credit_shard(4, (UUID(int=1), UUID(int=3)))
    with pytest.raises(ValueError, match='another shard'):
        journal.credit_shard(5, (UUID(int=2), UUID(int=3)))


def test_sampling_is_order_independent_and_resumes_at_exact_next_step() -> None:
    records = _records(5)
    first = DeterministicReplaySampler(123, 0, ReplaySamplerState(next_optimizer_step=7))
    first_result = first.sample(records, 8)
    resumed = DeterministicReplaySampler(123, 0, first.state)

    reordered = DeterministicReplaySampler(123, 0, ReplaySamplerState(next_optimizer_step=7))
    reordered_result = reordered.sample(tuple(reversed(records)), 8)
    uninterrupted_next = first.sample(records, 8)
    resumed_next = resumed.sample(records, 8)

    assert tuple(record.envelope.sample_id for record in first_result.records) == tuple(
        record.envelope.sample_id for record in reordered_result.records
    )
    assert first_result.augmentation_seeds == reordered_result.augmentation_seeds
    assert tuple(record.envelope.sample_id for record in uninterrupted_next.records) == tuple(
        record.envelope.sample_id for record in resumed_next.records
    )
    assert uninterrupted_next.augmentation_seeds == resumed_next.augmentation_seeds

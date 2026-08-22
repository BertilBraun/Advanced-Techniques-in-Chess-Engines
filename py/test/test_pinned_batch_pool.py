from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from typing import cast

import pytest
import torch
from src.replay import pinned_batch_pool
from src.replay.batch_loader import MappedReplayBatchLoader, PrefetchedReplayBatches
from src.replay.pinned_batch_pool import (
    PinnedBatchSlotPool,
    PinnedSlotState,
    record_training_batch_stream,
)
from src.training.batch import TrainingBatch


def _batch(value: float) -> TrainingBatch:
    return TrainingBatch(
        states=torch.full((2, 2), value),
        policy_targets=torch.full((2, 3), value),
        policy_legal_action_ids=torch.zeros((2, 2), dtype=torch.int64),
        wdl_targets=torch.full((2, 3), value),
        root_values=torch.full((2,), value),
        auxiliary_targets=(torch.full((2, 1), value),),
        auxiliary_legal_action_ids=(torch.full((2, 2), -1, dtype=torch.int64),),
        auxiliary_eligibility=(torch.ones(2, dtype=torch.bool),),
        sample_weights=torch.full((2,), value),
        source_model_generations=torch.zeros(2, dtype=torch.int64),
        source_created_at_seconds=torch.full((2,), value, dtype=torch.float64),
    )


def _unpinned_empty_batch(source: TrainingBatch) -> TrainingBatch:
    return TrainingBatch(
        states=torch.empty_like(source.states),
        policy_targets=torch.empty_like(source.policy_targets),
        policy_legal_action_ids=torch.empty_like(source.policy_legal_action_ids),
        wdl_targets=torch.empty_like(source.wdl_targets),
        root_values=torch.empty_like(source.root_values),
        auxiliary_targets=tuple(torch.empty_like(target) for target in source.auxiliary_targets),
        auxiliary_legal_action_ids=tuple(torch.empty_like(actions) for actions in source.auxiliary_legal_action_ids),
        auxiliary_eligibility=tuple(torch.empty_like(mask) for mask in source.auxiliary_eligibility),
        sample_weights=torch.empty_like(source.sample_weights),
        source_model_generations=torch.empty_like(source.source_model_generations),
        source_created_at_seconds=torch.empty_like(source.source_created_at_seconds),
    )


class _FakeCudaEvent:
    def __init__(self, failure: BaseException | None = None, complete: bool = True) -> None:
        self.failure = failure
        self.complete = complete
        self.query_count = 0
        self.synchronize_count = 0

    def query(self) -> bool:
        self.query_count += 1
        return self.complete

    def synchronize(self) -> None:
        self.synchronize_count += 1
        if self.failure is not None:
            raise self.failure
        self.complete = True


@pytest.fixture
def cpu_slot_allocations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pinned_batch_pool, 'allocate_pinned_batch_like', _unpinned_empty_batch)


def _cuda_event(event: _FakeCudaEvent) -> torch.cuda.Event:
    return cast(torch.cuda.Event, event)


def test_slot_moves_through_explicit_states_and_reuses_storage(cpu_slot_allocations: None) -> None:
    del cpu_slot_allocations
    pool = PinnedBatchSlotPool(1)
    first = pool.fill(_batch(1.0))
    assert pool.states == (PinnedSlotState.READY,)
    assert first.batch is not None
    storage_pointer = first.batch.states.data_ptr()
    event = _FakeCudaEvent()
    pool.mark_transfer_in_flight(first, _cuda_event(event))
    assert pool.states == (PinnedSlotState.TRANSFER_IN_FLIGHT,)

    second = pool.fill(_batch(2.0))

    assert second is first
    assert second.batch is not None
    assert second.batch.states.data_ptr() == storage_pointer
    assert torch.equal(second.batch.states, torch.full((2, 2), 2.0))
    assert event.query_count >= 1
    assert event.synchronize_count == 0
    assert pool.states == (PinnedSlotState.READY,)
    pool.release_untransferred(second)
    assert pool.states == (PinnedSlotState.REUSABLE,)
    pool.close()
    pool.close()


@pytest.mark.parametrize('depth', (1, 2, 4, 8))
def test_allocation_count_is_bounded_by_prefetch_depth(depth: int, cpu_slot_allocations: None) -> None:
    del cpu_slot_allocations
    pool = PinnedBatchSlotPool(depth)
    for index in range(depth * 3):
        slot = pool.fill(_batch(float(index)))
        pool.mark_transfer_in_flight(slot, _cuda_event(_FakeCudaEvent()))

    assert pool.allocation_count == depth
    pool.close()


def test_pool_reclaims_completed_transfer_without_synchronizing(cpu_slot_allocations: None) -> None:
    del cpu_slot_allocations
    pool = PinnedBatchSlotPool(3)
    events = (_FakeCudaEvent(complete=False), _FakeCudaEvent(), _FakeCudaEvent(complete=False))
    slots = tuple(pool.fill(_batch(float(index))) for index in range(3))
    for slot, event in zip(slots, events, strict=True):
        pool.mark_transfer_in_flight(slot, _cuda_event(event))

    reused = pool.fill(_batch(4.0))

    assert reused is slots[1]
    assert events[1].synchronize_count == 0
    pool.release_untransferred(reused)
    pool.close()


def test_pool_waits_on_oldest_submission_instead_of_lowest_slot(cpu_slot_allocations: None) -> None:
    del cpu_slot_allocations
    pool = PinnedBatchSlotPool(3)
    first_events = tuple(_FakeCudaEvent(complete=False) for _ in range(3))
    slots = tuple(pool.fill(_batch(float(index))) for index in range(3))
    for slot, event in zip(slots, first_events, strict=True):
        pool.mark_transfer_in_flight(slot, _cuda_event(event))

    first_reuse = pool.fill(_batch(3.0))
    replacement = _FakeCudaEvent(complete=False)
    pool.mark_transfer_in_flight(first_reuse, _cuda_event(replacement))
    second_reuse = pool.fill(_batch(4.0))

    assert first_reuse is slots[0]
    assert second_reuse is slots[1]
    assert first_events[0].synchronize_count == 1
    assert first_events[1].synchronize_count == 1
    assert replacement.synchronize_count == 0
    pool.release_untransferred(second_reuse)
    pool.close()


def test_fill_failure_releases_slot_for_reuse(
    monkeypatch: pytest.MonkeyPatch,
    cpu_slot_allocations: None,
) -> None:
    del cpu_slot_allocations
    pool = PinnedBatchSlotPool(1)
    copy_error = ValueError('forced pinned fill failure')

    def fail_copy(source: TrainingBatch, destination: TrainingBatch) -> None:
        del source, destination
        raise copy_error

    monkeypatch.setattr(pinned_batch_pool, 'copy_training_batch', fail_copy)
    with pytest.raises(ValueError, match='forced pinned fill failure') as raised:
        pool.fill(_batch(1.0))

    assert raised.value is copy_error
    assert pool.states == (PinnedSlotState.REUSABLE,)
    pool.close()


def test_transfer_failure_is_preserved_by_reuse_and_close(cpu_slot_allocations: None) -> None:
    del cpu_slot_allocations
    pool = PinnedBatchSlotPool(1)
    slot = pool.fill(_batch(1.0))
    transfer_error = RuntimeError('forced asynchronous transfer failure')
    pool.mark_transfer_in_flight(slot, _cuda_event(_FakeCudaEvent(transfer_error, complete=False)))

    with pytest.raises(RuntimeError, match='forced asynchronous transfer failure') as raised:
        pool.fill(_batch(2.0))
    assert raised.value is transfer_error

    with pytest.raises(RuntimeError, match='Pinned batch slot cleanup failed') as cleanup:
        pool.close()
    assert cleanup.value.__cause__ is transfer_error


def test_unrecoverable_transfer_is_never_reused(cpu_slot_allocations: None) -> None:
    del cpu_slot_allocations
    pool = PinnedBatchSlotPool(1)
    slot = pool.fill(_batch(1.0))
    pool.mark_unrecoverable(slot)

    with pytest.raises(RuntimeError, match='no transferable slot'):
        pool.fill(_batch(2.0))
    with pytest.raises(RuntimeError, match='cleanup failed'):
        pool.close()


def test_failed_copy_record_and_stream_sync_preserve_primary_error_and_poison_slot(
    monkeypatch: pytest.MonkeyPatch,
    cpu_slot_allocations: None,
) -> None:
    del cpu_slot_allocations
    transfer_error = ValueError('forced H2D failure')
    synchronization_error = RuntimeError('forced stream synchronization failure')

    class _FailedRecordEvent:
        def record(self, stream: object) -> None:
            del stream
            raise RuntimeError('forced event record failure')

    class _FailedStream:
        def synchronize(self) -> None:
            raise synchronization_error

    batches = PrefetchedReplayBatches.__new__(PrefetchedReplayBatches)
    batches.uses_cuda = True
    batches.device = torch.device('cuda')
    batches._prepared_batches = iter((_batch(1.0),))
    batches._transfer_stream = cast(torch.cuda.Stream, _FailedStream())
    batches._pinned_slots = PinnedBatchSlotPool(1)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, 'Event', _FailedRecordEvent)

    def fail_transfer(self: TrainingBatch, device: torch.device, non_blocking: bool) -> TrainingBatch:
        del self, device, non_blocking
        raise transfer_error

    monkeypatch.setattr(TrainingBatch, 'to_device', fail_transfer)

    with pytest.raises(ValueError, match='forced H2D failure') as raised:
        batches._prepare_next()

    assert raised.value is transfer_error
    assert raised.value.__cause__ is synchronization_error
    assert batches._pinned_slots.states == (PinnedSlotState.UNRECOVERABLE,)
    with pytest.raises(RuntimeError, match='cleanup failed'):
        batches._pinned_slots.close()


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable.')
@pytest.mark.parametrize('depth', (1, 2, 4, 8))
def test_cuda_slots_stay_pinned_until_nonblocking_transfer_completes(depth: int) -> None:
    device = torch.device('cuda')
    transfer_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device)
    pool = PinnedBatchSlotPool(depth)
    outputs: list[torch.Tensor] = []
    for index in range(depth * 2):
        slot = pool.fill(_batch(float(index)))
        assert slot.batch is not None
        assert slot.batch.states.is_pinned()
        with torch.cuda.stream(transfer_stream):
            device_batch = slot.batch.to_device(device, non_blocking=True)
            complete = torch.cuda.Event()
            complete.record(transfer_stream)
        pool.mark_transfer_in_flight(slot, complete)
        current_stream.wait_event(complete)
        record_training_batch_stream(device_batch, current_stream)
        outputs.append(device_batch.states)

    current_stream.synchronize()
    assert [float(output[0, 0].item()) for output in outputs] == [float(index) for index in range(depth * 2)]
    assert pool.allocation_count == depth
    pool.close()


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable.')
@pytest.mark.parametrize('depth', (1, 2, 4, 8))
def test_cuda_prefetch_iterator_preserves_order_and_device_lifetime(depth: int) -> None:
    class _CudaLoader:
        pin_memory = True

        def _prepared_batches(self) -> Iterator[TrainingBatch]:
            for index in range(depth * 2):
                yield _batch(float(index))

    device = torch.device('cuda')
    consumer_stream = torch.cuda.Stream(device=device)
    loader = cast(MappedReplayBatchLoader[object], _CudaLoader())
    batches = PrefetchedReplayBatches(loader, device, uses_cuda=True, depth=depth)
    outputs: list[torch.Tensor] = []
    with torch.cuda.stream(consumer_stream), batches:
        for batch in batches:
            outputs.append(batch.states.square())
            del batch
    consumer_stream.synchronize()

    assert [float(output[0, 0].item()) for output in outputs] == [float(index**2) for index in range(depth * 2)]
    assert batches._pinned_slots is not None
    assert batches._pinned_slots.allocation_count == depth

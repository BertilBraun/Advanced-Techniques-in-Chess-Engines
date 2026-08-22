from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock

import torch
from src.training.batch import TrainingBatch


class PinnedSlotState(Enum):
    FREE = 'free'
    FILLING = 'filling'
    READY = 'ready'
    TRANSFER_IN_FLIGHT = 'transfer_in_flight'
    REUSABLE = 'reusable'
    UNRECOVERABLE = 'unrecoverable'


@dataclass
class PinnedBatchSlot:
    index: int
    state: PinnedSlotState = PinnedSlotState.FREE
    batch: TrainingBatch | None = None
    transfer_complete: torch.cuda.Event | None = None
    transfer_sequence: int | None = None


class PinnedBatchSlotPool:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError('Pinned batch slot capacity must be positive.')
        self._slots = [PinnedBatchSlot(index=index) for index in range(capacity)]
        self._lock = Lock()
        self._closed = False
        self._allocation_count = 0
        self._next_transfer_sequence = 0

    @property
    def capacity(self) -> int:
        return len(self._slots)

    @property
    def allocation_count(self) -> int:
        return self._allocation_count

    @property
    def states(self) -> tuple[PinnedSlotState, ...]:
        with self._lock:
            return tuple(slot.state for slot in self._slots)

    def fill(self, source: TrainingBatch) -> PinnedBatchSlot:
        slot = self._acquire()
        try:
            if slot.batch is None:
                slot.batch = allocate_pinned_batch_like(source)
                self._allocation_count += 1
            copy_training_batch(source, slot.batch)
        except BaseException:
            self.release_untransferred(slot)
            raise
        with self._lock:
            self._require_state(slot, PinnedSlotState.FILLING)
            slot.state = PinnedSlotState.READY
        return slot

    def mark_transfer_in_flight(
        self,
        slot: PinnedBatchSlot,
        transfer_complete: torch.cuda.Event,
    ) -> None:
        with self._lock:
            self._require_state(slot, PinnedSlotState.READY)
            slot.transfer_complete = transfer_complete
            slot.transfer_sequence = self._next_transfer_sequence
            self._next_transfer_sequence += 1
            slot.state = PinnedSlotState.TRANSFER_IN_FLIGHT

    def mark_unrecoverable(self, slot: PinnedBatchSlot) -> None:
        with self._lock:
            self._require_state(slot, PinnedSlotState.READY)
            slot.state = PinnedSlotState.UNRECOVERABLE

    def release_untransferred(self, slot: PinnedBatchSlot) -> None:
        with self._lock:
            if slot.state not in (PinnedSlotState.FILLING, PinnedSlotState.READY):
                raise RuntimeError(f'Cannot release pinned slot from state {slot.state.value}.')
            slot.transfer_complete = None
            slot.transfer_sequence = None
            slot.state = PinnedSlotState.REUSABLE

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            in_flight = tuple(slot for slot in self._slots if slot.state is PinnedSlotState.TRANSFER_IN_FLIGHT)
            unrecoverable = tuple(slot for slot in self._slots if slot.state is PinnedSlotState.UNRECOVERABLE)
        errors: list[BaseException] = []
        for slot in in_flight:
            try:
                self._complete_transfer(slot)
            except BaseException as error:
                errors.append(error)
        if unrecoverable:
            errors.append(RuntimeError('Pinned batch slot has an unrecoverable asynchronous transfer.'))
        if errors:
            raise RuntimeError('Pinned batch slot cleanup failed.') from errors[0]

    def _acquire(self) -> PinnedBatchSlot:
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError('Pinned batch slot pool is closed.')
                for slot in self._slots:
                    if slot.state in (PinnedSlotState.FREE, PinnedSlotState.REUSABLE):
                        slot.state = PinnedSlotState.FILLING
                        return slot
                in_flight = tuple(
                    sorted(
                        (slot for slot in self._slots if slot.state is PinnedSlotState.TRANSFER_IN_FLIGHT),
                        key=lambda slot: slot.transfer_sequence if slot.transfer_sequence is not None else -1,
                    )
                )
            for slot in in_flight:
                transfer_complete = slot.transfer_complete
                assert transfer_complete is not None
                if transfer_complete.query():
                    self._complete_transfer(slot, synchronize=False)
                    break
            else:
                if not in_flight:
                    raise RuntimeError('Pinned batch slot pool has no transferable slot.')
                self._complete_transfer(in_flight[0], synchronize=True)

    def _complete_transfer(self, slot: PinnedBatchSlot, synchronize: bool = True) -> None:
        with self._lock:
            self._require_state(slot, PinnedSlotState.TRANSFER_IN_FLIGHT)
            transfer_complete = slot.transfer_complete
        assert transfer_complete is not None
        if synchronize:
            transfer_complete.synchronize()
        elif not transfer_complete.query():
            raise RuntimeError('Pinned slot transfer was not complete when reclaimed.')
        with self._lock:
            self._require_state(slot, PinnedSlotState.TRANSFER_IN_FLIGHT)
            slot.transfer_complete = None
            slot.transfer_sequence = None
            slot.state = PinnedSlotState.REUSABLE

    @staticmethod
    def _require_state(slot: PinnedBatchSlot, expected: PinnedSlotState) -> None:
        if slot.state is not expected:
            raise RuntimeError(f'Pinned slot {slot.index} is {slot.state.value}, expected {expected.value}.')


def allocate_pinned_batch_like(source: TrainingBatch) -> TrainingBatch:
    return TrainingBatch(
        states=torch.empty_like(source.states, pin_memory=True),
        policy_targets=torch.empty_like(source.policy_targets, pin_memory=True),
        policy_legal_action_ids=torch.empty_like(source.policy_legal_action_ids, pin_memory=True),
        wdl_targets=torch.empty_like(source.wdl_targets, pin_memory=True),
        root_values=torch.empty_like(source.root_values, pin_memory=True),
        auxiliary_targets=tuple(torch.empty_like(target, pin_memory=True) for target in source.auxiliary_targets),
        auxiliary_legal_action_ids=tuple(
            torch.empty_like(actions, pin_memory=True) for actions in source.auxiliary_legal_action_ids
        ),
        auxiliary_eligibility=tuple(torch.empty_like(mask, pin_memory=True) for mask in source.auxiliary_eligibility),
        sample_weights=torch.empty_like(source.sample_weights, pin_memory=True),
        source_model_generations=torch.empty_like(source.source_model_generations, pin_memory=True),
        source_created_at_seconds=torch.empty_like(source.source_created_at_seconds, pin_memory=True),
    )


def copy_training_batch(source: TrainingBatch, destination: TrainingBatch) -> None:
    if len(source.auxiliary_targets) != len(destination.auxiliary_targets):
        raise ValueError('Pinned batch slot auxiliary layout changed.')
    destination.states.copy_(source.states)
    destination.policy_targets.copy_(source.policy_targets)
    destination.policy_legal_action_ids.copy_(source.policy_legal_action_ids)
    destination.wdl_targets.copy_(source.wdl_targets)
    destination.root_values.copy_(source.root_values)
    for source_target, destination_target in zip(source.auxiliary_targets, destination.auxiliary_targets, strict=True):
        destination_target.copy_(source_target)
    for source_actions, destination_actions in zip(
        source.auxiliary_legal_action_ids,
        destination.auxiliary_legal_action_ids,
        strict=True,
    ):
        destination_actions.copy_(source_actions)
    for source_mask, destination_mask in zip(
        source.auxiliary_eligibility,
        destination.auxiliary_eligibility,
        strict=True,
    ):
        destination_mask.copy_(source_mask)
    destination.sample_weights.copy_(source.sample_weights)
    destination.source_model_generations.copy_(source.source_model_generations)
    destination.source_created_at_seconds.copy_(source.source_created_at_seconds)


def record_training_batch_stream(batch: TrainingBatch, stream: torch.cuda.Stream) -> None:
    batch.states.record_stream(stream)
    batch.policy_targets.record_stream(stream)
    batch.policy_legal_action_ids.record_stream(stream)
    batch.wdl_targets.record_stream(stream)
    batch.root_values.record_stream(stream)
    for target in batch.auxiliary_targets:
        target.record_stream(stream)
    for actions in batch.auxiliary_legal_action_ids:
        actions.record_stream(stream)
    for mask in batch.auxiliary_eligibility:
        mask.record_stream(stream)
    batch.sample_weights.record_stream(stream)
    batch.source_model_generations.record_stream(stream)
    batch.source_created_at_seconds.record_stream(stream)

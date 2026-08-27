from __future__ import annotations

from collections import deque
from collections.abc import Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from types import TracebackType
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from src.games.contracts import GameStateContract
from src.games.representation import decode_packed_plane_bytes_into
from src.replay.columnar import (
    ReplayColumnViews,
    ReplayLegalMovesColumnViews,
    ReplayNextPolicyColumnViews,
    ReplayPolicyColumnViews,
    ReplayScalarColumnViews,
    ReplaySearchBudgetColumnViews,
)
from src.replay.description import ReplayDescription
from src.replay.layout import ReplayLayout
from src.replay.pinned_batch_pool import (
    PinnedBatchSlot,
    PinnedBatchSlotPool,
    record_training_batch_stream,
)
from src.replay.store import ReplayStore
from src.training.batch import TrainingBatch

PositionT = TypeVar('PositionT')

_transfer_streams: dict[int, torch.cuda.Stream] = {}
_transfer_streams_lock = Lock()


def shared_transfer_stream(device: torch.device) -> torch.cuda.Stream:
    # Cached for the process lifetime: the caching allocator segregates blocks by stream and cuBLAS keeps a workspace
    # per stream, so a stream per training quantum strands both and grows device memory every generation.
    device_index = torch.cuda.current_device() if device.index is None else device.index
    with _transfer_streams_lock:
        stream = _transfer_streams.get(device_index)
        if stream is None:
            stream = torch.cuda.Stream(device=torch.device('cuda', device_index))
            _transfer_streams[device_index] = stream
        return stream


@dataclass(frozen=True)
class _PrefetchedBatch:
    host_batch: TrainingBatch
    device_batch: TrainingBatch
    transfer_complete: torch.cuda.Event | None
    pinned_slot: PinnedBatchSlot | None


@dataclass(frozen=True)
class DenseTargetArrays:
    policy: npt.NDArray[np.float32]
    policy_legal_action_ids: npt.NDArray[np.int64]
    auxiliary: tuple[npt.NDArray[np.float32], ...]
    auxiliary_legal_action_ids: tuple[npt.NDArray[np.int64], ...]
    auxiliary_eligibility: tuple[npt.NDArray[np.bool_], ...]


class MappedReplayBatchLoader(Generic[PositionT]):
    def __init__(
        self,
        replay: ReplayDescription,
        state: GameStateContract[PositionT],
        source_optimizer_step: int,
        optimizer_steps: int,
        global_batch_size: int,
        world_size: int,
        rank: int,
        sampler_seed: int,
        pin_memory: bool,
    ) -> None:
        if optimizer_steps <= 0 or global_batch_size <= 0 or world_size <= 0:
            raise ValueError('Optimizer steps, global batch size, and world size must be positive.')
        if global_batch_size % world_size:
            raise ValueError('Global batch size must divide evenly over DDP ranks.')
        if not 0 <= rank < world_size:
            raise ValueError('DDP rank lies outside the configured world.')
        if replay.size < global_batch_size:
            raise ValueError('Replay must contain at least one global batch.')
        if replay.layout.targets.action_size != state.action_size:
            raise ValueError('Replay action count does not match the game contract.')
        self.replay = replay
        self.state = state
        self.source_optimizer_step = source_optimizer_step
        self.optimizer_steps = optimizer_steps
        self.global_batch_size = global_batch_size
        self.local_batch_size = global_batch_size // world_size
        self.rank = rank
        self.sampler_seed = sampler_seed
        self.pin_memory = pin_memory

    def __iter__(self) -> Iterator[TrainingBatch]:
        return self._prepared_batches()

    def prefetch(
        self,
        device: torch.device,
        uses_cuda: bool,
        depth: int,
    ) -> PrefetchedReplayBatches:
        if uses_cuda != (device.type == 'cuda'):
            raise ValueError('CUDA prefetch must agree with the training device type.')
        if depth <= 0:
            raise ValueError('Replay prefetch depth must be positive.')
        return PrefetchedReplayBatches(self, device, uses_cuda, depth)

    def _prepared_batches(self) -> Generator[TrainingBatch, None, None]:
        store = ReplayStore.open(self.replay.path, self.replay.layout, writable=False)
        try:
            state = store.state
            if (
                state.head != self.replay.head
                or state.size != self.replay.size
                or state.logical_capacity != self.replay.logical_capacity
            ):
                raise ValueError('Replay changed after the training description was captured.')
            generator = np.random.default_rng(np.random.SeedSequence((self.sampler_seed, self.source_optimizer_step)))
            for _ in range(self.optimizer_steps):
                global_sample_indices = generator.choice(
                    self.replay.size,
                    size=self.global_batch_size,
                    replace=False,
                )
                global_augmentation_indices = generator.integers(
                    0,
                    self.state.augmentation_count,
                    size=self.global_batch_size,
                )
                local_start = self.rank * self.local_batch_size
                local_stop = local_start + self.local_batch_size
                sample_indices = global_sample_indices[local_start:local_stop]
                augmentation_indices = global_augmentation_indices[local_start:local_stop]
                batch = build_training_batch(
                    store,
                    self.state,
                    sample_indices,
                    augmentation_indices,
                )
                yield batch
        finally:
            store.close()


class PrefetchedReplayBatches(Iterator[TrainingBatch]):
    def __init__(
        self,
        loader: MappedReplayBatchLoader,
        device: torch.device,
        uses_cuda: bool,
        depth: int,
    ) -> None:
        self.loader = loader
        self.device = device
        self.uses_cuda = uses_cuda
        self._closed = False
        self._prepared_batches = loader._prepared_batches()
        self._transfer_stream = shared_transfer_stream(device) if uses_cuda else None
        self._pinned_slots = PinnedBatchSlotPool(depth) if uses_cuda and loader.pin_memory else None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='replay-batch-prefetch')
        self._pending_batches: deque[Future[_PrefetchedBatch]] = deque(
            self._executor.submit(self._prepare_next) for _ in range(depth)
        )
        self._active_batch: _PrefetchedBatch | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self) -> PrefetchedReplayBatches:
        return self

    def __next__(self) -> TrainingBatch:
        if self._closed:
            raise StopIteration
        next_batch = self._pending_batches.popleft()
        try:
            prefetched = next_batch.result()
        except StopIteration:
            try:
                self.close()
            except BaseException as error:
                raise RuntimeError('Replay batch prefetch failed during cleanup.') from error
            raise StopIteration from None
        except BaseException as error:
            self._close_ignoring_errors()
            raise RuntimeError('Replay batch prefetch failed.') from error
        self._active_batch = prefetched
        try:
            self._pending_batches.append(self._executor.submit(self._prepare_next))
            self._make_transfer_visible(prefetched)
        except BaseException as error:
            self._close_ignoring_errors()
            raise RuntimeError('Replay batch prefetch failed.') from error
        self._active_batch = None
        return prefetched.device_batch

    def __enter__(self) -> PrefetchedReplayBatches:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception is None:
            self.close()
        else:
            self._close_ignoring_errors()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cleanup_errors: list[BaseException] = []
        active_batch = self._active_batch
        self._active_batch = None
        pending_batches = tuple(self._pending_batches)
        self._pending_batches.clear()
        for pending_batch in pending_batches:
            pending_batch.cancel()
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except BaseException as error:
            cleanup_errors.append(error)
        completed_batches: list[_PrefetchedBatch] = []
        if active_batch is not None:
            completed_batches.append(active_batch)
        for pending_batch in pending_batches:
            if pending_batch.cancelled():
                continue
            try:
                prefetched = pending_batch.result()
            except BaseException:
                pass
            else:
                completed_batches.append(prefetched)
        pinned_slots = self._pinned_slots
        if pinned_slots is not None:
            try:
                pinned_slots.close()
            except BaseException as error:
                cleanup_errors.append(error)
        for prefetched in completed_batches:
            if prefetched.pinned_slot is not None or prefetched.transfer_complete is None:
                continue
            try:
                prefetched.transfer_complete.synchronize()
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            self._prepared_batches.close()
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise RuntimeError('Replay batch prefetch cleanup failed.') from cleanup_errors[0]

    def _close_ignoring_errors(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def _prepare_next(self) -> _PrefetchedBatch:
        prepared_batch = next(self._prepared_batches)
        if not self.uses_cuda:
            host_batch = prepared_batch.pin_memory() if self.loader.pin_memory else prepared_batch
            return _PrefetchedBatch(host_batch, host_batch, None, None)
        transfer_stream = self._transfer_stream
        assert transfer_stream is not None
        pinned_slots = self._pinned_slots
        pinned_slot = pinned_slots.fill(prepared_batch) if pinned_slots is not None else None
        host_batch = pinned_slot.batch if pinned_slot is not None else prepared_batch
        assert host_batch is not None
        transfer_complete: torch.cuda.Event | None = None
        slot_released_or_tracked = False
        try:
            with torch.cuda.device(self.device):
                transfer_complete = torch.cuda.Event()
                with torch.cuda.stream(transfer_stream):
                    try:
                        device_batch = host_batch.to_device(self.device, non_blocking=True)
                        transfer_complete.record(transfer_stream)
                    except BaseException as transfer_error:
                        try:
                            transfer_complete.record(transfer_stream)
                        except BaseException:
                            try:
                                transfer_stream.synchronize()
                            except BaseException as synchronization_error:
                                if pinned_slot is not None:
                                    pinned_slots.mark_unrecoverable(pinned_slot)
                                    slot_released_or_tracked = True
                                raise transfer_error from synchronization_error
                        else:
                            if pinned_slot is not None:
                                pinned_slots.mark_transfer_in_flight(pinned_slot, transfer_complete)
                                slot_released_or_tracked = True
                            raise transfer_error
                        if pinned_slot is not None:
                            pinned_slots.release_untransferred(pinned_slot)
                            slot_released_or_tracked = True
                        raise transfer_error
        except BaseException:
            if pinned_slot is not None and not slot_released_or_tracked:
                pinned_slots.release_untransferred(pinned_slot)
            raise
        assert transfer_complete is not None
        if pinned_slot is not None:
            pinned_slots.mark_transfer_in_flight(pinned_slot, transfer_complete)
            slot_released_or_tracked = True
        return _PrefetchedBatch(host_batch, device_batch, transfer_complete, pinned_slot)

    def _make_transfer_visible(self, prefetched: _PrefetchedBatch) -> None:
        transfer_complete = prefetched.transfer_complete
        if transfer_complete is None:
            return
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(transfer_complete)
        record_training_batch_stream(prefetched.device_batch, current_stream)


def build_training_batch(
    store: ReplayStore,
    state: GameStateContract[PositionT],
    sample_indices: npt.NDArray[np.int64] | Sequence[int],
    augmentation_indices: npt.NDArray[np.int64],
) -> TrainingBatch:
    if len(sample_indices) == 0:
        raise ValueError('Training batches cannot be empty.')
    if len(sample_indices) != len(augmentation_indices):
        raise ValueError('Every replay sample requires one augmentation index.')
    logical_indices = np.asarray(sample_indices, dtype=np.int64)
    _validate_augmentation_indices(augmentation_indices)
    physical_indices = store.logical_to_physical(logical_indices)
    columns = store.gather_physical(physical_indices)
    return build_training_batch_from_columns(columns, store.layout, state, augmentation_indices)


def build_training_batch_from_columns(
    columns: ReplayColumnViews,
    layout: ReplayLayout,
    state: GameStateContract[PositionT],
    augmentation_indices: npt.NDArray[np.int64],
) -> TrainingBatch:
    _validate_augmentation_indices(augmentation_indices)
    if columns.row_count == 0 or len(augmentation_indices) != columns.row_count:
        raise ValueError('Replay columns and augmentation indices must form a nonempty aligned batch.')
    states = decode_augmented_states(columns.encoded_state, state, augmentation_indices)
    targets = build_dense_targets(columns, layout, state, augmentation_indices)
    return TrainingBatch(
        states=torch.from_numpy(states),
        policy_targets=torch.from_numpy(targets.policy),
        policy_legal_action_ids=torch.from_numpy(targets.policy_legal_action_ids),
        wdl_targets=torch.from_numpy(columns.wdl_target),
        root_values=torch.from_numpy(columns.root_value),
        auxiliary_targets=tuple(torch.from_numpy(target) for target in targets.auxiliary),
        auxiliary_legal_action_ids=tuple(torch.from_numpy(actions) for actions in targets.auxiliary_legal_action_ids),
        auxiliary_eligibility=tuple(torch.from_numpy(mask) for mask in targets.auxiliary_eligibility),
        sample_weights=torch.from_numpy(columns.sample_weight),
        source_model_generations=torch.from_numpy(columns.source_model_generation.astype(np.int64)),
        source_created_at_seconds=torch.from_numpy(columns.source_timestamp),
    )


def build_dense_targets(
    columns: ReplayColumnViews,
    layout: ReplayLayout,
    state: GameStateContract[PositionT],
    augmentation_indices: npt.NDArray[np.int64],
) -> DenseTargetArrays:
    _validate_augmentation_indices(augmentation_indices)
    if np.any((augmentation_indices < 0) | (augmentation_indices >= state.augmentation_count)):
        raise ValueError('Augmentation index is outside the game contract.')
    permutations = state.action_permutations
    policies, policy_legal_action_ids = _dense_policy(
        columns.policy,
        augmentation_indices,
        permutations,
        state.action_size,
    )
    auxiliary_targets: list[npt.NDArray[np.float32]] = []
    auxiliary_legal_action_ids: list[npt.NDArray[np.int64]] = []
    auxiliary_eligibility: list[npt.NDArray[np.bool_]] = []
    for target in columns.auxiliary:
        empty_legal = np.full((columns.row_count, layout.maximum_legal_actions), -1, dtype=np.int64)
        match target:
            case ReplayNextPolicyColumnViews(policy=policy, eligible=eligible):
                eligible_rows = eligible.astype(np.bool_, copy=False)
                dense, legal = _dense_policy(
                    policy,
                    augmentation_indices,
                    permutations,
                    state.action_size,
                    eligible_rows,
                )
                auxiliary_targets.append(dense)
                auxiliary_legal_action_ids.append(legal)
                auxiliary_eligibility.append(eligible_rows)
            case ReplayScalarColumnViews(value=value, eligible=eligible):
                eligible_rows = eligible.astype(np.bool_, copy=False)
                auxiliary_targets.append(np.where(eligible_rows, value, np.float32(0.0)).reshape(-1, 1))
                auxiliary_legal_action_ids.append(empty_legal)
                auxiliary_eligibility.append(eligible_rows)
            case ReplaySearchBudgetColumnViews(value=value, eligible=eligible):
                eligible_rows = eligible.astype(np.bool_, copy=False)
                auxiliary_targets.append(np.where(eligible_rows, value, np.float32(0.0)).reshape(-1, 1))
                auxiliary_legal_action_ids.append(empty_legal)
                auxiliary_eligibility.append(eligible_rows)
            case ReplayLegalMovesColumnViews():
                legal_moves = np.zeros((columns.row_count, state.action_size), dtype=np.float32)
                valid = policy_legal_action_ids >= 0
                rows = np.broadcast_to(np.arange(columns.row_count)[:, np.newaxis], valid.shape)
                legal_moves[rows[valid], policy_legal_action_ids[valid]] = 1.0
                auxiliary_targets.append(legal_moves)
                auxiliary_legal_action_ids.append(policy_legal_action_ids.copy())
                auxiliary_eligibility.append(np.ones(columns.row_count, dtype=np.bool_))
    return DenseTargetArrays(
        policy=policies,
        policy_legal_action_ids=policy_legal_action_ids,
        auxiliary=tuple(auxiliary_targets),
        auxiliary_legal_action_ids=tuple(auxiliary_legal_action_ids),
        auxiliary_eligibility=tuple(auxiliary_eligibility),
    )


def _validate_augmentation_indices(augmentation_indices: npt.NDArray[np.int64]) -> None:
    if (
        not isinstance(augmentation_indices, np.ndarray)
        or augmentation_indices.dtype != np.int64
        or augmentation_indices.ndim != 1
    ):
        raise ValueError('Augmentation indices must be a one-dimensional int64 NumPy array.')


def decode_augmented_states(
    encoded_states: npt.NDArray[np.uint8],
    state: GameStateContract[PositionT],
    augmentation_indices: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    _validate_augmentation_indices(augmentation_indices)
    states = decode_states(encoded_states, state)
    state.transform_decoded_states(states, augmentation_indices)
    return states


def decode_states(
    encoded_states: npt.NDArray[np.uint8],
    state: GameStateContract[PositionT],
) -> npt.NDArray[np.float32]:
    representation = state.representation
    states = np.empty(
        (len(encoded_states), representation.channels, representation.rows, representation.columns),
        dtype=np.float32,
    )
    decode_packed_plane_bytes_into(
        encoded_states,
        representation.packed_planes,
        representation.binary_channels,
        representation.scalar_channels,
        states,
    )
    return states


def _dense_policy(
    policy: ReplayPolicyColumnViews,
    augmentation_indices: npt.NDArray[np.int64],
    action_permutations: npt.NDArray[np.uint16],
    action_size: int,
    eligible_rows: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    row_count = len(policy.entry_count)
    dense = np.zeros((row_count, action_size), dtype=np.float32)
    entry_positions = np.arange(policy.action_ids.shape[1])[np.newaxis, :]
    valid_entries = entry_positions < policy.entry_count[:, np.newaxis]
    if eligible_rows is not None:
        valid_entries &= eligible_rows[:, np.newaxis]
    safe_actions = np.where(valid_entries, policy.action_ids, 0)
    transformed_actions = action_permutations[augmentation_indices[:, np.newaxis], safe_actions]
    visits = policy.visit_counts.astype(np.float32)
    visits[~valid_entries] = 0.0
    totals = visits.sum(axis=1, keepdims=True)
    probabilities = np.divide(visits, totals, out=np.zeros_like(visits), where=totals > 0.0)
    rows = np.broadcast_to(np.arange(row_count)[:, np.newaxis], valid_entries.shape)
    dense[rows[valid_entries], transformed_actions[valid_entries]] = probabilities[valid_entries]

    legal = np.full(policy.legal_action_ids.shape, -1, dtype=np.int64)
    legal_positions = np.arange(policy.legal_action_ids.shape[1])[np.newaxis, :]
    valid_legal = legal_positions < policy.legal_count[:, np.newaxis]
    if eligible_rows is not None:
        valid_legal &= eligible_rows[:, np.newaxis]
    safe_legal = np.where(valid_legal, policy.legal_action_ids, 0)
    transformed_legal = action_permutations[augmentation_indices[:, np.newaxis], safe_legal]
    legal[valid_legal] = transformed_legal[valid_legal]
    return dense, legal

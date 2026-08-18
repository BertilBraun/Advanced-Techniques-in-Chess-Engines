from __future__ import annotations

import time
from collections.abc import Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from AlphaZeroCpp import GameSearchVisit
from src.games.contracts import GameStateContract
from src.games.representation import decode_packed_planes
from src.replay.contracts import (
    EligibleLegalMovesTarget,
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    EligibleScalarAuxiliaryTarget,
)
from src.replay.manager import ReplayDescription
from src.replay.store import ReplayStore
from src.training.batch import TrainingBatch
from src.training.targets import auxiliary_head_output_size

PositionT = TypeVar('PositionT')


@dataclass(frozen=True)
class _PrefetchedBatch:
    # Retain pinned source tensors until the asynchronous transfer completes.
    host_batch: TrainingBatch
    device_batch: TrainingBatch
    transfer_complete: torch.cuda.Event | None


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
        self.rows_read = 0
        self.read_seconds = 0.0

    @property
    def rows_per_second(self) -> float:
        return self.rows_read / self.read_seconds if self.read_seconds > 0.0 else 0.0

    def __iter__(self) -> Iterator[TrainingBatch]:
        return self._prepared_batches()

    def prefetch(self, device: torch.device, uses_cuda: bool) -> PrefetchedReplayBatches:
        if uses_cuda != (device.type == 'cuda'):
            raise ValueError('CUDA prefetch must agree with the training device type.')
        return PrefetchedReplayBatches(self, device, uses_cuda)

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
                started_at = time.perf_counter()
                batch = build_training_batch(
                    store,
                    self.state,
                    tuple(int(index) for index in sample_indices),
                    tuple(int(index) for index in augmentation_indices),
                )
                prepared = batch.pin_memory() if self.pin_memory else batch
                self.rows_read += len(sample_indices)
                self.read_seconds += time.perf_counter() - started_at
                yield prepared
        finally:
            store.close()


class PrefetchedReplayBatches(Iterator[TrainingBatch]):
    def __init__(
        self,
        loader: MappedReplayBatchLoader,
        device: torch.device,
        uses_cuda: bool,
    ) -> None:
        self.loader = loader
        self.device = device
        self.uses_cuda = uses_cuda
        self._closed = False
        self._prepared_batches = loader._prepared_batches()
        self._transfer_stream = torch.cuda.Stream(device=device) if uses_cuda else None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='replay-batch-prefetch')
        self._next_batch: Future[_PrefetchedBatch] = self._executor.submit(self._prepare_next)

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self) -> PrefetchedReplayBatches:
        return self

    def __next__(self) -> TrainingBatch:
        if self._closed:
            raise StopIteration
        try:
            prefetched = self._next_batch.result()
        except StopIteration:
            self.close()
            raise StopIteration from None
        except BaseException as error:
            self.close()
            raise RuntimeError('Replay batch prefetch failed.') from error
        self._next_batch = self._executor.submit(self._prepare_next)
        self._synchronize_transfer(prefetched)
        return prefetched.device_batch

    def __enter__(self) -> PrefetchedReplayBatches:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._next_batch.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        if not self._next_batch.cancelled():
            try:
                prefetched = self._next_batch.result()
            except BaseException:
                pass
            else:
                self._synchronize_transfer(prefetched)
        self._prepared_batches.close()
        self._closed = True

    def _prepare_next(self) -> _PrefetchedBatch:
        host_batch = next(self._prepared_batches)
        if not self.uses_cuda:
            return _PrefetchedBatch(host_batch, host_batch, None)
        transfer_stream = self._transfer_stream
        assert transfer_stream is not None
        with torch.cuda.device(self.device), torch.cuda.stream(transfer_stream):
            device_batch = host_batch.to_device(self.device, non_blocking=True)
            transfer_complete = torch.cuda.Event()
            transfer_complete.record(transfer_stream)
        return _PrefetchedBatch(host_batch, device_batch, transfer_complete)

    @staticmethod
    def _synchronize_transfer(prefetched: _PrefetchedBatch) -> None:
        if prefetched.transfer_complete is not None:
            prefetched.transfer_complete.synchronize()


def build_training_batch(
    store: ReplayStore,
    state: GameStateContract[PositionT],
    sample_indices: Sequence[int],
    augmentation_indices: Sequence[int],
) -> TrainingBatch:
    if not sample_indices:
        raise ValueError('Training batches cannot be empty.')
    if len(sample_indices) != len(augmentation_indices):
        raise ValueError('Every replay sample requires one augmentation index.')
    samples = tuple(
        state.transform_replay_targets(store.sample_at(sample_index), augmentation_index)
        for sample_index, augmentation_index in zip(sample_indices, augmentation_indices)
    )
    representation = state.representation
    states = np.empty(
        (len(samples), representation.channels, representation.rows, representation.columns),
        dtype=np.float32,
    )
    policies = np.zeros((len(samples), state.action_size), dtype=np.float32)
    policy_legal_action_ids = np.full(
        (len(samples), store.layout.maximum_legal_actions),
        -1,
        dtype=np.int64,
    )
    auxiliary_targets = tuple(
        np.zeros((len(samples), auxiliary_head_output_size(head)), dtype=np.float32)
        for head in store.layout.targets.auxiliary_heads
    )
    auxiliary_eligibility = tuple(np.zeros(len(samples), dtype=np.bool_) for _ in store.layout.targets.auxiliary_heads)
    auxiliary_legal_action_ids = tuple(
        np.full((len(samples), store.layout.maximum_legal_actions), -1, dtype=np.int64)
        for _ in store.layout.targets.auxiliary_heads
    )
    for row, sample in enumerate(samples):
        states[row] = decode_packed_planes(
            sample.encoded_state,
            representation.packed_planes,
            representation.binary_channels,
            representation.scalar_channels,
        )
        _write_dense_policy(policies[row], sample.policy.visits)
        policy_legal_action_ids[row, : len(sample.policy.legal_action_ids)] = sample.policy.legal_action_ids
        for target_index, target in enumerate(sample.auxiliary_targets):
            match target:
                case EligibleNextPolicyTarget(policy=policy):
                    _write_dense_policy(auxiliary_targets[target_index][row], policy.visits)
                    auxiliary_legal_action_ids[target_index][row, : len(policy.legal_action_ids)] = (
                        policy.legal_action_ids
                    )
                    auxiliary_eligibility[target_index][row] = True
                case EligibleRemainingGameLengthTarget(normalized_length=normalized_length):
                    auxiliary_targets[target_index][row, 0] = normalized_length
                    auxiliary_eligibility[target_index][row] = True
                case EligibleScalarAuxiliaryTarget(value=value):
                    auxiliary_targets[target_index][row, 0] = value
                    auxiliary_eligibility[target_index][row] = True
                case EligibleLegalMovesTarget():
                    legal_actions = sample.policy.legal_action_ids
                    auxiliary_targets[target_index][row, legal_actions] = 1.0
                    auxiliary_legal_action_ids[target_index][row, : len(legal_actions)] = legal_actions
                    auxiliary_eligibility[target_index][row] = True
    return TrainingBatch(
        states=torch.from_numpy(states),
        policy_targets=torch.from_numpy(policies),
        policy_legal_action_ids=torch.from_numpy(policy_legal_action_ids),
        wdl_targets=torch.tensor(
            [(sample.wdl_target.win, sample.wdl_target.draw, sample.wdl_target.loss) for sample in samples],
            dtype=torch.float32,
        ),
        root_values=torch.tensor([sample.root_value for sample in samples], dtype=torch.float32),
        auxiliary_targets=tuple(torch.from_numpy(target) for target in auxiliary_targets),
        auxiliary_legal_action_ids=tuple(torch.from_numpy(actions) for actions in auxiliary_legal_action_ids),
        auxiliary_eligibility=tuple(torch.from_numpy(mask) for mask in auxiliary_eligibility),
        sample_weights=torch.tensor([sample.sample_weight for sample in samples], dtype=torch.float32),
        source_model_generations=torch.tensor(
            [sample.source_model_generation for sample in samples],
            dtype=torch.int64,
        ),
        source_created_at_seconds=torch.tensor(
            [sample.source_created_at_seconds for sample in samples],
            dtype=torch.float64,
        ),
    )


def _write_dense_policy(
    destination: npt.NDArray[np.float32],
    visits: Sequence[GameSearchVisit],
) -> None:
    total_visits = sum(visit.visit_count for visit in visits)
    for visit in visits:
        destination[visit.action_id] = visit.visit_count / total_visits

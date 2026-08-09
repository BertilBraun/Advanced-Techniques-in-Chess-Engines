from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar
import time

import numpy as np
import numpy.typing as npt
import torch

from src.games.contracts import GameStateContract
from src.games.representation import decode_packed_planes
from src.replay.contracts import EligibleNextPolicyTarget
from src.replay.manager import ReplayDescription
from src.replay.store import ReplayStore
from src.self_play.completed_game import SparseSearchVisit
from src.training.batch import TrainingBatch


PositionT = TypeVar('PositionT')


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
    auxiliary_targets = tuple(
        np.zeros((len(samples), head.action_size), dtype=np.float32) for head in store.layout.targets.auxiliary_heads
    )
    auxiliary_eligibility = tuple(np.zeros(len(samples), dtype=np.bool_) for _ in store.layout.targets.auxiliary_heads)
    for row, sample in enumerate(samples):
        states[row] = decode_packed_planes(
            sample.encoded_state,
            representation.packed_planes,
            representation.binary_channels,
            representation.scalar_channels,
        )
        _write_dense_policy(policies[row], sample.policy.visits)
        for target_index, target in enumerate(sample.auxiliary_targets):
            match target:
                case EligibleNextPolicyTarget(policy=policy):
                    _write_dense_policy(auxiliary_targets[target_index][row], policy.visits)
                    auxiliary_eligibility[target_index][row] = True
    return TrainingBatch(
        states=torch.from_numpy(states),
        policy_targets=torch.from_numpy(policies),
        wdl_targets=torch.tensor(
            [(sample.wdl_target.win, sample.wdl_target.draw, sample.wdl_target.loss) for sample in samples],
            dtype=torch.float32,
        ),
        root_values=torch.tensor([sample.root_value for sample in samples], dtype=torch.float32),
        auxiliary_targets=tuple(torch.from_numpy(target) for target in auxiliary_targets),
        auxiliary_eligibility=tuple(torch.from_numpy(mask) for mask in auxiliary_eligibility),
        sample_weights=torch.tensor([sample.sample_weight for sample in samples], dtype=torch.float32),
    )


def _write_dense_policy(
    destination: npt.NDArray[np.float32],
    visits: Sequence[SparseSearchVisit],
) -> None:
    total_visits = sum(visit.visit_count for visit in visits)
    for visit in visits:
        destination[visit.action_id] = visit.visit_count / total_visits

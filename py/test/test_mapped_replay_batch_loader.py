from collections.abc import Sequence
from pathlib import Path
from threading import Event

import torch
import pytest
from AlphaZeroCpp import GameSearchVisit

from src.games.chess.contract import CHESS_STATE_CONTRACT, ChessStateContract
from src.games.contracts import WdlTarget
import src.replay.batch_loader as batch_loader_module
from src.replay.batch_loader import MappedReplayBatchLoader, build_training_batch
from src.replay.contracts import (
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.layout import ReplayLayout
from src.replay.manager import ReplayDescription
from src.replay.store import ReplayStore
from src.training.batch import TrainingBatch
from src.training.targets import NextPolicyHeadLayout, RemainingGameLengthHeadLayout, TrainingTargetLayout


class IdentityAugmentationChessStateContract(ChessStateContract):
    @property
    def augmentation_count(self) -> int:
        return 1

    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        if augmentation_index != 0:
            raise ValueError('Test contract supports only identity augmentation.')
        return action_id


IDENTITY_CHESS_STATE_CONTRACT = IdentityAugmentationChessStateContract()


def _layout() -> ReplayLayout:
    action_size = CHESS_STATE_CONTRACT.action_size
    return ReplayLayout(
        packed_planes=CHESS_STATE_CONTRACT.packed_plane_layout,
        targets=TrainingTargetLayout(
            action_size=action_size,
            wdl_size=3,
            auxiliary_heads=(
                NextPolicyHeadLayout(kind='next_policy', action_size=action_size, ply_offset=1),
                RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=196.0),
            ),
        ),
        maximum_policy_entries=2,
    )


def _sample(weight: float) -> ReplaySample:
    primary = SparsePolicyTarget(
        visits=(
            GameSearchVisit(action_id=0, visit_count=3),
            GameSearchVisit(action_id=1, visit_count=1),
        )
    )
    return ReplaySample(
        encoded_state=CHESS_STATE_CONTRACT.packed_plane_layout.value(
            bytes(CHESS_STATE_CONTRACT.packed_plane_layout.payload_bytes)
        ),
        policy=primary,
        wdl_target=WdlTarget(win=0.25, draw=0.5, loss=0.25),
        root_value=0.125,
        auxiliary_targets=(
            EligibleNextPolicyTarget(policy=primary),
            EligibleRemainingGameLengthTarget(normalized_length=0.25),
        ),
        sample_weight=weight,
        source_model_generation=1,
        source_created_at_seconds=10.0,
    )


def _description(path: Path, store: ReplayStore) -> ReplayDescription:
    state = store.state
    return ReplayDescription(
        path=path,
        head=state.head,
        size=state.size,
        logical_capacity=state.logical_capacity,
        maximum_capacity=state.maximum_capacity,
        layout=store.layout,
    )


def test_mapped_loader_builds_canonical_batches_and_disjoint_rank_slices(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, _layout(), maximum_capacity=4, logical_capacity=4)
    for weight in (1.0, 2.0, 3.0, 4.0):
        store.append(_sample(weight))
    store.flush()
    description = _description(path, store)
    store.close()

    common = {
        'replay': description,
        'state': IDENTITY_CHESS_STATE_CONTRACT,
        'source_optimizer_step': 20,
        'optimizer_steps': 1,
        'global_batch_size': 4,
        'world_size': 2,
        'sampler_seed': 91,
        'pin_memory': False,
    }
    rank_zero = next(iter(MappedReplayBatchLoader(rank=0, **common)))
    rank_one = next(iter(MappedReplayBatchLoader(rank=1, **common)))

    assert rank_zero.states.shape == (2, CHESS_STATE_CONTRACT.representation.channels, 8, 8)
    assert rank_zero.policy_targets.shape == (2, CHESS_STATE_CONTRACT.action_size)
    assert torch.allclose(rank_zero.policy_targets.sum(dim=1), torch.ones(2))
    assert torch.allclose(rank_zero.wdl_targets, torch.tensor(((0.25, 0.5, 0.25),) * 2))
    assert torch.all(rank_zero.auxiliary_eligibility[0])
    assert rank_zero.auxiliary_targets[0].shape == (2, CHESS_STATE_CONTRACT.action_size)
    assert rank_zero.auxiliary_targets[1].shape == (2, 1)
    assert torch.allclose(rank_zero.auxiliary_targets[1], torch.full((2, 1), 0.25))
    assert torch.all(rank_zero.auxiliary_eligibility[1])
    assert set(rank_zero.sample_weights.tolist()).isdisjoint(rank_one.sample_weights.tolist())


def test_prefetch_preserves_batch_order_and_exact_row_accounting(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, _layout(), maximum_capacity=4, logical_capacity=4)
    for weight in (1.0, 2.0, 3.0, 4.0):
        store.append(_sample(weight))
    store.flush()
    description = _description(path, store)
    store.close()
    common = {
        'replay': description,
        'state': IDENTITY_CHESS_STATE_CONTRACT,
        'source_optimizer_step': 20,
        'optimizer_steps': 3,
        'global_batch_size': 4,
        'world_size': 1,
        'rank': 0,
        'sampler_seed': 91,
        'pin_memory': False,
    }
    synchronous_loader = MappedReplayBatchLoader(**common)
    expected_weights = tuple(tuple(batch.sample_weights.tolist()) for batch in synchronous_loader)
    prefetched_loader = MappedReplayBatchLoader(**common)

    with prefetched_loader.prefetch(torch.device('cpu'), uses_cuda=False) as batches:
        actual_weights = tuple(tuple(batch.sample_weights.tolist()) for batch in batches)

    assert actual_weights == expected_weights
    assert synchronous_loader.rows_read == 12
    assert prefetched_loader.rows_read == 12
    assert batches.closed


def test_prefetch_prepares_next_batch_before_consumer_requests_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, _layout(), maximum_capacity=4, logical_capacity=4)
    for weight in (1.0, 2.0, 3.0, 4.0):
        store.append(_sample(weight))
    store.flush()
    description = _description(path, store)
    store.close()
    second_batch_started = Event()
    allow_second_batch = Event()
    build_count = 0

    def controlled_build(
        replay_store: ReplayStore,
        state: ChessStateContract,
        sample_indices: Sequence[int],
        augmentation_indices: Sequence[int],
    ) -> TrainingBatch:
        nonlocal build_count
        build_count += 1
        if build_count == 2:
            second_batch_started.set()
            if not allow_second_batch.wait(timeout=5.0):
                raise TimeoutError('Test did not release second batch construction.')
        return build_training_batch(replay_store, state, sample_indices, augmentation_indices)

    monkeypatch.setattr(batch_loader_module, 'build_training_batch', controlled_build)
    loader = MappedReplayBatchLoader(
        replay=description,
        state=IDENTITY_CHESS_STATE_CONTRACT,
        source_optimizer_step=20,
        optimizer_steps=2,
        global_batch_size=2,
        world_size=1,
        rank=0,
        sampler_seed=91,
        pin_memory=False,
    )
    batches = loader.prefetch(torch.device('cpu'), uses_cuda=False)

    try:
        first_batch = next(batches)
        assert len(first_batch) == 2
        assert second_batch_started.wait(timeout=1.0)
    finally:
        allow_second_batch.set()
        batches.close()

    assert batches.closed


def test_prefetch_propagates_producer_failure_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, _layout(), maximum_capacity=2, logical_capacity=2)
    store.append(_sample(1.0))
    store.append(_sample(2.0))
    store.flush()
    description = _description(path, store)
    store.close()

    def fail_build(
        replay_store: ReplayStore,
        state: ChessStateContract,
        sample_indices: Sequence[int],
        augmentation_indices: Sequence[int],
    ) -> TrainingBatch:
        raise ValueError('broken replay row')

    monkeypatch.setattr(batch_loader_module, 'build_training_batch', fail_build)
    loader = MappedReplayBatchLoader(
        replay=description,
        state=IDENTITY_CHESS_STATE_CONTRACT,
        source_optimizer_step=20,
        optimizer_steps=1,
        global_batch_size=2,
        world_size=1,
        rank=0,
        sampler_seed=91,
        pin_memory=False,
    )
    batches = loader.prefetch(torch.device('cpu'), uses_cuda=False)

    with pytest.raises(RuntimeError, match='Replay batch prefetch failed') as raised:
        next(batches)

    assert isinstance(raised.value.__cause__, ValueError)
    assert batches.closed

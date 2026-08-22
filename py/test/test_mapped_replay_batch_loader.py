from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Event, Thread

import numpy as np
import pytest
import torch

pytest.importorskip('AlphaZeroCpp')
import src.replay.batch_loader as batch_loader_module
from AlphaZeroCpp import GameSearchVisit
from src.games.chess.contract import CHESS_STATE_CONTRACT, ChessStateContract
from src.games.contracts import WdlTarget
from src.games.representation import decode_packed_planes
from src.replay.batch_loader import MappedReplayBatchLoader, build_training_batch
from src.replay.contracts import (
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.description import ReplayDescription
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.self_play.completed_game import SearchVisitCounts
from src.training.batch import TrainingBatch, TrainingModelOutput
from src.training.objective import ResolvedNextPolicyLoss, ResolvedRemainingGameLengthLoss, ResolvedTrainingObjective
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
        maximum_legal_actions=CHESS_STATE_CONTRACT.maximum_legal_action_count,
    )


def _sample(weight: float) -> ReplaySample:
    primary = SparsePolicyTarget(
        visits=SearchVisitCounts.from_native(
            (
                GameSearchVisit(action_id=0, visit_count=3),
                GameSearchVisit(action_id=1, visit_count=1),
            )
        ),
        legal_action_ids=(0, 1, 2),
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
    assert torch.equal(rank_zero.policy_legal_action_ids[:, :3], torch.tensor(((0, 1, 2),) * 2))
    assert torch.allclose(rank_zero.wdl_targets, torch.tensor(((0.25, 0.5, 0.25),) * 2))
    assert torch.all(rank_zero.auxiliary_eligibility[0])
    assert rank_zero.auxiliary_targets[0].shape == (2, CHESS_STATE_CONTRACT.action_size)
    assert rank_zero.auxiliary_targets[1].shape == (2, 1)
    assert torch.allclose(rank_zero.auxiliary_targets[1], torch.full((2, 1), 0.25))
    assert torch.all(rank_zero.auxiliary_eligibility[1])
    assert set(rank_zero.sample_weights.tolist()).isdisjoint(rank_one.sample_weights.tolist())


def test_build_training_batch_accepts_numpy_index_arrays(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, _layout(), maximum_capacity=2, logical_capacity=2)
    store.append(_sample(1.0))
    store.append(_sample(2.0))

    batch = build_training_batch(
        store,
        IDENTITY_CHESS_STATE_CONTRACT,
        np.asarray((1, 0), dtype=np.int64),
        np.asarray((0, 0), dtype=np.int64),
    )

    assert batch.sample_weights.tolist() == [2.0, 1.0]
    store.close()


def _reference_object_batch(
    store: ReplayStore,
    sample_indices: np.ndarray,
    augmentation_indices: np.ndarray,
) -> TrainingBatch:
    samples = tuple(store.sample_at(int(index)) for index in sample_indices)
    row_count = len(samples)
    states = np.empty((row_count, 29, 8, 8), dtype=np.float32)
    policies = np.zeros((row_count, CHESS_STATE_CONTRACT.action_size), dtype=np.float32)
    legal = np.full((row_count, store.layout.maximum_legal_actions), -1, dtype=np.int64)
    next_policies = np.zeros_like(policies)
    next_legal = np.full_like(legal, -1)
    remaining = np.zeros((row_count, 1), dtype=np.float32)
    for row, (sample, augmentation) in enumerate(zip(samples, augmentation_indices, strict=True)):
        decoded = decode_packed_planes(
            sample.encoded_state,
            CHESS_STATE_CONTRACT.packed_plane_layout,
            CHESS_STATE_CONTRACT.representation.binary_channels,
            CHESS_STATE_CONTRACT.representation.scalar_channels,
        ).astype(np.float32)[np.newaxis, ...]
        CHESS_STATE_CONTRACT.transform_decoded_states(decoded, np.asarray((augmentation,), dtype=np.int64))
        states[row] = decoded[0]
        permutation = CHESS_STATE_CONTRACT.action_permutations[int(augmentation)]
        visits = np.asarray(sample.policy.visits.visit_counts, dtype=np.float32)
        actions = permutation[np.asarray(sample.policy.visits.action_ids, dtype=np.uint16)]
        policies[row, actions] = visits / visits.sum()
        transformed_legal = permutation[np.asarray(sample.policy.legal_action_ids, dtype=np.uint16)]
        legal[row, : len(transformed_legal)] = transformed_legal
        next_target = sample.auxiliary_targets[0]
        assert isinstance(next_target, EligibleNextPolicyTarget)
        next_visits = np.asarray(next_target.policy.visits.visit_counts, dtype=np.float32)
        next_actions = permutation[np.asarray(next_target.policy.visits.action_ids, dtype=np.uint16)]
        next_policies[row, next_actions] = next_visits / next_visits.sum()
        next_transformed_legal = permutation[np.asarray(next_target.policy.legal_action_ids, dtype=np.uint16)]
        next_legal[row, : len(next_transformed_legal)] = next_transformed_legal
        remaining_target = sample.auxiliary_targets[1]
        assert isinstance(remaining_target, EligibleRemainingGameLengthTarget)
        remaining[row, 0] = remaining_target.normalized_length
    return TrainingBatch(
        states=torch.from_numpy(states),
        policy_targets=torch.from_numpy(policies),
        policy_legal_action_ids=torch.from_numpy(legal),
        wdl_targets=torch.tensor(
            tuple((sample.wdl_target.win, sample.wdl_target.draw, sample.wdl_target.loss) for sample in samples),
            dtype=torch.float32,
        ),
        root_values=torch.tensor(tuple(sample.root_value for sample in samples), dtype=torch.float32),
        auxiliary_targets=(torch.from_numpy(next_policies), torch.from_numpy(remaining)),
        auxiliary_legal_action_ids=(
            torch.from_numpy(next_legal),
            torch.full(legal.shape, -1, dtype=torch.int64),
        ),
        auxiliary_eligibility=(torch.ones(row_count, dtype=torch.bool), torch.ones(row_count, dtype=torch.bool)),
        sample_weights=torch.tensor(tuple(sample.sample_weight for sample in samples), dtype=torch.float32),
        source_model_generations=torch.tensor(
            tuple(sample.source_model_generation for sample in samples), dtype=torch.int64
        ),
        source_created_at_seconds=torch.tensor(
            tuple(sample.source_created_at_seconds for sample in samples), dtype=torch.float64
        ),
    )


def test_vectorized_batch_exactly_matches_object_reference_across_wrap_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, _layout(), maximum_capacity=4, logical_capacity=4)
    for weight in range(1, 7):
        store.append(_sample(float(weight)))
    sample_indices = np.asarray((2, 0, 2, 1), dtype=np.int64)
    augmentations = np.asarray((1, 0, 0, 1), dtype=np.int64)

    actual = build_training_batch(store, CHESS_STATE_CONTRACT, sample_indices, augmentations)
    expected = _reference_object_batch(store, sample_indices, augmentations)

    for actual_tensor, expected_tensor in zip(
        (
            actual.states,
            actual.policy_targets,
            actual.policy_legal_action_ids,
            actual.wdl_targets,
            actual.root_values,
            *actual.auxiliary_targets,
            *actual.auxiliary_legal_action_ids,
            *actual.auxiliary_eligibility,
            actual.sample_weights,
            actual.source_model_generations,
            actual.source_created_at_seconds,
        ),
        (
            expected.states,
            expected.policy_targets,
            expected.policy_legal_action_ids,
            expected.wdl_targets,
            expected.root_values,
            *expected.auxiliary_targets,
            *expected.auxiliary_legal_action_ids,
            *expected.auxiliary_eligibility,
            expected.sample_weights,
            expected.source_model_generations,
            expected.source_created_at_seconds,
        ),
        strict=True,
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)

    objective = ResolvedTrainingObjective(
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        root_value_blend=0.25,
        auxiliary_losses=(
            ResolvedNextPolicyLoss(weight=0.2),
            ResolvedRemainingGameLengthLoss(weight=0.1),
        ),
    )
    generator = torch.Generator().manual_seed(17)
    initial_logits = (
        torch.randn((4, CHESS_STATE_CONTRACT.action_size), generator=generator),
        torch.randn((4, 3), generator=generator),
        torch.randn((4, CHESS_STATE_CONTRACT.action_size), generator=generator),
        torch.randn((4, 1), generator=generator),
    )

    def loss_and_gradients(batch: TrainingBatch) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        parameters = tuple(torch.nn.Parameter(values.clone()) for values in initial_logits)
        output = TrainingModelOutput(parameters[0], parameters[1], (parameters[2], parameters[3]))
        loss = objective.calculate_loss(output, batch).total
        loss.backward()
        return loss.detach(), tuple(
            parameter.grad.detach().clone() for parameter in parameters if parameter.grad is not None
        )

    actual_loss, actual_gradients = loss_and_gradients(actual)
    expected_loss, expected_gradients = loss_and_gradients(expected)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0.0, atol=0.0)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0.0, atol=0.0)
    store.close()


@pytest.mark.parametrize('depth', (1, 2, 4, 8))
def test_prefetch_preserves_batch_order(tmp_path: Path, depth: int) -> None:
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

    with prefetched_loader.prefetch(torch.device('cpu'), uses_cuda=False, depth=depth) as batches:
        actual_weights = tuple(tuple(batch.sample_weights.tolist()) for batch in batches)

    assert actual_weights == expected_weights
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
    batches = loader.prefetch(torch.device('cpu'), uses_cuda=False, depth=1)

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
    batches = loader.prefetch(torch.device('cpu'), uses_cuda=False, depth=4)

    with pytest.raises(RuntimeError, match='Replay batch prefetch failed') as raised:
        next(batches)

    assert isinstance(raised.value.__cause__, ValueError)
    assert batches.closed


@pytest.mark.parametrize('depth', (1, 2, 4, 8))
def test_prefetch_production_is_bounded_by_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    depth: int,
) -> None:
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, _layout(), maximum_capacity=2, logical_capacity=2)
    store.append(_sample(1.0))
    store.append(_sample(2.0))
    store.flush()
    description = _description(path, store)
    store.close()
    reached_depth = Event()
    exceeded_depth = Event()
    build_count = 0

    def counted_build(
        replay_store: ReplayStore,
        state: ChessStateContract,
        sample_indices: Sequence[int],
        augmentation_indices: Sequence[int],
    ) -> TrainingBatch:
        nonlocal build_count
        build_count += 1
        if build_count == depth:
            reached_depth.set()
        elif build_count > depth:
            exceeded_depth.set()
        return build_training_batch(replay_store, state, sample_indices, augmentation_indices)

    monkeypatch.setattr(batch_loader_module, 'build_training_batch', counted_build)
    loader = MappedReplayBatchLoader(
        replay=description,
        state=IDENTITY_CHESS_STATE_CONTRACT,
        source_optimizer_step=20,
        optimizer_steps=depth + 2,
        global_batch_size=2,
        world_size=1,
        rank=0,
        sampler_seed=91,
        pin_memory=False,
    )
    batches = loader.prefetch(torch.device('cpu'), uses_cuda=False, depth=depth)

    try:
        assert reached_depth.wait(timeout=5.0)
        assert not exceeded_depth.wait(timeout=0.1)
        assert build_count == depth
    finally:
        batches.close()


def test_prefetch_early_close_cancels_queued_preparation(
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
    first_build_started = Event()
    allow_first_build = Event()
    close_finished = Event()
    build_count = 0

    def blocked_build(
        replay_store: ReplayStore,
        state: ChessStateContract,
        sample_indices: Sequence[int],
        augmentation_indices: Sequence[int],
    ) -> TrainingBatch:
        nonlocal build_count
        build_count += 1
        first_build_started.set()
        if not allow_first_build.wait(timeout=5.0):
            raise TimeoutError('Test did not release replay batch construction.')
        return build_training_batch(replay_store, state, sample_indices, augmentation_indices)

    monkeypatch.setattr(batch_loader_module, 'build_training_batch', blocked_build)
    loader = MappedReplayBatchLoader(
        replay=description,
        state=IDENTITY_CHESS_STATE_CONTRACT,
        source_optimizer_step=20,
        optimizer_steps=10,
        global_batch_size=2,
        world_size=1,
        rank=0,
        sampler_seed=91,
        pin_memory=False,
    )
    batches = loader.prefetch(torch.device('cpu'), uses_cuda=False, depth=8)

    def close_batches() -> None:
        batches.close()
        close_finished.set()

    assert first_build_started.wait(timeout=5.0)
    close_thread = Thread(target=close_batches)
    close_thread.start()
    try:
        assert not close_finished.wait(timeout=0.1)
    finally:
        allow_first_build.set()
        close_thread.join(timeout=5.0)

    assert close_finished.is_set()
    assert build_count == 1
    assert batches.closed


def test_prefetch_rejects_nonpositive_depth(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, _layout(), maximum_capacity=2, logical_capacity=2)
    store.append(_sample(1.0))
    store.append(_sample(2.0))
    store.flush()
    description = _description(path, store)
    store.close()
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

    with pytest.raises(ValueError, match='depth must be positive'):
        loader.prefetch(torch.device('cpu'), uses_cuda=False, depth=0)

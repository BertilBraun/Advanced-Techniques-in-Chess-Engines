from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from src.games.contracts import WdlTarget
from src.games.representation import NetworkDimensions, PackedPlaneLayout, RepresentationDimensions
from src.replay.batch_loader import MappedReplayBatchLoader, SearchBudgetLabelledBatches
from src.replay.contracts import (
    EligibleRemainingGameLengthTarget,
    EligibleSearchBudgetTarget,
    IneligibleSearchBudgetTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.description import ReplayDescription
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.self_play.completed_game import SearchVisitCounts
from src.training.batch import TrainingBatch, TrainingModelOutput
from src.training.network import DensePolicyHeadConfiguration, Network, NetworkParams
from src.training.objective import (
    ResolvedRemainingGameLengthLoss,
    ResolvedSearchBudgetLoss,
    ResolvedTrainingObjective,
    resolve_auxiliary_losses,
)
from src.training.targets import (
    RemainingGameLengthHeadLayout,
    SearchBudgetHeadLayout,
    SearchBudgetTargetConfiguration,
    TrainingTargetLayout,
    search_budget_auxiliary_index,
)

PACKED_PLANES = PackedPlaneLayout(board_size=3, binary_plane_count=1, scalar_count=1)
AUXILIARY_INDEX = 1


class StubStateContract:
    """The head-batch path only needs decoding and augmentation, so the native game contract stays out of these tests."""

    @property
    def action_size(self) -> int:
        return 2

    @property
    def augmentation_count(self) -> int:
        return 1

    @property
    def representation(self) -> RepresentationDimensions:
        return RepresentationDimensions(
            channels=2,
            rows=3,
            columns=3,
            binary_channels=(0,),
            scalar_channels=(1,),
            packed_planes=PACKED_PLANES,
        )

    @property
    def action_permutations(self) -> npt.NDArray[np.uint16]:
        return np.asarray(((0, 1),), dtype=np.uint16)

    def transform_decoded_states(
        self,
        states: npt.NDArray[np.float32],
        augmentation_indices: npt.NDArray[np.int64],
    ) -> None:
        if np.any(augmentation_indices != 0):
            raise ValueError('The stub contract supports only identity augmentation.')


STUB_STATE = StubStateContract()


def _layout() -> ReplayLayout:
    return ReplayLayout(
        packed_planes=PACKED_PLANES,
        targets=TrainingTargetLayout(
            action_size=2,
            wdl_size=3,
            auxiliary_heads=(
                RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=100.0),
                SearchBudgetHeadLayout(kind='search_budget'),
            ),
        ),
        maximum_policy_entries=2,
        maximum_legal_actions=2,
    )


def _sample(normalized_target: float | None, marker: int) -> ReplaySample:
    budget = (
        IneligibleSearchBudgetTarget()
        if normalized_target is None
        else EligibleSearchBudgetTarget(
            curve=(normalized_target,) * 10,
            raw_kl=0.125,
            source_generation=1,
            model_generation=1,
            inference_model_sha256='a' * 64,
        )
    )
    payload = bytearray(PACKED_PLANES.payload_bytes)
    payload[0] = marker % 256
    return ReplaySample(
        encoded_state=PACKED_PLANES.value(bytes(payload)),
        policy=SparsePolicyTarget(
            visits=SearchVisitCounts(action_ids=(0, 1), visit_counts=(3, 1)),
            legal_action_ids=(0, 1),
        ),
        wdl_target=WdlTarget(win=0.0, draw=1.0, loss=0.0),
        root_value=0.0,
        auxiliary_targets=(EligibleRemainingGameLengthTarget(normalized_length=0.25), budget),
        sample_weight=1.0,
        source_model_generation=1,
        source_created_at_seconds=10.0,
    )


def _store_with_labels(
    path: Path,
    targets: tuple[float | None, ...],
    maximum_capacity: int | None = None,
) -> ReplayStore:
    capacity = len(targets) if maximum_capacity is None else maximum_capacity
    store = ReplayStore.create(path, _layout(), maximum_capacity=capacity, logical_capacity=capacity)
    store.extend(tuple(_sample(target, marker) for marker, target in enumerate(targets)))
    return store


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


def test_labelled_rows_are_indexed_without_scanning_unlabelled_columns(tmp_path: Path) -> None:
    store = _store_with_labels(tmp_path / 'replay.bin', (None, 0.25, None, 0.75, 0.5))

    indices = store.eligible_logical_indices(AUXILIARY_INDEX)
    store.close()

    np.testing.assert_array_equal(indices, np.asarray((1, 3, 4), dtype=np.int64))


def test_labelled_row_index_follows_the_wrapped_replay_ring(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = _store_with_labels(path, (None, None, 0.25), maximum_capacity=3)
    store.extend((_sample(0.5, 9), _sample(None, 10)))

    indices = store.eligible_logical_indices(AUXILIARY_INDEX)
    state = store.state
    store.close()

    assert state.head == 2
    np.testing.assert_array_equal(indices, np.asarray((0, 1), dtype=np.int64))


def test_dedicated_head_batches_remove_the_search_budget_term_from_the_main_batch() -> None:
    targets = (SearchBudgetTargetConfiguration(),)
    shared = resolve_auxiliary_losses(targets, 0, search_budget_dedicated_batches=False)
    dedicated = resolve_auxiliary_losses(targets, 0, search_budget_dedicated_batches=True)

    assert isinstance(shared[0], ResolvedSearchBudgetLoss)
    assert isinstance(dedicated[0], ResolvedSearchBudgetLoss)
    assert shared[0].weight == dedicated[0].weight == pytest.approx(0.2)
    assert shared[0].main_batch_weight == pytest.approx(0.2)
    assert dedicated[0].main_batch_weight == 0.0


def _batch_with_one_labelled_row() -> TrainingBatch:
    return TrainingBatch(
        states=torch.zeros((2, 1, 1, 1)),
        policy_targets=torch.ones((2, 1)),
        policy_legal_action_ids=torch.zeros((2, 1), dtype=torch.int64),
        wdl_targets=torch.tensor(((0.0, 1.0, 0.0),)).repeat(2, 1),
        root_values=torch.zeros(2),
        auxiliary_targets=(torch.full((2, 1), 0.5), torch.tensor(((0.8,), (0.0,)))),
        auxiliary_legal_action_ids=(
            torch.empty((2, 0), dtype=torch.int64),
            torch.empty((2, 0), dtype=torch.int64),
        ),
        auxiliary_eligibility=(torch.tensor((True, True)), torch.tensor((True, False))),
        sample_weights=torch.ones(2),
        source_model_generations=torch.zeros(2, dtype=torch.int64),
        source_created_at_seconds=torch.zeros(2),
    )


def _output_for(batch: TrainingBatch) -> TrainingModelOutput:
    return TrainingModelOutput(
        policy_logits=torch.zeros((2, 1)),
        wdl_logits=torch.zeros((2, 3)),
        auxiliary_logits=(torch.full((2, 1), 0.2), torch.zeros((2, 1))),
        features=torch.empty((2, 0)),
    )


def _objective(dedicated: bool) -> ResolvedTrainingObjective:
    return ResolvedTrainingObjective(
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        root_value_blend=0.0,
        auxiliary_losses=(
            ResolvedRemainingGameLengthLoss(weight=0.1),
            ResolvedSearchBudgetLoss(weight=0.2, dedicated_batches=dedicated),
        ),
    )


def test_the_main_batch_path_is_unchanged_apart_from_the_search_budget_contribution() -> None:
    batch = _batch_with_one_labelled_row()
    output = _output_for(batch)

    shared = _objective(dedicated=False).calculate_loss(output, batch)
    dedicated = _objective(dedicated=True).calculate_loss(output, batch)

    assert dedicated.policy.item() == pytest.approx(shared.policy.item())
    assert dedicated.wdl.item() == pytest.approx(shared.wdl.item())
    assert [value.item() for value in dedicated.auxiliary] == pytest.approx(
        [value.item() for value in shared.auxiliary]
    )
    assert dedicated.total.item() == pytest.approx(shared.total.item() - 0.2 * shared.auxiliary[1].item())


def test_a_labelled_batch_restores_the_search_budget_term() -> None:
    batch = _batch_with_one_labelled_row()
    output = _output_for(batch)
    objective = _objective(dedicated=True)

    ordinary = objective.calculate_loss(output, batch)
    labelled = objective.calculate_loss(output, batch, search_budget_labelled_batch=True)

    assert labelled.total.item() == pytest.approx(ordinary.total.item() + 0.2 * ordinary.auxiliary[1].item())


def _loader(
    description: ReplayDescription,
    global_batch_size: int,
    interval_optimizer_steps: int,
    optimizer_steps: int,
    world_size: int = 1,
    rank: int = 0,
) -> MappedReplayBatchLoader:
    return MappedReplayBatchLoader(
        replay=description,
        state=STUB_STATE,
        source_optimizer_step=3,
        optimizer_steps=optimizer_steps,
        global_batch_size=global_batch_size,
        world_size=world_size,
        rank=rank,
        sampler_seed=11,
        pin_memory=False,
        labelled_batches=SearchBudgetLabelledBatches(
            auxiliary_index=AUXILIARY_INDEX,
            interval_optimizer_steps=interval_optimizer_steps,
        ),
    )


def _labelled_description(tmp_path: Path, targets: tuple[float | None, ...]) -> ReplayDescription:
    path = tmp_path / 'replay.bin'
    store = _store_with_labels(path, targets)
    description = _description(path, store)
    store.close()
    return description


def test_every_row_of_a_labelled_batch_carries_a_label(tmp_path: Path) -> None:
    targets = tuple(None if index % 2 else 0.1 * (index % 8) + 0.05 for index in range(32))
    loader = _loader(
        _labelled_description(tmp_path, targets), global_batch_size=4, interval_optimizer_steps=2, optimizer_steps=4
    )

    batches = list(loader)

    assert loader.labelled_pool_rows == 16
    for batch_index, batch in enumerate(batches):
        eligibility = batch.auxiliary_eligibility[AUXILIARY_INDEX]
        assert loader.is_labelled_batch(batch_index) == (batch_index % 2 == 0)
        if loader.is_labelled_batch(batch_index):
            assert bool(torch.all(eligibility))
        else:
            assert not bool(torch.all(eligibility))


def test_a_labelled_pool_shorter_than_the_batch_repeats_rows_to_fill_it(tmp_path: Path) -> None:
    targets = (None, 0.25, None, 0.75, 0.5, None, None, None)
    loader = _loader(
        _labelled_description(tmp_path, targets), global_batch_size=4, interval_optimizer_steps=1, optimizer_steps=2
    )

    batches = list(loader)

    assert loader.labelled_pool_rows == 3
    assert loader.is_labelled_batch(0)
    assert all(len(batch) == 4 for batch in batches)
    assert all(bool(torch.all(batch.auxiliary_eligibility[AUXILIARY_INDEX])) for batch in batches)


def test_an_empty_labelled_pool_falls_back_to_a_uniform_sample(tmp_path: Path) -> None:
    loader = _loader(
        _labelled_description(tmp_path, (None,) * 8),
        global_batch_size=4,
        interval_optimizer_steps=1,
        optimizer_steps=2,
    )

    batches = list(loader)

    assert loader.labelled_pool_rows == 0
    assert not loader.is_labelled_batch(0)
    assert all(len(batch) == 4 for batch in batches)


def test_every_rank_agrees_on_which_batches_are_labelled_and_splits_the_same_draw(tmp_path: Path) -> None:
    description = _labelled_description(tmp_path, tuple(0.1 * (index % 8) + 0.05 for index in range(16)))
    first = _loader(
        description, global_batch_size=4, interval_optimizer_steps=2, optimizer_steps=4, world_size=2, rank=0
    )
    second = _loader(
        description, global_batch_size=4, interval_optimizer_steps=2, optimizer_steps=4, world_size=2, rank=1
    )

    decisions = tuple(first.is_labelled_batch(index) for index in range(4))

    assert decisions == tuple(second.is_labelled_batch(index) for index in range(4))
    assert decisions == (True, False, True, False)
    for left, right in zip(list(first), list(second), strict=True):
        assert len(left) == len(right) == 2
        assert not torch.equal(left.states, right.states)


def _small_network() -> Network:
    return Network(
        NetworkParams(num_layers=1, hidden_size=8, policy_head=DensePolicyHeadConfiguration(channels=2)),
        torch.device('cpu'),
        NetworkDimensions(channels=2, rows=3, columns=3, actions=2),
        auxiliary_heads=(
            RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=100.0),
            SearchBudgetHeadLayout(kind='search_budget'),
        ),
    )


def _budget_only_objective() -> ResolvedTrainingObjective:
    """Zero sibling weights isolate how far the search-budget term reaches on one ordinary optimizer step."""
    return ResolvedTrainingObjective(
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        root_value_blend=0.0,
        auxiliary_losses=(
            ResolvedRemainingGameLengthLoss(weight=0.0),
            ResolvedSearchBudgetLoss(weight=0.2, dedicated_batches=True),
        ),
    )


def _moved_parameters(model: Network, before: dict[str, torch.Tensor]) -> set[str]:
    return {name for name, value in model.named_parameters() if not torch.equal(before[name], value)}


def _one_training_step(model: Network, batch: TrainingBatch, labelled_batch: bool) -> set[str]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    output = model.training_output(batch.states)
    loss = _budget_only_objective().calculate_loss(output, batch, search_budget_labelled_batch=labelled_batch)
    loss.total.backward()
    optimizer.step()
    return _moved_parameters(model, before)


def test_a_labelled_batch_moves_the_budget_head_and_the_trunk_but_not_sibling_heads(tmp_path: Path) -> None:
    description = _labelled_description(tmp_path, tuple(0.1 * (index % 10) + 0.05 for index in range(16)))
    loader = _loader(description, global_batch_size=8, interval_optimizer_steps=1, optimizer_steps=1)
    batch = next(iter(loader))
    model = _small_network()
    model.train()

    moved = _one_training_step(model, batch, labelled_batch=True)

    assert any(name.startswith('start_block') for name in moved)
    assert any(name.startswith('auxiliary_head_modules.1') for name in moved)
    assert not any(name.startswith('auxiliary_head_modules.0') for name in moved)


def test_an_ordinary_batch_leaves_the_search_budget_head_untouched(tmp_path: Path) -> None:
    description = _labelled_description(tmp_path, tuple(0.1 * (index % 10) + 0.05 for index in range(16)))
    loader = _loader(description, global_batch_size=8, interval_optimizer_steps=1, optimizer_steps=1)
    batch = next(iter(loader))
    model = _small_network()
    model.train()

    moved = _one_training_step(model, batch, labelled_batch=False)

    assert moved == set()


def test_the_search_budget_head_is_located_once_for_replay_and_telemetry() -> None:
    heads = _layout().targets.auxiliary_heads

    assert search_budget_auxiliary_index(heads) == AUXILIARY_INDEX
    assert search_budget_auxiliary_index(()) is None

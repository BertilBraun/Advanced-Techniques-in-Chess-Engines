from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from src.games.contracts import WdlTarget
from src.games.representation import NetworkDimensions, PackedPlaneLayout, RepresentationDimensions
from src.replay.contracts import (
    EligibleRemainingGameLengthTarget,
    EligibleSearchBudgetTarget,
    IneligibleSearchBudgetTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.description import ReplayDescription
from src.replay.head_batch import SearchBudgetLabelPool, build_search_budget_head_batch
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.search_budget.configuration import SearchBudgetHeadTrainingConfiguration
from src.self_play.completed_game import SearchVisitCounts
from src.training.batch import TrainingBatch, TrainingModelOutput
from src.training.configuration import TrainingPrecision
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
from src.training.trainer.search_budget_head import SearchBudgetHeadTrainer, resolved_global_batch_rows

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
            normalized_target=normalized_target,
            raw_kl=0.125,
            prediction_logit=-0.5,
            predicted_quantile=0.25,
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


def test_every_row_of_a_head_batch_carries_a_label(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = _store_with_labels(path, (None, 0.25, None, 0.75, 0.5, None))
    description = _description(path, store)
    store.close()

    with SearchBudgetLabelPool(description, STUB_STATE, AUXILIARY_INDEX) as pool:
        generator = np.random.default_rng(7)
        selected = pool.select_logical_indices(generator, 3)
        batch = pool.batch(selected, np.zeros(3, dtype=np.int64))

    assert pool.size == 3
    assert sorted(int(index) for index in selected) == [1, 3, 4]
    assert len(batch) == 3
    assert torch.all((batch.targets > 0.0) & (batch.targets <= 1.0))


def test_a_head_batch_of_unlabelled_rows_is_refused(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = _store_with_labels(path, (None, 0.25))

    with pytest.raises(ValueError, match='labelled replay rows'):
        build_search_budget_head_batch(
            store,
            STUB_STATE,
            AUXILIARY_INDEX,
            np.asarray((0, 1), dtype=np.int64),
            np.zeros(2, dtype=np.int64),
        )
    store.close()


@pytest.mark.parametrize(
    ('labelled_rows', 'world_size', 'expected'),
    (
        (5_000, 1, 2_000),
        (5_000, 8, 2_000),
        (1_100, 8, 1_096),
        (300, 1, 300),
        (255, 1, 0),
        (0, 4, 0),
    ),
)
def test_a_short_labelled_pool_shrinks_the_head_batch_instead_of_padding_it(
    labelled_rows: int,
    world_size: int,
    expected: int,
) -> None:
    configuration = SearchBudgetHeadTrainingConfiguration()

    assert resolved_global_batch_rows(configuration, labelled_rows, world_size) == expected


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


def test_a_head_step_moves_the_head_and_leaves_the_shared_trunk_untouched(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = _store_with_labels(path, tuple(0.1 * (index % 10) + 0.05 for index in range(64)))
    description = _description(path, store)
    store.close()
    model = _small_network()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trunk_before = tuple(parameter.detach().clone() for parameter in model.start_block.parameters())
    head_before = tuple(parameter.detach().clone() for parameter in model.auxiliary_head_modules[0].parameters())
    budget_before = tuple(
        parameter.detach().clone() for parameter in model.auxiliary_head_modules[AUXILIARY_INDEX].parameters()
    )

    with SearchBudgetHeadTrainer(
        replay=description,
        state=STUB_STATE,
        auxiliary_index=AUXILIARY_INDEX,
        configuration=SearchBudgetHeadTrainingConfiguration(batch_size=32, minimum_labelled_rows=8),
        loss_weight=0.2,
        model=model,
        optimizer=optimizer,
        device=torch.device('cpu'),
        precision=TrainingPrecision.FLOAT32,
        maximum_gradient_norm=0.5,
        world_size=1,
        rank=0,
        sampler_seed=11,
        source_optimizer_step=3,
    ) as trainer:
        assert trainer.global_batch_rows == 32
        trainer.train_step(batch_index=0, capture_distribution=True)
        statistics = trainer.statistics()
        distribution = trainer.distribution

    for before, after in zip(trunk_before, model.start_block.parameters(), strict=True):
        assert torch.equal(before, after)
    for before, after in zip(head_before, model.auxiliary_head_modules[0].parameters(), strict=True):
        assert torch.equal(before, after)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            budget_before,
            model.auxiliary_head_modules[AUXILIARY_INDEX].parameters(),
            strict=True,
        )
    )
    assert statistics.labelled_pool_rows == 64
    assert statistics.optimizer_steps == 1
    assert statistics.global_batch_rows == 32
    assert 0.0 < statistics.target_mean < 1.0
    assert statistics.target_standard_deviation > 0.0
    assert distribution is not None
    assert len(distribution.target) == 32


def test_a_labelled_pool_below_the_floor_disables_head_training(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = _store_with_labels(path, (None, 0.25, None, 0.75))
    description = _description(path, store)
    store.close()
    model = _small_network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    with SearchBudgetHeadTrainer(
        replay=description,
        state=STUB_STATE,
        auxiliary_index=AUXILIARY_INDEX,
        configuration=SearchBudgetHeadTrainingConfiguration(batch_size=32, minimum_labelled_rows=8),
        loss_weight=0.2,
        model=model,
        optimizer=optimizer,
        device=torch.device('cpu'),
        precision=TrainingPrecision.FLOAT32,
        maximum_gradient_norm=0.5,
        world_size=1,
        rank=0,
        sampler_seed=11,
        source_optimizer_step=3,
    ) as trainer:
        assert trainer.global_batch_rows == 0
        assert not trainer.due_at_step(0)
        statistics = trainer.statistics()

    assert statistics.labelled_pool_rows == 2
    assert statistics.optimizer_steps == 0


def test_the_search_budget_head_is_located_once_for_replay_and_telemetry() -> None:
    heads = _layout().targets.auxiliary_heads

    assert search_budget_auxiliary_index(heads) == AUXILIARY_INDEX
    assert search_budget_auxiliary_index(()) is None

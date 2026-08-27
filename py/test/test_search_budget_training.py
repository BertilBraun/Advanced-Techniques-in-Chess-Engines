from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from src.games.contracts import WdlTarget
from src.games.representation import PackedPlaneLayout
from src.replay.batch_loader import build_dense_targets
from src.replay.contracts import (
    EligibleSearchBudgetTarget,
    IneligibleSearchBudgetTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.self_play.completed_game import SearchVisitCounts
from src.training.batch import TrainingBatch, TrainingModelOutput
from src.training.objective import ResolvedSearchBudgetLoss, ResolvedTrainingObjective
from src.training.targets import SearchBudgetHeadLayout, SearchBudgetTargetConfiguration, TrainingTargetLayout


class _DenseTargetState:
    @property
    def action_size(self) -> int:
        return 2

    @property
    def augmentation_count(self) -> int:
        return 1

    @property
    def action_permutations(self) -> npt.NDArray[np.uint16]:
        return np.asarray(((0, 1),), dtype=np.uint16)


def _training_batch(eligibility: torch.Tensor, sample_weights: torch.Tensor) -> TrainingBatch:
    row_count = len(eligibility)
    return TrainingBatch(
        states=torch.zeros((row_count, 1, 1, 1)),
        policy_targets=torch.ones((row_count, 1)),
        policy_legal_action_ids=torch.zeros((row_count, 1), dtype=torch.int64),
        wdl_targets=torch.tensor(((0.0, 1.0, 0.0),)).repeat(row_count, 1),
        root_values=torch.zeros(row_count),
        auxiliary_targets=(torch.zeros((row_count, 1)),),
        auxiliary_legal_action_ids=(torch.empty((row_count, 0), dtype=torch.int64),),
        auxiliary_eligibility=(eligibility,),
        sample_weights=sample_weights,
        source_model_generations=torch.zeros(row_count, dtype=torch.int64),
        source_created_at_seconds=torch.zeros(row_count),
    )


def _model_output(search_budget_logits: torch.Tensor) -> TrainingModelOutput:
    row_count = len(search_budget_logits)
    return TrainingModelOutput(
        policy_logits=torch.zeros((row_count, 1)),
        wdl_logits=torch.zeros((row_count, 3)),
        auxiliary_logits=(search_budget_logits.reshape(-1, 1),),
        features=torch.empty((row_count, 0)),
    )


def _objective() -> ResolvedTrainingObjective:
    return ResolvedTrainingObjective(
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        root_value_blend=0.0,
        auxiliary_losses=(ResolvedSearchBudgetLoss(weight=0.2),),
    )


def test_search_budget_configuration_defaults_to_documented_loss_weight() -> None:
    configuration = SearchBudgetTargetConfiguration()

    assert configuration.loss_weight.value_at(0) == pytest.approx(0.2)


def test_search_budget_uses_weight_normalized_masked_plain_l1() -> None:
    batch = _training_batch(torch.tensor((True, True, False)), torch.tensor((1.0, 3.0, 100.0)))
    probabilities = torch.tensor((0.1, 0.9, 0.999))
    output = _model_output(torch.logit(probabilities))

    loss = _objective().calculate_loss(output, batch)

    assert loss.auxiliary[0].item() == pytest.approx(0.7)
    assert loss.total.item() == pytest.approx(0.14)


def test_search_budget_batch_without_eligible_labels_has_finite_zero_loss() -> None:
    batch = _training_batch(torch.tensor((False, False)), torch.tensor((1.0, 2.0)))
    output = _model_output(torch.tensor((float('nan'), float('nan'))))

    loss = _objective().calculate_loss(output, batch)

    assert loss.auxiliary[0].isfinite()
    assert loss.auxiliary[0].item() == 0.0
    assert loss.total.isfinite()


def test_eligible_and_ineligible_search_budget_targets_round_trip_with_provenance(tmp_path: Path) -> None:
    packed_planes = PackedPlaneLayout(board_size=3, binary_plane_count=1, scalar_count=1)
    layout = ReplayLayout(
        packed_planes=packed_planes,
        targets=TrainingTargetLayout(
            action_size=2,
            wdl_size=3,
            auxiliary_heads=(SearchBudgetHeadLayout(kind='search_budget'),),
        ),
        maximum_policy_entries=2,
        maximum_legal_actions=2,
    )
    policy = SparsePolicyTarget(
        visits=SearchVisitCounts(action_ids=(0, 1), visit_counts=(3, 1)),
        legal_action_ids=(0, 1),
    )
    eligible = EligibleSearchBudgetTarget(
        normalized_target=0.75,
        raw_kl=0.125,
        prediction_logit=-0.5,
        predicted_quantile=0.25,
        source_generation=12,
        model_generation=13,
        inference_model_sha256='a' * 64,
    )

    def sample(target: EligibleSearchBudgetTarget | IneligibleSearchBudgetTarget) -> ReplaySample:
        return ReplaySample(
            encoded_state=packed_planes.value(bytes(packed_planes.payload_bytes)),
            policy=policy,
            wdl_target=WdlTarget(win=0.0, draw=1.0, loss=0.0),
            root_value=0.0,
            auxiliary_targets=(target,),
            sample_weight=1.0,
            source_model_generation=12,
            source_created_at_seconds=10.0,
        )

    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=2, logical_capacity=2)
    store.extend((sample(eligible), sample(IneligibleSearchBudgetTarget())))

    columns = store.gather_logical(np.asarray((0, 1), dtype=np.int64))
    dense = build_dense_targets(columns, layout, _DenseTargetState(), np.asarray((0, 0), dtype=np.int64))
    stored_eligible = store.sample_at(0).auxiliary_targets[0]
    stored_ineligible = store.sample_at(1).auxiliary_targets[0]
    store.close()

    assert isinstance(stored_eligible, EligibleSearchBudgetTarget)
    assert stored_eligible.normalized_target == pytest.approx(0.75)
    assert stored_eligible.raw_kl == pytest.approx(0.125)
    assert stored_eligible.prediction_logit == pytest.approx(-0.5)
    assert stored_eligible.predicted_quantile == pytest.approx(0.25)
    assert stored_eligible.source_generation == 12
    assert stored_eligible.model_generation == 13
    assert stored_eligible.inference_model_sha256 == 'a' * 64
    assert isinstance(stored_ineligible, IneligibleSearchBudgetTarget)
    np.testing.assert_array_equal(dense.auxiliary[0], np.asarray(((0.75,), (0.0,)), dtype=np.float32))
    np.testing.assert_array_equal(dense.auxiliary_eligibility[0], np.asarray((True, False)))

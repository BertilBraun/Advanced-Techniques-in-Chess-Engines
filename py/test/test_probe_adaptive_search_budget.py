from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from tools.measure_policy_target_fidelity import FixedBudgetRecord, LabelCandidate, PerPositionReport, PositionRecord
from tools.probe_adaptive_search_budget import (
    AllocationData,
    Arguments,
    _allocation_data,
    _extract_features,
    _flat_divergence,
    _gain_fraction,
    _oracle_allocation,
    _ranked_allocation,
    fit_out_of_fold_predictions,
    quantile_ranks,
)
from torch import Tensor, nn


def test_quantile_ranks_average_ties_and_span_unit_interval() -> None:
    values = np.asarray((3.0, 1.0, 2.0, 2.0, 4.0), dtype=np.float64)

    assert quantile_ranks(values) == pytest.approx((0.75, 0.0, 0.375, 0.375, 1.0))


def _position(label: float, divergences: tuple[float, float, float]) -> PositionRecord:
    return PositionRecord(
        fen='position',
        policy_correction=0.0,
        value_correction=0.0,
        search_correction_target=0.0,
        budgets=tuple(
            FixedBudgetRecord(
                visits=visits,
                root_value=0.0,
                kullback_leibler=divergence,
                total_variation=0.0,
                top_visit_share=1.0,
                top_two_margin=1.0,
                leader_matches_reference=True,
            )
            for visits, divergence in zip((100, 600, 2_400), divergences, strict=True)
        ),
        label_candidates=(
            LabelCandidate(
                baseline_visits=600,
                depth_visits=2_400,
                total_variation=0.0,
                kullback_leibler=label,
            ),
        ),
    )


def test_ranked_label_allocation_captures_oracle_gain() -> None:
    records = tuple(
        [_position(0.01 + index * 0.001, (0.10, 0.09, 0.09)) for index in range(8)]
        + [_position(1.0 + index, (1.00, 0.80, 0.00)) for index in range(2)]
    )
    report = PerPositionReport(
        schema_version=1,
        source_revision='a' * 40,
        model_sha256='b' * 64,
        generation=1,
        reference_visits=2_400,
        parallel_searches=1,
        records=records,
    )
    data = _allocation_data(report)
    flat = _flat_divergence(data, 600)
    oracle = _oracle_allocation(data, 600.0)
    signal = np.asarray([record.label_candidates[0].kullback_leibler for record in records], dtype=np.float64)
    ranked = _ranked_allocation(data, oracle.assigned_budgets, signal)

    assert ranked.mean_visits == pytest.approx(oracle.mean_visits)
    assert _gain_fraction(flat, oracle, ranked) == pytest.approx(1.0)


def _arguments(tmp_path: Path) -> Arguments:
    return Arguments(
        model=tmp_path / 'model.pt',
        per_position=tmp_path / 'positions.json.gz',
        output=tmp_path / 'probe.json',
        device=torch.device('cpu'),
        baseline_visits=600,
        depth_visits=2_400,
        folds=3,
        epochs=80,
        batch_size=16,
        feature_batch_size=16,
        learning_rate=0.02,
        weight_decay=0.0,
        bootstrap_samples=20,
        random_orderings=5,
        seed=42,
    )


def test_frozen_scalar_head_learns_signal_out_of_fold(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn((60, 4, 2, 2), generator=generator)
    labels = quantile_ranks(features[:, 0].mean(dim=(1, 2)).numpy().astype(np.float64))

    predictions, folds = fit_out_of_fold_predictions(features, labels, _arguments(tmp_path))

    correlation = np.corrcoef(predictions, labels)[0, 1]
    assert correlation > 0.6
    assert len(folds) == 3
    assert sum(fold.held_out_positions for fold in folds) == len(features)


def test_allocation_data_rejects_mismatched_shape() -> None:
    with pytest.raises(ValueError, match='one column per budget'):
        AllocationData(
            budgets=np.asarray((100.0, 600.0), dtype=np.float64),
            monotone_divergences=np.zeros((4, 3), dtype=np.float64),
        )


class _ScriptedBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.start_block = nn.Sequential(nn.Conv2d(2, 3, kernel_size=1), nn.ReLU())
        self.backbone = nn.ModuleList((nn.Sequential(nn.Conv2d(3, 3, kernel_size=1), nn.ReLU()),))
        self.finish_block = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.start_block(inputs)
        for block in self.backbone:
            features = block(features)
        return self.finish_block(features)


def test_extract_features_uses_scripted_backbone_modules() -> None:
    model = torch.jit.script(_ScriptedBackbone())
    inputs = torch.randn((3, 2, 2, 2), generator=torch.Generator().manual_seed(9))

    assert torch.equal(_extract_features(model, inputs), model(inputs))

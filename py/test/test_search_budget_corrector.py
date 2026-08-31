from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from src.search_budget.corrector import (
    CurveCorrectorNetwork,
    LoadedCurveCorrector,
    export_corrector,
    fit_curve_corrector,
)
from src.search_budget.policy import (
    BUDGET_CURVE_POINTS,
    CORRECTOR_INPUT_FEATURES,
    BudgetSelectionFeatures,
    SearchBudgetPolicy,
    select_budget_index,
)
from test_helpers.search_budget_records import analysis_records as _records


def learned_policy(lagrange_multiplier: float = 1.0) -> SearchBudgetPolicy:
    return SearchBudgetPolicy(
        lagrange_multiplier=lagrange_multiplier,
        corrector_path=None,
        corrector_sha256=None,
        apply_learned=True,
    )


def _synthetic_records(count: int = 1024, noise_scale: float = 0.0, seed: int = 7) -> np.ndarray:
    random = np.random.default_rng(seed)
    predicted = random.normal(-2.0, 1.0, size=(count, BUDGET_CURVE_POINTS))
    top_visit_share = random.uniform(0.2, 1.0, size=count)
    policy_entropy = random.uniform(0.0, 3.0, size=count)
    ply = random.integers(0, 200, size=count)
    # The true residual depends on the observables the corrector injects, not on capacity alone.
    residual = (
        0.8 + 0.3 * predicted - 1.5 * top_visit_share[:, None] + 0.4 * policy_entropy[:, None] + 0.002 * ply[:, None]
    )
    target = predicted + residual + noise_scale * random.normal(size=(count, BUDGET_CURVE_POINTS))
    return _records(predicted, target, top_visit_share, policy_entropy, ply)


def _export(network: CurveCorrectorNetwork, tmp_path: Path) -> Path:
    path = tmp_path / 'corrector.jit.pt'
    export_corrector(network, path)
    return path


def test_a_learnable_residual_improves_the_held_out_residual_and_is_applied() -> None:
    fit = fit_curve_corrector(_synthetic_records())
    assert fit.applied
    assert fit.network is not None
    assert fit.corrected_holdout_residual < fit.uncorrected_holdout_residual


def test_fit_export_load_round_trip_preserves_the_correction(tmp_path: Path) -> None:
    fit = fit_curve_corrector(_synthetic_records())
    assert fit.network is not None
    loaded = LoadedCurveCorrector.load(_export(fit.network, tmp_path))
    curve = tuple(float(value) for value in np.linspace(-4.0, 0.0, BUDGET_CURVE_POINTS))
    features = BudgetSelectionFeatures(
        top_visit_share=0.6, policy_entropy=1.2, ply=40, baseline_visits=400, source_generation=12
    )
    corrected = loaded(curve, features)
    inputs = torch.tensor([[*curve, 0.6, 1.2, 40.0, 400.0, 12.0]], dtype=torch.float32)
    with torch.no_grad():
        expected = fit.network(inputs)[0]
    assert corrected == pytest.approx(
        tuple(float(value) + float(delta) for value, delta in zip(curve, expected, strict=True))
    )


def test_loading_rejects_a_digest_mismatch(tmp_path: Path) -> None:
    fit = fit_curve_corrector(_synthetic_records())
    assert fit.network is not None
    path = _export(fit.network, tmp_path)
    with pytest.raises(ValueError, match='digest'):
        LoadedCurveCorrector.load(path, expected_sha256='0' * 64)


def test_a_corrector_that_learned_a_deep_advantage_steers_selection(tmp_path: Path) -> None:
    # The training residual makes the deepest point five units of log-KL better than predicted, so
    # the corrected Lagrangian argmin must move off the cheapest point the raw curve would select.
    random = np.random.default_rng(11)
    count = 2048
    predicted = np.zeros((count, BUDGET_CURVE_POINTS))
    target = predicted.copy()
    target[:, -1] -= 5.0
    records = _records(
        predicted,
        target,
        random.uniform(0.2, 1.0, size=count),
        random.uniform(0.0, 3.0, size=count),
        random.integers(0, 200, size=count),
    )
    fit = fit_curve_corrector(records)
    assert fit.network is not None
    loaded = LoadedCurveCorrector.load(_export(fit.network, tmp_path))
    curve = (0.0,) * BUDGET_CURVE_POINTS
    features = BudgetSelectionFeatures(
        top_visit_share=0.6, policy_entropy=1.5, ply=40, baseline_visits=400, source_generation=12
    )
    assert select_budget_index(curve, learned_policy(0.01), features) == 0
    assert select_budget_index(curve, learned_policy(0.01), features, loaded) == BUDGET_CURVE_POINTS - 1


def test_the_guard_rejects_a_fit_that_cannot_improve_on_perfect_predictions() -> None:
    random = np.random.default_rng(3)
    count = 1024
    predicted = random.normal(-2.0, 1.0, size=(count, BUDGET_CURVE_POINTS))
    records = _records(
        predicted,
        predicted,
        random.uniform(0.2, 1.0, size=count),
        random.uniform(0.0, 3.0, size=count),
        random.integers(0, 200, size=count),
    )
    fit = fit_curve_corrector(records)
    assert not fit.applied
    assert fit.network is None
    assert fit.rejection_reason is not None
    assert 'improve' in fit.rejection_reason


def test_fitting_rejects_nonfinite_inputs_and_tiny_windows() -> None:
    records = _synthetic_records()
    records['policy_kl'][5, 2] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        fit_curve_corrector(records)
    with pytest.raises(ValueError, match='split'):
        fit_curve_corrector(_synthetic_records(count=4))


def test_export_rejects_a_network_with_nonfinite_parameters(tmp_path: Path) -> None:
    network = CurveCorrectorNetwork()
    with torch.no_grad():
        network.layers[-1].bias.fill_(float('inf'))
    with pytest.raises(ValueError, match='finite'):
        export_corrector(network, tmp_path / 'corrupt.jit.pt')


def test_the_network_standardises_with_its_folded_buffers() -> None:
    network = CurveCorrectorNetwork()
    with torch.no_grad():
        network.feature_mean.fill_(2.0)
        network.feature_scale.fill_(4.0)
        raw = torch.full((1, CORRECTOR_INPUT_FEATURES), 6.0)
        expected = network.layers(torch.ones((1, CORRECTOR_INPUT_FEATURES)))
        actual = network(raw)
    assert torch.allclose(actual, expected)

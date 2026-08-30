from __future__ import annotations

import math

import numpy as np
import pytest
from src.search_budget.analysis_log import ANALYSIS_RECORD_DTYPE
from src.search_budget.calibrator import IDENTITY_CALIBRATOR, calibrator_rejection_reason, fit_linear_calibrator
from src.search_budget.policy import (
    BUDGET_CURVE_POINTS,
    LOG_KL_EPSILON,
    BudgetSelectionFeatures,
    SearchBudgetPolicy,
    calibrate_curve,
)


def _records(
    count: int,
    predicted: np.ndarray,
    target_log_kl: np.ndarray,
    top_visit_share: np.ndarray,
    policy_entropy: np.ndarray,
    ply: np.ndarray,
    baseline_visits: int = 400,
) -> np.ndarray:
    records = np.zeros(count, dtype=ANALYSIS_RECORD_DTYPE)
    records['predicted_curve'] = predicted.astype(np.float32)
    records['policy_kl'] = (np.exp(target_log_kl) - LOG_KL_EPSILON).clip(min=0.0).astype(np.float32)
    records['top_visit_share'] = top_visit_share.astype(np.float32)
    records['policy_entropy'] = policy_entropy.astype(np.float32)
    records['ply'] = ply.astype(np.uint32)
    records['baseline_visits'] = baseline_visits
    return records


def _synthetic_records(count: int = 512, noise_scale: float = 0.0, seed: int = 7) -> np.ndarray:
    random = np.random.default_rng(seed)
    predicted = random.normal(-2.0, 1.0, size=(count, BUDGET_CURVE_POINTS))
    top_visit_share = random.uniform(0.2, 1.0, size=count)
    policy_entropy = random.uniform(0.0, 3.0, size=count)
    ply = random.integers(0, 200, size=count)
    # The true residual is an affine function of the observables the calibrator uses.
    residual = (
        0.8 + 0.3 * predicted - 1.5 * top_visit_share[:, None] + 0.4 * policy_entropy[:, None] + 0.002 * ply[:, None]
    )
    target = predicted + residual + noise_scale * random.normal(size=(count, BUDGET_CURVE_POINTS))
    return _records(count, predicted, target, top_visit_share, policy_entropy, ply)


def test_a_noiseless_affine_residual_is_recovered_and_applies_through_the_policy() -> None:
    records = _synthetic_records(noise_scale=0.0)
    fit = fit_linear_calibrator(records, ridge_coefficient=1e-6)
    assert fit.applied
    assert fit.calibrated_squared_residual < 1e-3 * fit.uncalibrated_squared_residual

    policy = SearchBudgetPolicy(
        lagrange_multiplier=0.5,
        calibration_bias=fit.coefficients.bias,
        calibration_weights=fit.coefficients.weights,
        apply_learned=True,
    )
    row = records[3]
    features = BudgetSelectionFeatures(
        top_visit_share=float(row['top_visit_share']),
        policy_entropy=float(row['policy_entropy']),
        ply=int(row['ply']),
        baseline_visits=int(row['baseline_visits']),
    )
    calibrated = calibrate_curve(tuple(float(value) for value in row['predicted_curve']), policy, features)
    target = np.log(np.asarray(row['policy_kl'], dtype=np.float64) + LOG_KL_EPSILON)
    assert calibrated == pytest.approx(tuple(target), abs=1e-3)


def test_a_noisy_fit_still_reduces_the_in_window_residual() -> None:
    fit = fit_linear_calibrator(_synthetic_records(noise_scale=0.3), ridge_coefficient=1.0)
    assert fit.applied
    assert fit.calibrated_squared_residual < fit.uncalibrated_squared_residual


def test_standardisation_is_folded_into_the_shipped_coefficients() -> None:
    # baseline_visits is constant across the window, so its folded weight must be exactly zero
    # and its effect absorbed into the bias.
    fit = fit_linear_calibrator(_synthetic_records(), ridge_coefficient=1e-6)
    assert all(row[4] == 0.0 for row in fit.coefficients.weights)


def test_a_nonfinite_fit_ships_the_identity_calibration() -> None:
    records = _synthetic_records()
    records['policy_kl'][5, 2] = float('nan')
    fit = fit_linear_calibrator(records, ridge_coefficient=1.0)
    assert not fit.applied
    assert fit.rejection_reason is not None
    assert 'non-finite' in fit.rejection_reason
    assert fit.coefficients == IDENTITY_CALIBRATOR


def test_the_guard_rejects_a_fit_that_does_not_reduce_the_residual() -> None:
    assert calibrator_rejection_reason(IDENTITY_CALIBRATOR, 1.0, 1.0) is not None
    assert calibrator_rejection_reason(IDENTITY_CALIBRATOR, 1.0, 1.5) is not None
    assert calibrator_rejection_reason(IDENTITY_CALIBRATOR, 0.0, 0.0) is not None
    assert calibrator_rejection_reason(IDENTITY_CALIBRATOR, 1.0, 0.5) is None


def test_identity_coefficients_are_all_zero_and_finite() -> None:
    assert all(value == 0.0 for value in IDENTITY_CALIBRATOR.bias)
    assert all(math.isfinite(value) and value == 0.0 for row in IDENTITY_CALIBRATOR.weights for value in row)


def test_fitting_rejects_an_empty_window_and_a_nonpositive_ridge() -> None:
    with pytest.raises(ValueError, match='at least one'):
        fit_linear_calibrator(np.zeros(0, dtype=ANALYSIS_RECORD_DTYPE), ridge_coefficient=1.0)
    with pytest.raises(ValueError, match='ridge'):
        fit_linear_calibrator(_synthetic_records(count=8), ridge_coefficient=0.0)

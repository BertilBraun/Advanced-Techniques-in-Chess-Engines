from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from src.search_budget.analysis_log import ANALYSIS_RECORD_DTYPE
from src.search_budget.policy import (
    BUDGET_CURVE_POINTS,
    CALIBRATION_FEATURE_COUNT,
    IDENTITY_CALIBRATION_BIAS,
    IDENTITY_CALIBRATION_WEIGHTS,
    LOG_KL_EPSILON,
)


@dataclass(frozen=True)
class CalibratorCoefficients:
    bias: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]


IDENTITY_CALIBRATOR = CalibratorCoefficients(
    bias=IDENTITY_CALIBRATION_BIAS,
    weights=IDENTITY_CALIBRATION_WEIGHTS,
)


@dataclass(frozen=True)
class CalibratorFit:
    coefficients: CalibratorCoefficients
    applied: bool
    rejection_reason: str | None
    uncalibrated_squared_residual: float
    calibrated_squared_residual: float


def fit_linear_calibrator(
    records: npt.NDArray[np.void],
    ridge_coefficient: float,
) -> CalibratorFit:
    """Ridge-fit per-grid-point linear corrections of the predicted log-KL curve.

    Features are standardised over the fitting window and the standardisation is folded back into
    the shipped coefficients, so applying them is a plain affine map with no runtime normalisation.
    """
    if records.dtype != ANALYSIS_RECORD_DTYPE:
        raise ValueError('Calibrator fitting requires the fixed analysis record dtype.')
    if records.shape[0] == 0:
        raise ValueError('Calibrator fitting requires at least one labelled position.')
    if not math.isfinite(ridge_coefficient) or ridge_coefficient <= 0.0:
        raise ValueError('The calibrator ridge coefficient must be finite and positive.')

    predicted = np.asarray(records['predicted_curve'], dtype=np.float64)
    targets = np.log(np.asarray(records['policy_kl'], dtype=np.float64) + LOG_KL_EPSILON)
    shared = np.stack(
        [
            np.asarray(records['top_visit_share'], dtype=np.float64),
            np.asarray(records['policy_entropy'], dtype=np.float64),
            np.asarray(records['ply'], dtype=np.float64),
            np.asarray(records['baseline_visits'], dtype=np.float64),
        ],
        axis=1,
    )

    bias = [0.0] * BUDGET_CURVE_POINTS
    weights = [[0.0] * CALIBRATION_FEATURE_COUNT for _ in range(BUDGET_CURVE_POINTS)]
    uncalibrated_residual = 0.0
    calibrated_residual = 0.0
    for index in range(BUDGET_CURVE_POINTS):
        features = np.concatenate([predicted[:, index : index + 1], shared], axis=1)
        residual = targets[:, index] - predicted[:, index]
        means = features.mean(axis=0)
        deviations = features.std(axis=0)
        informative = deviations > 0.0
        standardized = np.zeros_like(features)
        standardized[:, informative] = (features[:, informative] - means[informative]) / deviations[informative]

        intercept = float(residual.mean())
        centered = residual - intercept
        gram = standardized.T @ standardized + ridge_coefficient * np.eye(CALIBRATION_FEATURE_COUNT)
        beta = np.linalg.solve(gram, standardized.T @ centered)

        folded = np.zeros(CALIBRATION_FEATURE_COUNT)
        folded[informative] = beta[informative] / deviations[informative]
        folded_bias = intercept - float(folded @ means)

        uncalibrated_residual += float(residual @ residual)
        fitted = folded_bias + features @ folded
        calibrated_residual += float((residual - fitted) @ (residual - fitted))
        bias[index] = folded_bias
        weights[index] = [float(value) for value in folded]

    coefficients = CalibratorCoefficients(
        bias=tuple(bias),
        weights=tuple(tuple(row) for row in weights),
    )
    rejection_reason = calibrator_rejection_reason(coefficients, uncalibrated_residual, calibrated_residual)
    return CalibratorFit(
        coefficients=IDENTITY_CALIBRATOR if rejection_reason is not None else coefficients,
        applied=rejection_reason is None,
        rejection_reason=rejection_reason,
        uncalibrated_squared_residual=uncalibrated_residual,
        calibrated_squared_residual=calibrated_residual,
    )


def calibrator_rejection_reason(
    coefficients: CalibratorCoefficients,
    uncalibrated_residual: float,
    calibrated_residual: float,
) -> str | None:
    """A calibrator that makes predictions worse must not reach production, so fail toward identity."""
    flattened = (*coefficients.bias, *(value for row in coefficients.weights for value in row))
    if any(not math.isfinite(value) for value in flattened):
        return 'the fitted calibrator has a non-finite coefficient'
    if not math.isfinite(calibrated_residual) or calibrated_residual >= uncalibrated_residual:
        return 'the fitted calibrator does not reduce the in-window residual'
    return None

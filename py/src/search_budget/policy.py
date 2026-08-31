from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import Field, model_validator
from src.util.frozen_model import FrozenModel

BUDGET_CURVE_MULTIPLES: tuple[float, ...] = (0.125, 0.2, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5, 2.0, 3.0, 4.0)
BUDGET_CURVE_POINTS = 10
BASELINE_CURVE_INDEX = BUDGET_CURVE_MULTIPLES.index(1.0)
HALF_DEEP_CURVE_INDEX = BUDGET_CURVE_MULTIPLES.index(4.0)
LOG_KL_EPSILON = 1e-6

# Per grid point: the point's own raw prediction, top visit share, policy entropy, ply, baseline visits.
CALIBRATION_FEATURE_COUNT = 5

IDENTITY_CALIBRATION_BIAS: tuple[float, ...] = (0.0,) * BUDGET_CURVE_POINTS
IDENTITY_CALIBRATION_WEIGHTS: tuple[tuple[float, ...], ...] = tuple(
    (0.0,) * CALIBRATION_FEATURE_COUNT for _ in range(BUDGET_CURVE_POINTS)
)


@dataclass(frozen=True)
class BudgetSelectionFeatures:
    top_visit_share: float
    policy_entropy: float
    ply: int
    baseline_visits: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.top_visit_share) or not math.isfinite(self.policy_entropy):
            raise ValueError('Budget selection features must be finite.')
        if self.ply < 0 or self.baseline_visits <= 0:
            raise ValueError('Budget selection features require a nonnegative ply and positive baseline visits.')


class SearchBudgetPolicy(FrozenModel):
    lagrange_multiplier: float
    calibration_bias: tuple[float, ...] = Field(min_length=BUDGET_CURVE_POINTS, max_length=BUDGET_CURVE_POINTS)
    calibration_weights: tuple[tuple[float, ...], ...] = Field(
        min_length=BUDGET_CURVE_POINTS,
        max_length=BUDGET_CURVE_POINTS,
    )
    apply_learned: bool

    @model_validator(mode='after')
    def validate_policy(self) -> SearchBudgetPolicy:
        if not math.isfinite(self.lagrange_multiplier) or self.lagrange_multiplier < 0.0:
            raise ValueError('The search-budget Lagrange multiplier must be finite and nonnegative.')
        if any(not math.isfinite(value) for value in self.calibration_bias):
            raise ValueError('Search-budget calibration biases must be finite.')
        for row in self.calibration_weights:
            if len(row) != CALIBRATION_FEATURE_COUNT:
                raise ValueError('Search-budget calibration weights need one coefficient per feature.')
            if any(not math.isfinite(value) for value in row):
                raise ValueError('Search-budget calibration weights must be finite.')
        return self


def disabled_policy() -> SearchBudgetPolicy:
    return SearchBudgetPolicy(
        lagrange_multiplier=0.0,
        calibration_bias=IDENTITY_CALIBRATION_BIAS,
        calibration_weights=IDENTITY_CALIBRATION_WEIGHTS,
        apply_learned=False,
    )


def deep_label_visit_limit(baseline_new_visits: int) -> int:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    return 8 * baseline_new_visits


def grid_visit_counts(baseline_new_visits: int) -> tuple[int, ...]:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    return tuple(max(1, int(math.floor(multiple * baseline_new_visits + 0.5))) for multiple in BUDGET_CURVE_MULTIPLES)


def grid_checkpoint_visits(baseline_new_visits: int) -> tuple[int, ...]:
    return tuple(sorted(set(grid_visit_counts(baseline_new_visits))))


def log_kl_curve(kl_values: tuple[float, ...]) -> tuple[float, ...]:
    if len(kl_values) != BUDGET_CURVE_POINTS:
        raise ValueError('A search-budget curve label requires one KL value per grid point.')
    if any(not math.isfinite(value) or value < 0.0 for value in kl_values):
        raise ValueError('Search-budget curve KL values must be finite and nonnegative.')
    return tuple(math.log(value + LOG_KL_EPSILON) for value in kl_values)


def project_non_increasing(values: tuple[float, ...]) -> tuple[float, ...]:
    """Running minimum from the cheapest budget upward, so more search never predicts more error.

    Sweeping the other way takes a suffix minimum, which is nondecreasing and therefore flattens an
    already well-formed curve to its deepest value.
    """
    if len(values) != BUDGET_CURVE_POINTS:
        raise ValueError('Isotonic projection requires one value per grid point.')
    projected = list(values)
    for index in range(1, BUDGET_CURVE_POINTS):
        projected[index] = min(projected[index], projected[index - 1])
    return tuple(projected)


def calibrate_curve(
    predicted_curve: tuple[float, ...],
    policy: SearchBudgetPolicy,
    features: BudgetSelectionFeatures,
) -> tuple[float, ...]:
    if len(predicted_curve) != BUDGET_CURVE_POINTS:
        raise ValueError('Curve calibration requires one prediction per grid point.')
    if any(not math.isfinite(value) for value in predicted_curve):
        raise ValueError('Curve calibration requires finite predictions.')
    return tuple(
        predicted_curve[index]
        + policy.calibration_bias[index]
        + policy.calibration_weights[index][0] * predicted_curve[index]
        + policy.calibration_weights[index][1] * features.top_visit_share
        + policy.calibration_weights[index][2] * features.policy_entropy
        + policy.calibration_weights[index][3] * float(features.ply)
        + policy.calibration_weights[index][4] * float(features.baseline_visits)
        for index in range(BUDGET_CURVE_POINTS)
    )


def select_budget_index(
    predicted_curve: tuple[float, ...],
    policy: SearchBudgetPolicy,
    features: BudgetSelectionFeatures,
) -> int:
    """Lagrangian selection: the grid point minimising predicted raw KL plus dual-priced spend.

    The objective works in raw KL space because the run-level quantity being minimised is a sum of
    KLs, not of logs. Ties go to the cheapest grid point.
    """
    projected = project_non_increasing(calibrate_curve(predicted_curve, policy, features))
    best_index = 0
    best_objective = math.inf
    for index in range(BUDGET_CURVE_POINTS):
        objective = math.exp(projected[index]) + policy.lagrange_multiplier * BUDGET_CURVE_MULTIPLES[index]
        if objective < best_objective:
            best_objective = objective
            best_index = index
    return best_index

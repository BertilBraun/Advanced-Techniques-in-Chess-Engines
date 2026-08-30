from __future__ import annotations

import math

from pydantic import Field, model_validator
from src.util.frozen_model import FrozenModel

BUDGET_CURVE_MULTIPLES: tuple[float, ...] = (0.125, 0.2, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5, 2.0, 3.0, 4.0)
BUDGET_CURVE_POINTS = 10
BASELINE_CURVE_INDEX = BUDGET_CURVE_MULTIPLES.index(1.0)
HALF_DEEP_CURVE_INDEX = BUDGET_CURVE_MULTIPLES.index(4.0)
LOG_KL_EPSILON = 1e-6


class SearchBudgetPolicy(FrozenModel):
    sigma: tuple[float, ...] = Field(min_length=BUDGET_CURVE_POINTS, max_length=BUDGET_CURVE_POINTS)
    log_tau: float
    selection_threshold: float
    apply_learned: bool

    @model_validator(mode='after')
    def validate_policy(self) -> SearchBudgetPolicy:
        if any(not math.isfinite(value) or value <= 0.0 for value in self.sigma):
            raise ValueError('Search-budget sigma values must be finite and positive.')
        if not math.isfinite(self.log_tau):
            raise ValueError('Search-budget log tau must be finite.')
        if not math.isfinite(self.selection_threshold) or not 0.0 < self.selection_threshold < 1.0:
            raise ValueError('Search-budget selection threshold must lie strictly in (0, 1).')
        return self


def disabled_policy() -> SearchBudgetPolicy:
    return SearchBudgetPolicy(
        sigma=(1.0,) * BUDGET_CURVE_POINTS,
        log_tau=0.0,
        selection_threshold=0.8,
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


def isotonic_from_top(values: tuple[float, ...]) -> tuple[float, ...]:
    """Running minimum from the largest budget downward: more search never predicts more error."""
    if len(values) != BUDGET_CURVE_POINTS:
        raise ValueError('Isotonic projection requires one value per grid point.')
    projected = list(values)
    for index in range(BUDGET_CURVE_POINTS - 2, -1, -1):
        projected[index] = min(projected[index], projected[index + 1])
    return tuple(projected)


def standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def select_budget_index(predicted_curve: tuple[float, ...], policy: SearchBudgetPolicy) -> int:
    if len(predicted_curve) != BUDGET_CURVE_POINTS:
        raise ValueError('Budget selection requires one prediction per grid point.')
    if any(not math.isfinite(value) for value in predicted_curve):
        raise ValueError('Budget selection requires finite predictions.')
    projected = isotonic_from_top(predicted_curve)
    for index in range(BUDGET_CURVE_POINTS):
        probability = standard_normal_cdf((policy.log_tau - projected[index]) / policy.sigma[index])
        if probability > policy.selection_threshold:
            return index
    return BUDGET_CURVE_POINTS - 1

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import model_validator
from src.util.frozen_model import FrozenModel

BUDGET_CURVE_MULTIPLES: tuple[float, ...] = (0.125, 0.2, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5, 2.0)
BUDGET_CURVE_POINTS = 8
BASELINE_CURVE_INDEX = BUDGET_CURVE_MULTIPLES.index(1.0)
LOG_KL_EPSILON = 1e-6

# The label-quality diagnostic compares the deep policy against its own half-way checkpoint, which
# sits above the narrowed grid and is checkpointed separately.
HALF_DEEP_VISIT_MULTIPLE = 4

# Corrector inputs: the full predicted curve, then top visit share, policy entropy, ply, baseline
# visits, source generation. Native builds the identical vector; the order is a binding contract.
CORRECTOR_SHARED_FEATURE_COUNT = 5
CORRECTOR_INPUT_FEATURES = BUDGET_CURVE_POINTS + CORRECTOR_SHARED_FEATURE_COUNT


@dataclass(frozen=True)
class BudgetSelectionFeatures:
    top_visit_share: float
    policy_entropy: float
    ply: int
    baseline_visits: int
    source_generation: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.top_visit_share) or not math.isfinite(self.policy_entropy):
            raise ValueError('Budget selection features must be finite.')
        if self.ply < 0 or self.baseline_visits <= 0 or self.source_generation < 0:
            raise ValueError(
                'Budget selection features require a nonnegative ply and source generation and positive '
                'baseline visits.'
            )


CurveCorrection = Callable[[tuple[float, ...], BudgetSelectionFeatures], tuple[float, ...]]


def identity_correction(predicted_curve: tuple[float, ...], features: BudgetSelectionFeatures) -> tuple[float, ...]:
    return predicted_curve


class SearchBudgetPolicy(FrozenModel):
    lagrange_multiplier: float
    corrector_path: Path | None
    corrector_sha256: str | None
    apply_learned: bool

    @model_validator(mode='after')
    def validate_policy(self) -> SearchBudgetPolicy:
        if not math.isfinite(self.lagrange_multiplier) or self.lagrange_multiplier < 0.0:
            raise ValueError('The search-budget Lagrange multiplier must be finite and nonnegative.')
        if (self.corrector_path is None) != (self.corrector_sha256 is None):
            raise ValueError('A search-budget corrector reference requires both its path and its digest.')
        if self.corrector_sha256 is not None and not _is_sha256(self.corrector_sha256):
            raise ValueError('A search-budget corrector digest must be 64 lowercase hex characters.')
        return self


def _is_sha256(digest: str) -> bool:
    return len(digest) == 64 and all(character in '0123456789abcdef' for character in digest)


def disabled_policy() -> SearchBudgetPolicy:
    return SearchBudgetPolicy(
        lagrange_multiplier=0.0,
        corrector_path=None,
        corrector_sha256=None,
        apply_learned=False,
    )


def deep_label_visit_limit(baseline_new_visits: int) -> int:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    return 8 * baseline_new_visits


def half_deep_visit_count(baseline_new_visits: int) -> int:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    return HALF_DEEP_VISIT_MULTIPLE * baseline_new_visits


def grid_visit_counts(baseline_new_visits: int) -> tuple[int, ...]:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    return tuple(max(1, int(math.floor(multiple * baseline_new_visits + 0.5))) for multiple in BUDGET_CURVE_MULTIPLES)


def grid_checkpoint_visits(baseline_new_visits: int) -> tuple[int, ...]:
    return tuple(sorted({*grid_visit_counts(baseline_new_visits), half_deep_visit_count(baseline_new_visits)}))


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


def corrected_curve(
    predicted_curve: tuple[float, ...],
    features: BudgetSelectionFeatures,
    correction: CurveCorrection = identity_correction,
) -> tuple[float, ...]:
    if len(predicted_curve) != BUDGET_CURVE_POINTS:
        raise ValueError('Curve correction requires one prediction per grid point.')
    if any(not math.isfinite(value) for value in predicted_curve):
        raise ValueError('Curve correction requires finite predictions.')
    corrected = correction(predicted_curve, features)
    if len(corrected) != BUDGET_CURVE_POINTS or any(not math.isfinite(value) for value in corrected):
        raise ValueError('A curve correction must produce one finite value per grid point.')
    return corrected


def select_budget_index(
    predicted_curve: tuple[float, ...],
    policy: SearchBudgetPolicy,
    features: BudgetSelectionFeatures,
    correction: CurveCorrection = identity_correction,
) -> int:
    """Lagrangian selection: the grid point minimising predicted raw KL plus dual-priced spend.

    The objective works in raw KL space because the run-level quantity being minimised is a sum of
    KLs, not of logs. Ties go to the cheapest grid point.
    """
    projected = project_non_increasing(corrected_curve(predicted_curve, features, correction))
    best_index = 0
    best_objective = math.inf
    for index in range(BUDGET_CURVE_POINTS):
        objective = math.exp(projected[index]) + policy.lagrange_multiplier * BUDGET_CURVE_MULTIPLES[index]
        if objective < best_objective:
            best_objective = objective
            best_index = index
    return best_index

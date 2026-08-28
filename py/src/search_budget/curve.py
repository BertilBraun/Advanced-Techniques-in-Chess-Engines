from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

from pydantic import Field, model_validator
from src.util.frozen_model import FrozenModel

CURVE_BUCKET_COUNT = 10
CURVE_BUCKET_BOUNDARIES: tuple[float, ...] = tuple((index + 1) / CURVE_BUCKET_COUNT for index in range(10))
INITIALIZER_VERSION = 'analytic_q5_v1'


class SearchBudgetCurve(FrozenModel):
    multipliers: tuple[float, ...] = Field(min_length=CURVE_BUCKET_COUNT, max_length=CURVE_BUCKET_COUNT)

    @model_validator(mode='after')
    def validate_curve(self) -> SearchBudgetCurve:
        if any(not math.isfinite(value) or value <= 0.0 for value in self.multipliers):
            raise ValueError('Search-budget curve multipliers must be finite and positive.')
        if any(left > right for left, right in zip(self.multipliers, self.multipliers[1:])):
            raise ValueError('Search-budget curve multipliers must be monotone nondecreasing.')
        if not math.isclose(fmean(self.multipliers), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('Search-budget curve multipliers must have arithmetic mean one.')
        return self

    @property
    def minimum(self) -> float:
        return self.multipliers[0]

    @property
    def maximum(self) -> float:
        return self.multipliers[-1]


@dataclass(frozen=True)
class CurveUpdate:
    curve: SearchBudgetCurve
    raw_log_updates: tuple[float, ...]
    projection_adjustments: tuple[float, ...]
    backtracking_steps: int


def flat_curve() -> SearchBudgetCurve:
    return SearchBudgetCurve(multipliers=(1.0,) * CURVE_BUCKET_COUNT)


def analytic_initial_curve() -> SearchBudgetCurve:
    width = 1.0 / CURVE_BUCKET_COUNT
    averages = tuple(
        0.2 + 8.0 * (((index + 1) * width) ** 6 - (index * width) ** 6) for index in range(CURVE_BUCKET_COUNT)
    )
    return _normalized_curve(averages)


def bucket_index(predicted_quantile: float) -> int:
    if not math.isfinite(predicted_quantile) or not 0.0 <= predicted_quantile <= 1.0:
        raise ValueError('Predicted search-budget quantile must be finite and in [0, 1].')
    return min(CURVE_BUCKET_COUNT - 1, int(predicted_quantile * CURVE_BUCKET_COUNT))


def multiplier_for_quantile(curve: SearchBudgetCurve, predicted_quantile: float) -> float:
    return curve.multipliers[bucket_index(predicted_quantile)]


def probe_curve(curve: SearchBudgetCurve, selected_bucket: int, ratio: float, upper: bool) -> tuple[float, ...]:
    if not 0 <= selected_bucket < CURVE_BUCKET_COUNT:
        raise ValueError('Probe bucket index is outside the curve.')
    if not math.isfinite(ratio) or ratio <= 1.0:
        raise ValueError('Probe ratio must be finite and greater than one.')
    values = list(curve.multipliers)
    values[selected_bucket] *= ratio if upper else 1.0 / ratio
    scale = CURVE_BUCKET_COUNT / sum(values)
    return tuple(value * scale for value in values)


def update_shadow_curve(
    curve: SearchBudgetCurve,
    ema_utilities: tuple[float | None, ...],
    sample_counts: tuple[int, ...],
    maximum_step_ratio: float,
) -> CurveUpdate:
    if len(ema_utilities) != CURVE_BUCKET_COUNT or len(sample_counts) != CURVE_BUCKET_COUNT:
        raise ValueError('Curve updates require one utility and count per bucket.')
    if any(count < 0 for count in sample_counts):
        raise ValueError('Curve-update sample counts must be nonnegative.')
    if not math.isfinite(maximum_step_ratio) or maximum_step_ratio <= 1.0:
        raise ValueError('Maximum curve step ratio must be finite and greater than one.')
    observed = tuple(
        (utility, count)
        for utility, count in zip(ema_utilities, sample_counts, strict=True)
        if utility is not None and count > 0
    )
    if not observed:
        zeros = (0.0,) * CURVE_BUCKET_COUNT
        return CurveUpdate(curve=curve, raw_log_updates=zeros, projection_adjustments=zeros, backtracking_steps=0)
    total_count = sum(count for _, count in observed)
    compute_price = sum(utility * count for utility, count in observed) / total_count
    centered = tuple(
        None if utility is None or count == 0 else utility - compute_price
        for utility, count in zip(ema_utilities, sample_counts, strict=True)
    )
    scale = max((abs(value) for value in centered if value is not None), default=0.0)
    if scale == 0.0:
        zeros = (0.0,) * CURVE_BUCKET_COUNT
        return CurveUpdate(curve=curve, raw_log_updates=zeros, projection_adjustments=zeros, backtracking_steps=0)
    maximum_log_step = math.log(maximum_step_ratio)
    initial_updates = tuple(0.0 if value is None else maximum_log_step * value / scale for value in centered)
    updates = initial_updates
    backtracking_steps = 0
    while True:
        raw_logs = tuple(math.log(value) + update for value, update in zip(curve.multipliers, updates, strict=True))
        projected_logs = _isotonic_non_decreasing(raw_logs)
        candidate = _normalized_curve(tuple(math.exp(value) for value in projected_logs))
        ratios = tuple(
            selected / previous for selected, previous in zip(candidate.multipliers, curve.multipliers, strict=True)
        )
        tolerance = 1e-12
        if all(1.0 / maximum_step_ratio - tolerance <= ratio <= maximum_step_ratio + tolerance for ratio in ratios):
            adjustments = tuple(
                math.log(selected / previous) - raw_update
                for selected, previous, raw_update in zip(
                    candidate.multipliers,
                    curve.multipliers,
                    initial_updates,
                    strict=True,
                )
            )
            return CurveUpdate(
                curve=candidate,
                raw_log_updates=initial_updates,
                projection_adjustments=adjustments,
                backtracking_steps=backtracking_steps,
            )
        updates = tuple(value / 2.0 for value in updates)
        backtracking_steps += 1
        if backtracking_steps > 64:
            raise AssertionError('Curve-update backtracking did not converge.')


def bounded_curve_toward(
    current: SearchBudgetCurve,
    target: SearchBudgetCurve,
    maximum_step_ratio: float,
) -> SearchBudgetCurve:
    if not math.isfinite(maximum_step_ratio) or maximum_step_ratio <= 1.0:
        raise ValueError('Maximum curve step ratio must be finite and greater than one.')
    target_logs = tuple(math.log(value) for value in target.multipliers)
    interpolation = 1.0
    while interpolation > 2.0**-64:
        logs = tuple(
            math.log(previous) + interpolation * (selected - math.log(previous))
            for previous, selected in zip(current.multipliers, target_logs, strict=True)
        )
        candidate = _normalized_curve(tuple(math.exp(value) for value in _isotonic_non_decreasing(logs)))
        ratios = tuple(
            selected / previous for selected, previous in zip(candidate.multipliers, current.multipliers, strict=True)
        )
        if all(1.0 / maximum_step_ratio - 1e-12 <= ratio <= maximum_step_ratio + 1e-12 for ratio in ratios):
            return candidate
        interpolation /= 2.0
    return current


def _isotonic_non_decreasing(values: tuple[float, ...]) -> tuple[float, ...]:
    blocks: list[tuple[float, int]] = []
    for value in values:
        blocks.append((value, 1))
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right_mean, right_weight = blocks.pop()
            left_mean, left_weight = blocks.pop()
            weight = left_weight + right_weight
            blocks.append(((left_mean * left_weight + right_mean * right_weight) / weight, weight))
    return tuple(mean for mean, weight in blocks for _ in range(weight))


def _normalized_curve(values: tuple[float, ...]) -> SearchBudgetCurve:
    if len(values) != CURVE_BUCKET_COUNT:
        raise ValueError('Search-budget curves require exactly ten multipliers.')
    scale = CURVE_BUCKET_COUNT / sum(values)
    return SearchBudgetCurve(multipliers=tuple(value * scale for value in values))

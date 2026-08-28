from __future__ import annotations

import math
from inspect import getsource
from statistics import fmean

import pytest
import src.search_budget.curve as curve_module
from src.search_budget.curve import (
    CURVE_BUCKET_COUNT,
    SearchBudgetCurve,
    analytic_initial_curve,
    bounded_curve_toward,
    bucket_index,
    flat_curve,
    multiplier_for_quantile,
    update_shadow_curve,
)


def test_analytic_initializer_uses_exact_bucket_average_formula() -> None:
    curve = analytic_initial_curve()
    expected = tuple(0.2 + 8.0 * (((index + 1) / 10) ** 6 - (index / 10) ** 6) for index in range(CURVE_BUCKET_COUNT))
    assert curve.multipliers == pytest.approx(expected, abs=1e-12)
    assert fmean(curve.multipliers) == pytest.approx(1.0, abs=1e-12)
    assert curve.minimum > 0.0
    assert curve.maximum > curve.minimum


@pytest.mark.parametrize(('quantile', 'expected_bucket'), [(0.0, 0), (0.099, 0), (0.1, 1), (0.9, 9), (1.0, 9)])
def test_equal_width_curve_boundaries_are_right_open(quantile: float, expected_bucket: int) -> None:
    curve = analytic_initial_curve()
    assert bucket_index(quantile) == expected_bucket
    assert multiplier_for_quantile(curve, quantile) == curve.multipliers[expected_bucket]


@pytest.mark.parametrize('quantile', [-0.01, 1.01, float('nan'), float('inf')])
def test_curve_rejects_invalid_quantiles(quantile: float) -> None:
    with pytest.raises(ValueError, match='quantile'):
        bucket_index(quantile)


def test_curve_update_is_monotone_mean_one_and_bounded_to_ten_percent() -> None:
    curve = analytic_initial_curve()
    update = update_shadow_curve(
        curve,
        tuple(float(index) for index in range(CURVE_BUCKET_COUNT)),
        (10,) * CURVE_BUCKET_COUNT,
        1.1,
    )
    assert fmean(update.curve.multipliers) == pytest.approx(1.0, abs=1e-12)
    assert tuple(sorted(update.curve.multipliers)) == update.curve.multipliers
    assert all(
        1 / 1.1 - 1e-12 <= selected / previous <= 1.1 + 1e-12
        for selected, previous in zip(update.curve.multipliers, curve.multipliers, strict=True)
    )
    assert max(abs(value) for value in update.raw_log_updates) == pytest.approx(math.log(1.1))


def test_empty_bucket_has_no_update_signal_and_projection_is_deterministic() -> None:
    curve = flat_curve()
    utilities = (1.0, None, 3.0, None, 5.0, None, 7.0, None, 9.0, None)
    first = update_shadow_curve(curve, utilities, (1, 0) * 5, 1.1)
    second = update_shadow_curve(curve, utilities, (1, 0) * 5, 1.1)
    assert first == second
    assert all(first.raw_log_updates[index] == 0.0 for index in range(1, CURVE_BUCKET_COUNT, 2))


def test_bounded_curve_toward_preserves_curve_invariants_and_trust_bound() -> None:
    current = flat_curve()
    bounded = bounded_curve_toward(current, analytic_initial_curve(), 1.1)
    assert isinstance(bounded, SearchBudgetCurve)
    assert all(1 / 1.1 - 1e-12 <= value <= 1.1 + 1e-12 for value in bounded.multipliers)


def test_executable_curve_has_no_historical_or_blend_constants() -> None:
    source = getsource(curve_module)
    assert '3761' not in source
    assert '1186' not in source
    assert 'BLEND_CANDIDATES' not in source
    assert 'CURVE_QUANTILE_BOUNDARIES' not in source

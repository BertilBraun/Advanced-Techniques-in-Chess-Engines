from __future__ import annotations

from fractions import Fraction

import pytest
from src.search_budget.curve import (
    BELOW_BASELINE_QUANTILE_SHARE,
    CURVE_CEILING,
    CURVE_FLOOR,
    CURVE_MEAN,
    CURVE_MULTIPLIERS,
    CURVE_QUANTILE_BOUNDARIES,
    multiplier_for_quantile,
)


def test_exact_curve_mean_floor_ceiling_and_below_baseline_share() -> None:
    assert CURVE_MEAN == Fraction(1)
    assert CURVE_FLOOR == Fraction(750, 3761)
    assert CURVE_CEILING == Fraction(18000, 3761)
    assert BELOW_BASELINE_QUANTILE_SHARE == Fraction(2188, 3000)


@pytest.mark.parametrize('index', range(9))
def test_curve_intervals_are_right_open(index: int) -> None:
    boundary = float(CURVE_QUANTILE_BOUNDARIES[index])
    assert multiplier_for_quantile(boundary) == CURVE_MULTIPLIERS[index + 1]


def test_curve_includes_both_endpoint_quantiles() -> None:
    assert multiplier_for_quantile(0.0) == CURVE_FLOOR
    assert multiplier_for_quantile(1.0) == CURVE_CEILING


@pytest.mark.parametrize('quantile', [-0.01, 1.01, float('nan'), float('inf')])
def test_curve_rejects_invalid_quantiles(quantile: float) -> None:
    with pytest.raises(ValueError, match='quantile'):
        multiplier_for_quantile(quantile)

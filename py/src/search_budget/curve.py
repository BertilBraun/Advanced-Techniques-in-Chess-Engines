from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

CURVE_QUANTILE_BOUNDARIES: tuple[Fraction, ...] = (
    Fraction(1186, 3000),
    Fraction(1570, 3000),
    Fraction(1838, 3000),
    Fraction(2048, 3000),
    Fraction(2188, 3000),
    Fraction(2347, 3000),
    Fraction(2547, 3000),
    Fraction(2699, 3000),
    Fraction(2806, 3000),
    Fraction(1, 1),
)
CURVE_MULTIPLIERS: tuple[Fraction, ...] = (
    Fraction(750, 3761),
    Fraction(1500, 3761),
    Fraction(2250, 3761),
    Fraction(3000, 3761),
    Fraction(3750, 3761),
    Fraction(4500, 3761),
    Fraction(6000, 3761),
    Fraction(9000, 3761),
    Fraction(12000, 3761),
    Fraction(18000, 3761),
)
CURVE_FLOOR = CURVE_MULTIPLIERS[0]
CURVE_CEILING = CURVE_MULTIPLIERS[-1]
CURVE_MEAN = sum(
    (boundary - (Fraction(0) if index == 0 else CURVE_QUANTILE_BOUNDARIES[index - 1])) * multiplier
    for index, (boundary, multiplier) in enumerate(zip(CURVE_QUANTILE_BOUNDARIES, CURVE_MULTIPLIERS, strict=True))
)
BELOW_BASELINE_QUANTILE_SHARE = CURVE_QUANTILE_BOUNDARIES[4]
BLEND_CANDIDATES: tuple[Decimal, ...] = tuple(Decimal(index) / Decimal(10) for index in range(11))

assert CURVE_MEAN == 1


def multiplier_for_quantile(predicted_quantile: float) -> Fraction:
    if not math.isfinite(predicted_quantile) or not 0.0 <= predicted_quantile <= 1.0:
        raise ValueError('Predicted search-budget quantile must be finite and in [0, 1].')
    for boundary, multiplier in zip(CURVE_QUANTILE_BOUNDARIES, CURVE_MULTIPLIERS, strict=True):
        if predicted_quantile < float(boundary) or boundary == 1:
            return multiplier
    raise AssertionError('The final multiplier-curve interval includes one.')


def blended_multiplier(predicted_quantile: float, blend: Decimal) -> Fraction:
    if not blend.is_finite() or not Decimal(0) <= blend <= Decimal(1):
        raise ValueError('Search-budget blend must be finite and in [0, 1].')
    exact_blend = Fraction(blend)
    return (1 - exact_blend) + exact_blend * multiplier_for_quantile(predicted_quantile)

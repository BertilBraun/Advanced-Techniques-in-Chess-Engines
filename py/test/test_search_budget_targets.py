from __future__ import annotations

import math

import pytest
from src.search_budget.targets import PolicyDistribution, midrank_quantiles, policy_kl, shadow_gain


def policy(*probabilities: float) -> PolicyDistribution:
    return PolicyDistribution(probabilities=probabilities)


def test_midrank_quantiles_cover_ties_and_preserve_input_order() -> None:
    assert midrank_quantiles((4.0, 1.0, 2.0, 2.0)) == pytest.approx((1.0, 0.0, 0.5, 0.5))


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        ((3.0,), (0.5,)),
        ((3.0, 3.0), (0.5, 0.5)),
        ((3.0, 3.0, 3.0), (0.5, 0.5, 0.5)),
    ],
)
def test_midrank_quantiles_assign_half_to_degenerate_samples(
    values: tuple[float, ...],
    expected: tuple[float, ...],
) -> None:
    assert midrank_quantiles(values) == expected


def test_policy_kl_and_shadow_gain_use_deep_to_approximate_direction() -> None:
    deep = (policy(0.8, 0.2), policy(0.4, 0.6))
    flat = (policy(0.5, 0.5), policy(0.5, 0.5))
    candidate = (policy(0.7, 0.3), policy(0.4, 0.6))
    expected = (
        sum(
            policy_kl(reference, baseline) - policy_kl(reference, improved)
            for reference, baseline, improved in zip(deep, flat, candidate, strict=True)
        )
        / 2
    )
    assert shadow_gain(deep, flat, candidate) == pytest.approx(expected)
    assert shadow_gain(deep, flat, candidate) > 0.0


def test_policy_kl_returns_infinity_when_candidate_omits_deep_mass() -> None:
    assert math.isinf(policy_kl(policy(1.0, 0.0), policy(0.0, 1.0)))

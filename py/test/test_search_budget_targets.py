from __future__ import annotations

import math

import pytest
from src.search_budget.targets import PolicyDistribution, policy_entropy, policy_kl, shadow_gain, top_visit_share


def policy(*probabilities: float) -> PolicyDistribution:
    return PolicyDistribution(probabilities=probabilities)


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


def test_policy_kl_applies_documented_floor_when_candidate_omits_deep_mass() -> None:
    assert policy_kl(policy(1.0, 0.0), policy(0.0, 1.0)) == pytest.approx(math.log(1_000_000))


def test_policy_entropy_is_natural_log_entropy_over_positive_mass() -> None:
    assert policy_entropy(policy(0.5, 0.5, 0.0)) == pytest.approx(math.log(2.0))


def test_top_visit_share_is_the_largest_probability() -> None:
    assert top_visit_share(policy(0.2, 0.7, 0.1)) == pytest.approx(0.7)

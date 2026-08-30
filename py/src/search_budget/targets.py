from __future__ import annotations

import math

from pydantic import Field, model_validator
from src.util.frozen_model import FrozenModel

POLICY_PROBABILITY_FLOOR = 1e-6


class PolicyDistribution(FrozenModel):
    probabilities: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_probabilities(self) -> PolicyDistribution:
        if any(not math.isfinite(probability) or probability < 0.0 for probability in self.probabilities):
            raise ValueError('Policy probabilities must be finite and nonnegative.')
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError('Policy probabilities must sum to one.')
        return self


def policy_kl(deep_policy: PolicyDistribution, approximate_policy: PolicyDistribution) -> float:
    if len(deep_policy.probabilities) != len(approximate_policy.probabilities):
        raise ValueError('Deep and approximate policies must use the same action space.')
    divergence = 0.0
    for deep_probability, approximate_probability in zip(
        deep_policy.probabilities,
        approximate_policy.probabilities,
        strict=True,
    ):
        if deep_probability == 0.0:
            continue
        divergence += deep_probability * math.log(
            max(POLICY_PROBABILITY_FLOOR, deep_probability) / max(POLICY_PROBABILITY_FLOOR, approximate_probability)
        )
    return max(0.0, divergence)


def shadow_gain(
    deep_policies: tuple[PolicyDistribution, ...],
    flat_policies: tuple[PolicyDistribution, ...],
    candidate_policies: tuple[PolicyDistribution, ...],
) -> float:
    if not deep_policies:
        raise ValueError('Shadow scoring requires at least one labelled position.')
    if len(deep_policies) != len(flat_policies) or len(deep_policies) != len(candidate_policies):
        raise ValueError('Shadow policy collections must have equal position counts.')
    paired_gains = tuple(
        policy_kl(deep, flat) - policy_kl(deep, candidate)
        for deep, flat, candidate in zip(deep_policies, flat_policies, candidate_policies, strict=True)
    )
    if any(not math.isfinite(gain) for gain in paired_gains):
        raise ValueError('Shadow gain requires finite policy divergences.')
    return math.fsum(paired_gains) / len(paired_gains)


def policy_entropy(policy: PolicyDistribution) -> float:
    return -sum(probability * math.log(probability) for probability in policy.probabilities if probability > 0.0)


def top_visit_share(policy: PolicyDistribution) -> float:
    return max(policy.probabilities)

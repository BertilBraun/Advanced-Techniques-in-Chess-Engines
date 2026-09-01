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


def policy_kl(reference_policy: PolicyDistribution, approximate_policy: PolicyDistribution) -> float:
    if len(reference_policy.probabilities) != len(approximate_policy.probabilities):
        raise ValueError('Reference and approximate policies must use the same action space.')
    divergence = 0.0
    for reference_probability, approximate_probability in zip(
        reference_policy.probabilities,
        approximate_policy.probabilities,
        strict=True,
    ):
        if reference_probability == 0.0:
            continue
        divergence += reference_probability * math.log(
            max(POLICY_PROBABILITY_FLOOR, reference_probability)
            / max(POLICY_PROBABILITY_FLOOR, approximate_probability)
        )
    return max(0.0, divergence)


def policy_entropy(policy: PolicyDistribution) -> float:
    return -sum(probability * math.log(probability) for probability in policy.probabilities if probability > 0.0)


def top_visit_share(policy: PolicyDistribution) -> float:
    return max(policy.probabilities)

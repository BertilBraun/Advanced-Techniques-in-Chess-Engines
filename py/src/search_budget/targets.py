from __future__ import annotations

import math

from pydantic import Field, model_validator
from src.util.frozen_model import FrozenModel


class PolicyDistribution(FrozenModel):
    probabilities: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_probabilities(self) -> PolicyDistribution:
        if any(not math.isfinite(probability) or probability < 0.0 for probability in self.probabilities):
            raise ValueError('Policy probabilities must be finite and nonnegative.')
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError('Policy probabilities must sum to one.')
        return self


def midrank_quantiles(raw_kl_values: tuple[float, ...]) -> tuple[float, ...]:
    if not raw_kl_values:
        raise ValueError('At least one raw KL value is required.')
    if any(not math.isfinite(value) or value < 0.0 for value in raw_kl_values):
        raise ValueError('Raw KL values must be finite and nonnegative.')
    if len(raw_kl_values) == 1:
        return (0.5,)

    indexed_values = sorted(enumerate(raw_kl_values), key=lambda item: (item[1], item[0]))
    quantiles = [0.0] * len(raw_kl_values)
    group_start = 0
    while group_start < len(indexed_values):
        group_end = group_start + 1
        while group_end < len(indexed_values) and indexed_values[group_end][1] == indexed_values[group_start][1]:
            group_end += 1
        midrank = (group_start + group_end - 1) / 2
        quantile = midrank / (len(indexed_values) - 1)
        for position in range(group_start, group_end):
            quantiles[indexed_values[position][0]] = quantile
        group_start = group_end
    return tuple(quantiles)


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
        if approximate_probability == 0.0:
            return math.inf
        divergence += deep_probability * math.log(deep_probability / approximate_probability)
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

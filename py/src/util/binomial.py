from __future__ import annotations

from math import exp, isfinite, lgamma, log, log1p


def one_sided_binomial_upper_bound(failure_count: int, trial_count: int, confidence_level: float) -> float:
    if trial_count <= 0:
        return 1.0
    if not 0 <= failure_count <= trial_count:
        raise ValueError('Failure count must lie between zero and the trial count.')
    if not 0.0 < confidence_level < 1.0:
        raise ValueError('Confidence level must lie in (0, 1).')
    if failure_count == trial_count:
        return 1.0
    target_probability = 1.0 - confidence_level
    lower = failure_count / trial_count
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if binomial_cumulative_probability(failure_count, trial_count, midpoint) > target_probability:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def binomial_cumulative_probability(maximum_failures: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    logarithms = tuple(
        lgamma(trials + 1)
        - lgamma(failures + 1)
        - lgamma(trials - failures + 1)
        + failures * log(probability)
        + (trials - failures) * log1p(-probability)
        for failures in range(maximum_failures + 1)
    )
    maximum = max(logarithms)
    result = exp(maximum) * sum(exp(item - maximum) for item in logarithms)
    if not isfinite(result):
        raise ValueError('Binomial confidence calculation produced a non-finite result.')
    return result

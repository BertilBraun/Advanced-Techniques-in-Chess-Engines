from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from src.search_stopping.configuration import SearchStoppingConfiguration
from src.util.binomial import one_sided_binomial_upper_bound

_ATTENUATED_THRESHOLD = 0.0
_EPS_SOLVE_ITERATIONS = 40
_UNIT_SPEND_TARGET = 1.0


@dataclass(frozen=True)
class AuditWindowArrays:
    """Raw audit evidence for the solve: one row per audited position, one column per checkpoint.

    `kl_to_final` and `value_gap` are the raw measurements (labels are derived under a candidate
    eps at solve time — the reason eps can move at all), `guard_movement` the observed
    checkpoint-to-checkpoint KL, and `stop_probability` the current predictor's u.
    """

    kl_to_final: npt.NDArray[np.float64]
    value_gap: npt.NDArray[np.float64]
    guard_movement: npt.NDArray[np.float64]
    stop_probability: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        shape = self.kl_to_final.shape
        if len(shape) != 2 or shape[0] == 0 or shape[1] == 0:
            raise ValueError('Audit window arrays require one row per position and one column per checkpoint.')
        for array in (self.value_gap, self.guard_movement, self.stop_probability):
            if array.shape != shape:
                raise ValueError('Audit window arrays must share one shape.')
        for array in (self.kl_to_final, self.value_gap, self.guard_movement, self.stop_probability):
            if not np.isfinite(array).all():
                raise ValueError('Audit window arrays must be finite.')

    @property
    def checkpoint_count(self) -> int:
        return self.kl_to_final.shape[1]


def uncertain_labels(arrays: AuditWindowArrays, eps_pi: float, eps_v: float) -> npt.NDArray[np.bool_]:
    """Section 3.1 with the future-max clause, evaluated on the raw window measurements."""
    if eps_pi <= 0.0 or eps_v <= 0.0:
        raise ValueError('Label epsilons must be positive.')
    future_kl = np.maximum.accumulate(arrays.kl_to_final[:, ::-1], axis=1)[:, ::-1]
    future_gap = np.maximum.accumulate(arrays.value_gap[:, ::-1], axis=1)[:, ::-1]
    return (future_kl >= eps_pi) | (future_gap >= eps_v)


@dataclass(frozen=True)
class CheckpointThresholdSolution:
    threshold: float
    trigger_count: int
    false_stop_count: int
    false_stop_upper_bound: float
    attenuated: bool


@dataclass(frozen=True)
class ThresholdSolution:
    checkpoints: tuple[CheckpointThresholdSolution, ...]
    simulated_mean_spend: float
    simulated_stop_fraction: tuple[float, ...]
    any_checkpoint_open: bool


def solve_thresholds(
    arrays: AuditWindowArrays,
    labels: npt.NDArray[np.bool_],
    configuration: SearchStoppingConfiguration,
) -> ThresholdSolution:
    """Stateless per-generation threshold solve, cheapest checkpoint first on simulated survivors.

    Per checkpoint: the largest threshold whose guarded triggers number at least the evidence
    minimum and whose one-sided binomial upper bound on the false-stop rate stays under beta.
    No qualifying threshold attenuates the checkpoint (threshold 0, stops nothing).
    """
    if labels.shape != arrays.kl_to_final.shape:
        raise ValueError('Threshold solving requires one label per position and checkpoint.')
    checkpoint_count = arrays.checkpoint_count
    if checkpoint_count != len(configuration.checkpoint_multiples):
        raise ValueError('Audit window checkpoints must match the configured checkpoint multiples.')
    position_count = arrays.kl_to_final.shape[0]
    surviving = np.ones(position_count, dtype=bool)
    spend = np.full(position_count, configuration.cap_multiple)
    solutions: list[CheckpointThresholdSolution] = []
    stop_fractions: list[float] = []
    for index in range(checkpoint_count):
        guard_pass = surviving & (arrays.guard_movement[:, index] < configuration.movement_guard_epsilon)
        probabilities = arrays.stop_probability[guard_pass, index]
        uncertain = labels[guard_pass, index]
        threshold, triggers, false_stops, upper_bound = _largest_safe_threshold(
            probabilities,
            uncertain,
            configuration.false_stop_rate_ceiling,
            configuration.minimum_evidence_trigger_count,
            configuration.confidence_level,
        )
        attenuated = threshold == _ATTENUATED_THRESHOLD
        solutions.append(
            CheckpointThresholdSolution(
                threshold=threshold,
                trigger_count=triggers,
                false_stop_count=false_stops,
                false_stop_upper_bound=upper_bound,
                attenuated=attenuated,
            )
        )
        stopped = guard_pass & (arrays.stop_probability[:, index] < threshold)
        spend[stopped] = configuration.checkpoint_multiples[index]
        stop_fractions.append(float(stopped.sum()) / position_count)
        surviving &= ~stopped
    return ThresholdSolution(
        checkpoints=tuple(solutions),
        simulated_mean_spend=float(spend.mean()),
        simulated_stop_fraction=tuple(stop_fractions),
        any_checkpoint_open=any(not solution.attenuated for solution in solutions),
    )


def _largest_safe_threshold(
    probabilities: npt.NDArray[np.float64],
    uncertain: npt.NDArray[np.bool_],
    beta: float,
    minimum_evidence: int,
    confidence_level: float,
) -> tuple[float, int, int, float]:
    total = probabilities.shape[0]
    if total < minimum_evidence:
        return _ATTENUATED_THRESHOLD, 0, 0, 1.0
    order = np.argsort(probabilities, kind='stable')
    ordered_probabilities = probabilities[order]
    ordered_uncertain = uncertain[order]
    ordered_false = np.cumsum(ordered_uncertain.astype(np.int64))
    # For a fixed false-stop count the binomial upper bound falls as triggers grow, so only the
    # last count of each constant-false-count run can be the largest safe candidate.
    run_ends = np.nonzero(np.concatenate([ordered_uncertain[1:], [True]]))[0] + 1
    best: tuple[float, int, int, float] | None = None
    for count in (int(value) for value in reversed(run_ends)):
        if count < minimum_evidence:
            break
        false_stops = int(ordered_false[count - 1])
        if false_stops > beta * count:  # the exact bound only exceeds this point estimate
            continue
        upper_bound = one_sided_binomial_upper_bound(false_stops, count, confidence_level)
        if upper_bound > beta:
            continue
        if count < total:
            threshold = float((ordered_probabilities[count - 1] + ordered_probabilities[count]) / 2.0)
            if threshold <= ordered_probabilities[count - 1]:
                continue
        else:
            threshold = float(np.nextafter(ordered_probabilities[-1], np.inf))
        best = (threshold, count, false_stops, upper_bound)
        break
    if best is None:
        return _ATTENUATED_THRESHOLD, 0, 0, 1.0
    return best


@dataclass(frozen=True)
class EpsSolution:
    eps_pi: float
    saturated_at_maximum: bool
    thresholds: ThresholdSolution


def solve_spend_pinned_eps(
    arrays: AuditWindowArrays,
    configuration: SearchStoppingConfiguration,
) -> EpsSolution:
    """One-shot solve for the eps whose simulated stopping spends a mean multiple of 1.0.

    `spend(eps)` is monotone nonincreasing (a larger eps certifies more positions), so bisection is
    well-posed. The result is clamped to the configured quality clamp; saturation at the maximum
    means unit spend is unreachable without crossing the quality floor and is reported, never
    crossed.
    """

    def spend_at(eps_pi: float) -> ThresholdSolution:
        return solve_thresholds(arrays, uncertain_labels(arrays, eps_pi, configuration.eps_v), configuration)

    at_maximum = spend_at(configuration.eps_pi_maximum)
    if at_maximum.simulated_mean_spend > _UNIT_SPEND_TARGET:
        return EpsSolution(eps_pi=configuration.eps_pi_maximum, saturated_at_maximum=True, thresholds=at_maximum)
    at_minimum = spend_at(configuration.eps_pi_minimum)
    if at_minimum.simulated_mean_spend <= _UNIT_SPEND_TARGET:
        return EpsSolution(eps_pi=configuration.eps_pi_minimum, saturated_at_maximum=False, thresholds=at_minimum)
    low = configuration.eps_pi_minimum
    high = configuration.eps_pi_maximum
    for _ in range(_EPS_SOLVE_ITERATIONS):
        midpoint = (low + high) / 2.0
        if spend_at(midpoint).simulated_mean_spend > _UNIT_SPEND_TARGET:
            low = midpoint
        else:
            high = midpoint
    solution = spend_at(high)
    if not math.isfinite(solution.simulated_mean_spend):
        raise ValueError('The eps solve produced a non-finite simulated spend.')
    return EpsSolution(eps_pi=high, saturated_at_maximum=False, thresholds=solution)

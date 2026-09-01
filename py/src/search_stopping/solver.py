from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt
from src.search_stopping.configuration import SearchStoppingConfiguration
from src.util.binomial import one_sided_binomial_upper_bound

_ATTENUATED_THRESHOLD = 0.0


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
    """Section 3.1: instantaneous exceedance on the raw window measurements — the written target
    of a stop is the current distribution, so this is the harm event a false stop realizes."""
    if eps_pi <= 0.0 or eps_v <= 0.0:
        raise ValueError('Label epsilons must be positive.')
    return (arrays.kl_to_final >= eps_pi) | (arrays.value_gap >= eps_v)


@dataclass(frozen=True)
class CheckpointThresholdSolution:
    threshold: float
    trigger_count: int
    mean_excess_cost: float
    excess_cost_upper_bound: float
    false_stop_count: int
    attenuated: bool


@dataclass(frozen=True)
class ThresholdSolution:
    checkpoints: tuple[CheckpointThresholdSolution, ...]
    simulated_mean_spend: float
    simulated_stop_fraction: tuple[float, ...]
    any_checkpoint_open: bool


def solve_thresholds(
    arrays: AuditWindowArrays,
    configuration: SearchStoppingConfiguration,
    eps_pi: float,
) -> ThresholdSolution:
    """Stateless per-generation threshold solve, cheapest checkpoint first on simulated survivors.

    The admission criterion is a COST ceiling, not a false-stop rate: per admitted stop the cost
    is the excess of the written target over eps, max(0, kl_to_final - eps), so a barely-wrong
    stop at 1.5x baseline is no longer priced like a badly-wrong one at 1/3x. Two conditions per
    checkpoint, both UPPER bounds and both relative to eps so the criterion tracks the noise
    floor the labels are defined against:

    1. mean excess: a one-sided normal-approximation upper confidence bound (mean + z * SE at
       confidence_level, over at least minimum_evidence_trigger_count triggers) must stay within
       excess_cost_ceiling * eps. Defensible by the CLT at the enforced sample sizes; its known
       weakness - anti-conservatism under heavy tails - is exactly what the second condition
       covers.
    2. catastrophe rate: the exact one-sided binomial upper bound on
       P(excess > catastrophic_excess_multiple * eps) must stay within
       catastrophic_stop_ceiling. The v21 audit window measured individual admitted stops up to
       ~60x eps while the mean stayed bounded; a mean criterion alone would admit them.

    The largest threshold satisfying both wins; none attenuates the checkpoint (threshold 0).
    """
    if not math.isfinite(eps_pi) or eps_pi <= 0.0:
        raise ValueError('Threshold solving requires a positive finite eps.')
    checkpoint_count = arrays.checkpoint_count
    if checkpoint_count != len(configuration.checkpoint_multiples):
        raise ValueError('Audit window checkpoints must match the configured checkpoint multiples.')
    labels = uncertain_labels(arrays, eps_pi, configuration.eps_v)
    excess = np.maximum(0.0, arrays.kl_to_final - eps_pi)
    z_score = NormalDist().inv_cdf(configuration.confidence_level)
    cost_budget = configuration.excess_cost_ceiling * eps_pi
    position_count = arrays.kl_to_final.shape[0]
    surviving = np.ones(position_count, dtype=bool)
    spend = np.full(position_count, configuration.cap_multiple)
    solutions: list[CheckpointThresholdSolution] = []
    stop_fractions: list[float] = []
    for index in range(checkpoint_count):
        guard_pass = surviving & (arrays.guard_movement[:, index] < configuration.movement_guard_epsilon)
        threshold, triggers, mean_cost, upper_bound, false_stops = _largest_admissible_threshold(
            arrays.stop_probability[guard_pass, index],
            excess[guard_pass, index],
            labels[guard_pass, index],
            cost_budget,
            configuration.catastrophic_excess_multiple * eps_pi,
            configuration.catastrophic_stop_ceiling,
            configuration.minimum_evidence_trigger_count,
            z_score,
            configuration.confidence_level,
        )
        solutions.append(
            CheckpointThresholdSolution(
                threshold=threshold,
                trigger_count=triggers,
                mean_excess_cost=mean_cost,
                excess_cost_upper_bound=upper_bound,
                false_stop_count=false_stops,
                attenuated=threshold == _ATTENUATED_THRESHOLD,
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


def _largest_admissible_threshold(
    probabilities: npt.NDArray[np.float64],
    excess_costs: npt.NDArray[np.float64],
    uncertain: npt.NDArray[np.bool_],
    cost_budget: float,
    catastrophic_cost: float,
    catastrophic_ceiling: float,
    minimum_evidence: int,
    z_score: float,
    confidence_level: float,
) -> tuple[float, int, float, float, int]:
    total = probabilities.shape[0]
    if total < minimum_evidence:
        return _ATTENUATED_THRESHOLD, 0, 0.0, math.inf, 0
    order = np.argsort(probabilities, kind='stable')
    ordered_probabilities = probabilities[order]
    ordered_costs = excess_costs[order]
    counts = np.arange(1, total + 1, dtype=np.float64)
    prefix_mean = np.cumsum(ordered_costs) / counts
    prefix_variance = np.maximum(np.cumsum(ordered_costs**2) / counts - prefix_mean**2, 0.0)
    upper_bounds = prefix_mean + z_score * np.sqrt(prefix_variance / counts)
    catastrophic = np.cumsum(ordered_costs > catastrophic_cost)
    admissible = (counts >= minimum_evidence) & (upper_bounds <= cost_budget)
    # The exact binomial bound always exceeds the point estimate, so this prefilter is sound.
    admissible &= catastrophic <= catastrophic_ceiling * counts
    # A cut is only realisable where the threshold separates the admitted prefix from the rest.
    separable = np.ones(total, dtype=bool)
    separable[:-1] = ordered_probabilities[:-1] < ordered_probabilities[1:]
    admissible &= separable
    for last in reversed(np.nonzero(admissible)[0].tolist()):
        count = last + 1
        if one_sided_binomial_upper_bound(int(catastrophic[last]), count, confidence_level) > catastrophic_ceiling:
            continue
        if count < total:
            threshold = float((ordered_probabilities[last] + ordered_probabilities[last + 1]) / 2.0)
        else:
            threshold = float(np.nextafter(ordered_probabilities[-1], np.inf))
        false_stops = int(uncertain[order][:count].sum())
        return threshold, count, float(prefix_mean[last]), float(upper_bounds[last]), false_stops
    return _ATTENUATED_THRESHOLD, 0, 0.0, math.inf, 0


@dataclass(frozen=True)
class EpsSolution:
    eps_pi: float
    measured_noise_floor: float
    clamped: bool
    thresholds: ThresholdSolution


def solve_noise_floor_anchored_eps(
    arrays: AuditWindowArrays,
    paired_floor_divergences: npt.NDArray[np.float64],
    configuration: SearchStoppingConfiguration,
) -> EpsSolution:
    """eps is anchored to the measured cross-seed noise floor of the capped target, never to a
    spend target: eps = clamp(noise_floor_multiple * median paired-seed KL, [minimum, maximum]).
    Spend is an output of the threshold solve, reported and bounded above by the circuit breaker
    only — spend falling well below one is the win, not a fault."""
    if paired_floor_divergences.size == 0:
        raise ValueError('The eps anchor requires paired-audit noise-floor measurements.')
    if not np.isfinite(paired_floor_divergences).all() or bool((paired_floor_divergences < 0.0).any()):
        raise ValueError('Paired noise-floor divergences must be finite and nonnegative.')
    measured_floor = float(np.median(paired_floor_divergences))
    anchored = configuration.noise_floor_multiple * measured_floor
    eps_pi = min(max(anchored, configuration.eps_pi_minimum), configuration.eps_pi_maximum)
    if not math.isfinite(eps_pi) or eps_pi <= 0.0:
        raise ValueError('The anchored eps must be finite and positive.')
    thresholds = solve_thresholds(arrays, configuration, eps_pi)
    return EpsSolution(
        eps_pi=eps_pi,
        measured_noise_floor=measured_floor,
        clamped=eps_pi != anchored,
        thresholds=thresholds,
    )

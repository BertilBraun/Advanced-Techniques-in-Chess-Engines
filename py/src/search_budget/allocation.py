from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from src.search_budget.curve import CURVE_BUCKET_COUNT, SearchBudgetCurve, multiplier_for_quantile
from src.search_budget.sampling import LabelPositionIdentity


class CurveAllocationPurpose(str, Enum):
    FLAT = 'flat'
    PENDING_VALIDATION = 'pending_validation'
    PROBE_LOWER = 'probe_lower'
    PROBE_UPPER = 'probe_upper'


@dataclass(frozen=True)
class CurveAllocationIdentity:
    purpose: CurveAllocationPurpose
    bucket_index: int | None = None

    def __post_init__(self) -> None:
        is_probe = self.purpose in {CurveAllocationPurpose.PROBE_LOWER, CurveAllocationPurpose.PROBE_UPPER}
        if is_probe != (self.bucket_index is not None):
            raise ValueError('Only local-probe allocations carry a bucket index.')
        if self.bucket_index is not None and not 0 <= self.bucket_index < CURVE_BUCKET_COUNT:
            raise ValueError('Probe allocation bucket index is outside the curve.')


@dataclass(frozen=True)
class AllocationPosition:
    identity: LabelPositionIdentity
    predicted_quantile: float


@dataclass(frozen=True)
class AllocatedBudget:
    identity: LabelPositionIdentity
    assigned_new_visits: int


@dataclass(frozen=True)
class CandidateBudgetSet:
    identity: CurveAllocationIdentity
    allocation_multipliers: tuple[float, ...]
    budgets: tuple[AllocatedBudget, ...]
    total_assigned_new_visits: int
    flat_total_new_visits: int

    @property
    def spend_error(self) -> int:
        return self.total_assigned_new_visits - self.flat_total_new_visits


@dataclass(frozen=True)
class SequentialBudgetState:
    cumulative_baseline_visits: int = 0
    cumulative_assigned_visits: int = 0

    @property
    def spend_error(self) -> int:
        return self.cumulative_assigned_visits - self.cumulative_baseline_visits


def allocate_generation_multiplier_vector(
    positions: tuple[AllocationPosition, ...],
    baseline_new_visits: int,
    multipliers: tuple[float, ...],
    identity: CurveAllocationIdentity,
) -> CandidateBudgetSet:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    if not positions:
        raise ValueError('A candidate allocation requires at least one position.')
    if len(multipliers) != CURVE_BUCKET_COUNT:
        raise ValueError('A candidate allocation requires exactly ten bucket multipliers.')
    ordered_positions = tuple(sorted(positions, key=lambda position: _identity_key(position.identity)))
    identities = tuple(position.identity for position in ordered_positions)
    if len(set(identities)) != len(identities):
        raise ValueError('Candidate allocation position identities must be unique.')

    flat_total = baseline_new_visits * len(ordered_positions)
    weights = tuple(
        Fraction(str(multipliers[min(CURVE_BUCKET_COUNT - 1, int(position.predicted_quantile * CURVE_BUCKET_COUNT))]))
        for position in ordered_positions
    )
    exact_budgets = _bounded_normalized_budgets(
        weights,
        flat_total,
        Fraction(1),
        Fraction(deep_label_visit_limit(baseline_new_visits)),
    )
    cumulative_exact = Fraction(0)
    previous_rounded = 0
    budgets: list[AllocatedBudget] = []
    for index, (position, exact_budget) in enumerate(zip(ordered_positions, exact_budgets, strict=True)):
        cumulative_exact += exact_budget
        remaining_positions = len(ordered_positions) - index - 1
        minimum_cumulative = max(previous_rounded + 1, flat_total - remaining_positions * 8 * baseline_new_visits)
        maximum_cumulative = min(
            previous_rounded + 8 * baseline_new_visits,
            flat_total - remaining_positions,
        )
        cumulative_rounded = min(max(_round_fraction(cumulative_exact), minimum_cumulative), maximum_cumulative)
        budgets.append(
            AllocatedBudget(
                identity=position.identity,
                assigned_new_visits=cumulative_rounded - previous_rounded,
            )
        )
        previous_rounded = cumulative_rounded
    assert previous_rounded == flat_total
    return CandidateBudgetSet(
        identity=identity,
        allocation_multipliers=multipliers,
        budgets=tuple(budgets),
        total_assigned_new_visits=previous_rounded,
        flat_total_new_visits=flat_total,
    )


def allocate_next_production_budget(
    state: SequentialBudgetState,
    baseline_new_visits: int,
    predicted_quantile: float,
    curve: SearchBudgetCurve,
) -> tuple[int, SequentialBudgetState]:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    ideal_budget = Fraction(baseline_new_visits) * Fraction(str(multiplier_for_quantile(curve, predicted_quantile)))
    corrected_budget = ideal_budget - state.spend_error
    assigned_budget = min(max(_round_fraction(corrected_budget), 1), deep_label_visit_limit(baseline_new_visits))
    next_state = SequentialBudgetState(
        cumulative_baseline_visits=state.cumulative_baseline_visits + baseline_new_visits,
        cumulative_assigned_visits=state.cumulative_assigned_visits + assigned_budget,
    )
    return assigned_budget, next_state


def production_parallel_searches(assigned_new_visits: int) -> int:
    if assigned_new_visits <= 0:
        raise ValueError('Assigned new visits must be positive.')
    required_parallelism = (assigned_new_visits + 199) // 200
    next_power_of_two = 1 << (required_parallelism - 1).bit_length()
    return min(16, next_power_of_two)


def production_spend_error_bound(baseline_new_visits: int) -> int:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    return 7 * baseline_new_visits + 1


def deep_label_visit_limit(baseline_new_visits: int) -> int:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    return 8 * baseline_new_visits


def _bounded_normalized_budgets(
    weights: tuple[Fraction, ...],
    target_total: int,
    minimum_budget: Fraction,
    maximum_budget: Fraction,
) -> tuple[Fraction, ...]:
    unresolved = set(range(len(weights)))
    resolved: dict[int, Fraction] = {}
    while unresolved:
        remaining_target = Fraction(target_total) - sum(resolved.values())
        scale = remaining_target / sum(weights[index] for index in unresolved)
        below_floor = {index for index in unresolved if scale * weights[index] < minimum_budget}
        above_ceiling = {index for index in unresolved if scale * weights[index] > maximum_budget}
        if not below_floor and not above_ceiling:
            for index in unresolved:
                resolved[index] = scale * weights[index]
            break
        for index in below_floor:
            resolved[index] = minimum_budget
        for index in above_ceiling:
            resolved[index] = maximum_budget
        unresolved -= below_floor | above_ceiling
    exact_budgets = tuple(resolved[index] for index in range(len(weights)))
    assert sum(exact_budgets) == target_total
    return exact_budgets


def _round_fraction(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(2 * remainder >= value.denominator)


def _identity_key(identity: LabelPositionIdentity) -> tuple[int, str, int]:
    return identity.source_generation, identity.game_identity, identity.ply

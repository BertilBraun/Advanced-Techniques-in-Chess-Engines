from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from src.search_budget.curve import BLEND_CANDIDATES, CURVE_CEILING, CURVE_FLOOR, blended_multiplier
from src.search_budget.sampling import LabelPositionIdentity


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
    blend: Decimal
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


def allocate_generation_candidate(
    positions: tuple[AllocationPosition, ...],
    baseline_new_visits: int,
    blend: Decimal,
) -> CandidateBudgetSet:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    if not positions:
        raise ValueError('A candidate allocation requires at least one position.')
    ordered_positions = tuple(sorted(positions, key=lambda position: _identity_key(position.identity)))
    identities = tuple(position.identity for position in ordered_positions)
    if len(set(identities)) != len(identities):
        raise ValueError('Candidate allocation position identities must be unique.')

    flat_total = baseline_new_visits * len(ordered_positions)
    weights = tuple(blended_multiplier(position.predicted_quantile, blend) for position in ordered_positions)
    minimum_exact_budget = baseline_new_visits * _blended_bound(blend, CURVE_FLOOR)
    maximum_exact_budget = baseline_new_visits * _blended_bound(blend, CURVE_CEILING)
    exact_budgets = _bounded_normalized_budgets(weights, flat_total, minimum_exact_budget, maximum_exact_budget)
    minimum_integer_budget = _ceil_fraction(minimum_exact_budget)
    maximum_integer_budget = _floor_fraction(maximum_exact_budget)
    cumulative_exact = Fraction(0)
    previous_rounded = 0
    budgets: list[AllocatedBudget] = []
    for index, (position, exact_budget) in enumerate(zip(ordered_positions, exact_budgets, strict=True)):
        cumulative_exact += exact_budget
        remaining_positions = len(ordered_positions) - index - 1
        minimum_cumulative = max(
            previous_rounded + minimum_integer_budget,
            flat_total - remaining_positions * maximum_integer_budget,
        )
        maximum_cumulative = min(
            previous_rounded + maximum_integer_budget,
            flat_total - remaining_positions * minimum_integer_budget,
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
        blend=blend,
        budgets=tuple(budgets),
        total_assigned_new_visits=previous_rounded,
        flat_total_new_visits=flat_total,
    )


def allocate_candidate_budget_grid(
    positions: tuple[AllocationPosition, ...],
    baseline_new_visits: int,
    blends: tuple[Decimal, ...] = BLEND_CANDIDATES,
) -> tuple[CandidateBudgetSet, ...]:
    if len(set(blends)) != len(blends):
        raise ValueError('Candidate blends must be unique.')
    return tuple(allocate_generation_candidate(positions, baseline_new_visits, blend) for blend in sorted(blends))


def allocate_next_production_budget(
    state: SequentialBudgetState,
    baseline_new_visits: int,
    predicted_quantile: float,
    blend: Decimal,
) -> tuple[int, SequentialBudgetState]:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    ideal_budget = baseline_new_visits * blended_multiplier(predicted_quantile, blend)
    corrected_budget = ideal_budget - state.spend_error
    minimum_budget = _ceil_fraction(baseline_new_visits * _blended_bound(blend, CURVE_FLOOR))
    maximum_budget = _floor_fraction(baseline_new_visits * _blended_bound(blend, CURVE_CEILING))
    assigned_budget = min(max(_round_fraction(corrected_budget), minimum_budget), maximum_budget)
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


def production_spend_error_bound(
    baseline_new_visits: int,
) -> int:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    minimum_budget = _ceil_fraction(baseline_new_visits * CURVE_FLOOR)
    maximum_budget = _floor_fraction(baseline_new_visits * CURVE_CEILING)
    return max(baseline_new_visits - minimum_budget, maximum_budget - baseline_new_visits) + 1


def deep_label_visit_limit(baseline_new_visits: int) -> int:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    return 8 * baseline_new_visits


def _blended_bound(blend: Decimal, curve_bound: Fraction) -> Fraction:
    exact_blend = Fraction(blend)
    return (1 - exact_blend) + exact_blend * curve_bound


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


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def _identity_key(identity: LabelPositionIdentity) -> tuple[int, str, int]:
    return identity.source_generation, identity.game_identity, identity.ply

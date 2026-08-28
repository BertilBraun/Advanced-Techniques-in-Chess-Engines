from __future__ import annotations

from decimal import Decimal

import pytest
from src.search_budget.allocation import (
    AllocationPosition,
    SequentialBudgetState,
    allocate_candidate_budget_grid,
    allocate_generation_candidate,
    allocate_next_production_budget,
    deep_label_visit_limit,
    production_parallel_searches,
    production_spend_error_bound,
)
from src.search_budget.sampling import LabelPositionIdentity


def positions(quantiles: tuple[float, ...]) -> tuple[AllocationPosition, ...]:
    return tuple(
        AllocationPosition(
            identity=LabelPositionIdentity(source_generation=3, game_identity=f'game-{index // 3}', ply=index),
            predicted_quantile=quantile,
        )
        for index, quantile in enumerate(quantiles)
    )


def test_generation_candidate_preserves_exact_global_mean() -> None:
    allocation = allocate_generation_candidate(
        positions((0.0, 0.1, 0.4, 0.6, 0.8, 0.95, 1.0)),
        baseline_new_visits=600,
        blend=Decimal('1.0'),
    )
    assert allocation.total_assigned_new_visits == 7 * 600
    assert allocation.spend_error == 0
    assert sum(item.assigned_new_visits for item in allocation.budgets) == 4200
    assert all(120 <= item.assigned_new_visits <= 2871 for item in allocation.budgets)


def test_generation_normalization_never_violates_curve_floor_or_ceiling() -> None:
    skewed = positions((0.0,) * 999 + (1.0,))
    allocation = allocate_generation_candidate(skewed, 600, Decimal('1.0'))
    assert allocation.spend_error == 0
    assert all(120 <= item.assigned_new_visits <= 2871 for item in allocation.budgets)


def test_generation_normalization_is_not_repeated_per_execution_shard() -> None:
    generation_positions = positions((0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    global_allocation = allocate_generation_candidate(generation_positions, 600, Decimal('1.0'))
    first_shard = allocate_generation_candidate(generation_positions[:3], 600, Decimal('1.0'))
    second_shard = allocate_generation_candidate(generation_positions[3:], 600, Decimal('1.0'))
    assert tuple(item.assigned_new_visits for item in global_allocation.budgets) != tuple(
        item.assigned_new_visits for item in (*first_shard.budgets, *second_shard.budgets)
    )
    assert global_allocation.spend_error == first_shard.spend_error == second_shard.spend_error == 0


def test_candidate_grid_is_complete_and_every_candidate_is_exact() -> None:
    allocations = allocate_candidate_budget_grid(positions((0.1, 0.5, 0.9)), 300)
    assert tuple(allocation.blend for allocation in allocations) == tuple(Decimal(index) / 10 for index in range(11))
    assert all(allocation.spend_error == 0 for allocation in allocations)


def test_sequential_allocator_corrects_prediction_and_rounding_residual() -> None:
    state = SequentialBudgetState()
    assigned: list[int] = []
    error_bound = production_spend_error_bound(600)
    for index in range(200):
        quantile = 1.0 if index % 5 == 0 else 0.0
        budget, state = allocate_next_production_budget(state, 600, quantile, Decimal('1.0'))
        assigned.append(budget)
        assert 120 <= budget <= 2871
        assert abs(state.spend_error) <= error_bound
    assert abs(sum(assigned) - 200 * 600) <= 2272


def test_zero_blend_is_exactly_flat_even_with_prior_residual() -> None:
    state = SequentialBudgetState(cumulative_baseline_visits=600, cumulative_assigned_visits=900)
    assigned, next_state = allocate_next_production_budget(state, 600, 1.0, Decimal('0.0'))
    assert assigned == 600
    assert next_state.spend_error == 300


@pytest.mark.parametrize('baseline', [200, 300, 400, 500, 600, 700, 800, 1000])
def test_deep_label_limit_is_exactly_eight_times_source_baseline(baseline: int) -> None:
    assert deep_label_visit_limit(baseline) == 8 * baseline


@pytest.mark.parametrize(
    ('visits', 'parallel_searches'),
    [(100, 1), (200, 1), (201, 2), (300, 2), (600, 4), (1600, 8), (2400, 16), (10000, 16)],
)
def test_production_parallelism_mapping_and_cap(visits: int, parallel_searches: int) -> None:
    assert production_parallel_searches(visits) == parallel_searches

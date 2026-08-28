from __future__ import annotations

import pytest
from src.search_budget.allocation import (
    AllocationPosition,
    CurveAllocationIdentity,
    CurveAllocationPurpose,
    SequentialBudgetState,
    allocate_generation_multiplier_vector,
    allocate_next_production_budget,
    deep_label_visit_limit,
    production_parallel_searches,
    production_spend_error_bound,
)
from src.search_budget.curve import analytic_initial_curve, flat_curve
from src.search_budget.sampling import LabelPositionIdentity


def positions(quantiles: tuple[float, ...]) -> tuple[AllocationPosition, ...]:
    return tuple(
        AllocationPosition(
            identity=LabelPositionIdentity(source_generation=3, game_identity=f'game-{index // 3}', ply=index),
            predicted_quantile=quantile,
        )
        for index, quantile in enumerate(quantiles)
    )


def test_generation_curve_preserves_exact_global_mean() -> None:
    allocation = allocate_generation_multiplier_vector(
        positions((0.0, 0.1, 0.4, 0.6, 0.8, 0.95, 1.0)),
        600,
        analytic_initial_curve().multipliers,
        CurveAllocationIdentity(CurveAllocationPurpose.PENDING_VALIDATION),
    )
    assert allocation.total_assigned_new_visits == 7 * 600
    assert allocation.spend_error == 0
    assert all(1 <= item.assigned_new_visits <= 4800 for item in allocation.budgets)


def test_generation_normalization_is_not_repeated_per_execution_shard() -> None:
    generation_positions = positions((0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    identity = CurveAllocationIdentity(CurveAllocationPurpose.PENDING_VALIDATION)
    curve = analytic_initial_curve().multipliers
    global_allocation = allocate_generation_multiplier_vector(generation_positions, 600, curve, identity)
    first_shard = allocate_generation_multiplier_vector(generation_positions[:3], 600, curve, identity)
    second_shard = allocate_generation_multiplier_vector(generation_positions[3:], 600, curve, identity)
    assert tuple(item.assigned_new_visits for item in global_allocation.budgets) != tuple(
        item.assigned_new_visits for item in (*first_shard.budgets, *second_shard.budgets)
    )


def test_sequential_allocator_corrects_prediction_and_rounding_residual() -> None:
    state = SequentialBudgetState()
    error_bound = production_spend_error_bound(600)
    for index in range(200):
        quantile = 1.0 if index % 5 == 0 else 0.0
        budget, state = allocate_next_production_budget(state, 600, quantile, analytic_initial_curve())
        assert 1 <= budget <= 4800
        assert abs(state.spend_error) <= error_bound


def test_flat_curve_corrects_existing_residual_without_resetting_ledger() -> None:
    state = SequentialBudgetState(cumulative_baseline_visits=600, cumulative_assigned_visits=900)
    assigned, next_state = allocate_next_production_budget(state, 600, 1.0, flat_curve())
    assert assigned == 300
    assert next_state.spend_error == 0


@pytest.mark.parametrize('baseline', [200, 300, 400, 500, 600, 700, 800, 1000])
def test_deep_label_limit_is_exactly_eight_times_source_baseline(baseline: int) -> None:
    assert deep_label_visit_limit(baseline) == 8 * baseline


@pytest.mark.parametrize(
    ('visits', 'parallel_searches'),
    [(100, 1), (200, 1), (201, 2), (300, 2), (600, 4), (1600, 8), (2400, 16), (10000, 16)],
)
def test_production_parallelism_mapping_and_cap(visits: int, parallel_searches: int) -> None:
    assert production_parallel_searches(visits) == parallel_searches

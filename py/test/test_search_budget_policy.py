from __future__ import annotations

import math

import pytest
from src.search_budget.policy import (
    BASELINE_CURVE_INDEX,
    BUDGET_CURVE_MULTIPLES,
    BUDGET_CURVE_POINTS,
    HALF_DEEP_CURVE_INDEX,
    SearchBudgetPolicy,
    deep_label_visit_limit,
    disabled_policy,
    grid_checkpoint_visits,
    grid_visit_counts,
    log_kl_curve,
    project_non_increasing,
    select_budget_index,
    standard_normal_cdf,
)


def learned_policy(
    sigma: tuple[float, ...] = (1.0,) * BUDGET_CURVE_POINTS,
    log_tau: float = 1.0,
    selection_threshold: float = 0.8,
) -> SearchBudgetPolicy:
    return SearchBudgetPolicy(
        sigma=sigma,
        log_tau=log_tau,
        selection_threshold=selection_threshold,
        apply_learned=True,
    )


def test_grid_places_baseline_and_half_deep_multiples() -> None:
    assert len(BUDGET_CURVE_MULTIPLES) == BUDGET_CURVE_POINTS
    assert BUDGET_CURVE_MULTIPLES[BASELINE_CURVE_INDEX] == 1.0
    assert BUDGET_CURVE_MULTIPLES[HALF_DEEP_CURVE_INDEX] == 4.0
    assert tuple(sorted(BUDGET_CURVE_MULTIPLES)) == BUDGET_CURVE_MULTIPLES


@pytest.mark.parametrize('baseline', [200, 300, 400, 500, 600, 700, 800, 1000])
def test_deep_label_limit_is_exactly_eight_times_source_baseline(baseline: int) -> None:
    assert deep_label_visit_limit(baseline) == 8 * baseline


def test_grid_visits_round_half_up_with_a_floor_of_one() -> None:
    assert grid_visit_counts(600) == (75, 120, 200, 300, 400, 600, 900, 1200, 1800, 2400)
    assert grid_visit_counts(1)[0] == 1


def test_grid_checkpoints_deduplicate_but_grid_stays_ten_wide() -> None:
    visits = grid_visit_counts(4)
    assert len(visits) == BUDGET_CURVE_POINTS
    checkpoints = grid_checkpoint_visits(4)
    assert checkpoints == tuple(sorted(set(visits)))
    assert len(checkpoints) < BUDGET_CURVE_POINTS
    assert grid_visit_counts(4)[BASELINE_CURVE_INDEX] == 4


def test_curve_label_is_log_of_kl_plus_epsilon() -> None:
    kl_values = tuple(0.1 * (BUDGET_CURVE_POINTS - index) for index in range(BUDGET_CURVE_POINTS))
    assert log_kl_curve(kl_values) == tuple(math.log(value + 1e-6) for value in kl_values)


def test_isotonic_projection_is_running_minimum_from_cheapest_budget_upward() -> None:
    values = (5.0, 1.0, 4.0, 2.0, 3.0, 2.0, 9.0, 0.0, 8.0, 7.0)
    assert project_non_increasing(values) == (5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)


def test_a_well_formed_decreasing_curve_survives_the_projection_unchanged() -> None:
    # A suffix minimum would flatten this to its deepest value, reducing selection to a two-point
    # rule keyed on the 4x prediction and discarding the curve the head exists to predict.
    curve = (-1.0, -1.4, -1.9, -2.3, -2.6, -3.0, -3.4, -3.9, -4.5, -5.2)
    assert project_non_increasing(curve) == curve


def test_selection_takes_the_lowest_grid_point_whose_sigma_supports_confidence() -> None:
    # The projected curve is constant, so only the per-point sigma separates the grid points:
    # wide sigma at the cheap points blocks them and the first tight point qualifies.
    prediction = (-4.0,) * BUDGET_CURVE_POINTS
    sigma = (5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert select_budget_index(prediction, learned_policy(sigma=sigma, log_tau=-2.0)) == 4


def test_selection_falls_back_to_the_deepest_point_when_none_qualifies() -> None:
    assert select_budget_index((5.0,) * BUDGET_CURVE_POINTS, learned_policy()) == BUDGET_CURVE_POINTS - 1


def test_selection_confidence_uses_the_standard_normal_cdf_against_theta() -> None:
    # Phi(1) ~= 0.8413: a zero prediction with log_tau 1 clears theta 0.8 but not theta 0.85.
    zeros = (0.0,) * BUDGET_CURVE_POINTS
    assert select_budget_index(zeros, learned_policy(selection_threshold=0.8)) == 0
    assert select_budget_index(zeros, learned_policy(selection_threshold=0.85)) == BUDGET_CURVE_POINTS - 1
    assert standard_normal_cdf(1.0) == pytest.approx(0.8413447, abs=1e-6)


def test_a_cheap_budget_dip_propagates_to_deeper_budgets_through_isotonic_projection() -> None:
    # The projection may only pull deeper points down to a cheaper point's level, never the reverse:
    # a position that already looks converged cheaply cannot be made to look worse by more search.
    prediction = [5.0] * BUDGET_CURVE_POINTS
    prediction[6] = -3.0
    assert select_budget_index(tuple(prediction), learned_policy()) == 6


def test_wide_sigma_dilutes_confidence_and_pushes_selection_deeper() -> None:
    prediction = (0.0,) * BUDGET_CURVE_POINTS
    wide = learned_policy(sigma=(10.0,) * BUDGET_CURVE_POINTS)
    assert select_budget_index(prediction, wide) == BUDGET_CURVE_POINTS - 1


def test_selection_rejects_nonfinite_predictions() -> None:
    prediction = (float('nan'),) + (0.0,) * (BUDGET_CURVE_POINTS - 1)
    with pytest.raises(ValueError, match='finite'):
        select_budget_index(prediction, learned_policy())


def test_disabled_policy_never_applies_the_learned_rule() -> None:
    assert not disabled_policy().apply_learned


@pytest.mark.parametrize(
    'invalid',
    [
        {'sigma': (0.0,) + (1.0,) * 9},
        {'sigma': (float('nan'),) + (1.0,) * 9},
        {'log_tau': float('inf')},
        {'selection_threshold': 1.0},
        {'selection_threshold': 0.0},
    ],
)
def test_policy_rejects_invalid_parameters(invalid: dict[str, object]) -> None:
    payload: dict[str, object] = {
        'sigma': (1.0,) * BUDGET_CURVE_POINTS,
        'log_tau': 0.0,
        'selection_threshold': 0.8,
        'apply_learned': True,
    }
    payload.update(invalid)
    with pytest.raises(ValueError):
        SearchBudgetPolicy(**payload)  # type: ignore[arg-type]

from __future__ import annotations

import math

import pytest
from src.search_budget.policy import (
    BASELINE_CURVE_INDEX,
    BUDGET_CURVE_MULTIPLES,
    BUDGET_CURVE_POINTS,
    CALIBRATION_FEATURE_COUNT,
    HALF_DEEP_CURVE_INDEX,
    IDENTITY_CALIBRATION_BIAS,
    IDENTITY_CALIBRATION_WEIGHTS,
    BudgetSelectionFeatures,
    SearchBudgetPolicy,
    calibrate_curve,
    deep_label_visit_limit,
    disabled_policy,
    grid_checkpoint_visits,
    grid_visit_counts,
    log_kl_curve,
    project_non_increasing,
    select_budget_index,
)


def learned_policy(
    lagrange_multiplier: float = 1.0,
    calibration_bias: tuple[float, ...] = IDENTITY_CALIBRATION_BIAS,
    calibration_weights: tuple[tuple[float, ...], ...] = IDENTITY_CALIBRATION_WEIGHTS,
) -> SearchBudgetPolicy:
    return SearchBudgetPolicy(
        lagrange_multiplier=lagrange_multiplier,
        calibration_bias=calibration_bias,
        calibration_weights=calibration_weights,
        apply_learned=True,
    )


def neutral_features(ply: int = 0, baseline_visits: int = 400) -> BudgetSelectionFeatures:
    return BudgetSelectionFeatures(top_visit_share=1.0, policy_entropy=0.0, ply=ply, baseline_visits=baseline_visits)


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


def test_selection_minimises_raw_kl_plus_dual_priced_spend() -> None:
    # exp(curve) falls by one unit of KL per grid point; a dual of 2.5 makes the fifth step the
    # last one whose marginal KL reduction exceeds its priced spend increment.
    curve = tuple(math.log(10.0 - index) for index in range(BUDGET_CURVE_POINTS))
    objectives = tuple(
        math.exp(value) + 2.5 * multiple for value, multiple in zip(curve, BUDGET_CURVE_MULTIPLES, strict=True)
    )
    expected = min(range(BUDGET_CURVE_POINTS), key=objectives.__getitem__)
    assert select_budget_index(curve, learned_policy(lagrange_multiplier=2.5), neutral_features()) == expected
    assert expected == 5


def test_a_flat_curve_selects_the_cheapest_grid_point_under_any_positive_dual() -> None:
    flat = (0.0,) * BUDGET_CURVE_POINTS
    assert select_budget_index(flat, learned_policy(lagrange_multiplier=0.01), neutral_features()) == 0


def test_a_steeply_improving_curve_selects_the_deepest_grid_point_under_a_cheap_dual() -> None:
    curve = tuple(math.log(10.0 - index) for index in range(BUDGET_CURVE_POINTS))
    assert select_budget_index(curve, learned_policy(lagrange_multiplier=0.1), neutral_features()) == 9


def test_an_exact_objective_tie_resolves_to_the_cheapest_grid_point() -> None:
    # With a zero dual the objective is exactly the projected raw KL, which is constant here.
    flat = (-2.0,) * BUDGET_CURVE_POINTS
    assert select_budget_index(flat, learned_policy(lagrange_multiplier=0.0), neutral_features()) == 0


def test_a_cheap_budget_dip_propagates_to_deeper_budgets_through_isotonic_projection() -> None:
    # The projection may only pull deeper points down to a cheaper point's level, never the
    # reverse: the dip is the cheapest point whose projected KL is low, so it wins the argmin.
    prediction = [5.0] * BUDGET_CURVE_POINTS
    prediction[6] = -3.0
    assert select_budget_index(tuple(prediction), learned_policy(), neutral_features()) == 6


def test_a_spuriously_low_cheap_prediction_wins_the_argmin_only_at_its_own_cost() -> None:
    prediction = [5.0] * BUDGET_CURVE_POINTS
    prediction[0] = -6.0
    assert select_budget_index(tuple(prediction), learned_policy(), neutral_features()) == 0


def test_calibration_is_the_documented_affine_feature_map() -> None:
    weights = list(list(row) for row in IDENTITY_CALIBRATION_WEIGHTS)
    weights[3] = [0.5, -1.0, 0.25, 0.01, -0.001]
    bias = [0.0] * BUDGET_CURVE_POINTS
    bias[3] = 0.75
    policy = learned_policy(
        calibration_bias=tuple(bias),
        calibration_weights=tuple(tuple(row) for row in weights),
    )
    features = BudgetSelectionFeatures(top_visit_share=0.6, policy_entropy=1.2, ply=40, baseline_visits=400)
    prediction = [0.0] * BUDGET_CURVE_POINTS
    prediction[3] = -2.0
    calibrated = calibrate_curve(tuple(prediction), policy, features)
    assert calibrated[3] == pytest.approx(-2.0 + 0.75 + 0.5 * -2.0 + -1.0 * 0.6 + 0.25 * 1.2 + 0.01 * 40 - 0.001 * 400)
    assert calibrated[0] == 0.0


def test_calibration_bias_steers_selection() -> None:
    bias = [0.0] * BUDGET_CURVE_POINTS
    bias[9] = -3.0
    policy = learned_policy(lagrange_multiplier=0.01, calibration_bias=tuple(bias))
    assert select_budget_index((0.0,) * BUDGET_CURVE_POINTS, policy, neutral_features()) == 9


def test_selection_rejects_nonfinite_predictions() -> None:
    prediction = (float('nan'),) + (0.0,) * (BUDGET_CURVE_POINTS - 1)
    with pytest.raises(ValueError, match='finite'):
        select_budget_index(prediction, learned_policy(), neutral_features())


def test_features_reject_nonfinite_and_out_of_range_values() -> None:
    with pytest.raises(ValueError, match='finite'):
        BudgetSelectionFeatures(top_visit_share=float('nan'), policy_entropy=0.0, ply=0, baseline_visits=400)
    with pytest.raises(ValueError, match='positive baseline'):
        BudgetSelectionFeatures(top_visit_share=1.0, policy_entropy=0.0, ply=0, baseline_visits=0)


def test_disabled_policy_never_applies_the_learned_rule() -> None:
    assert not disabled_policy().apply_learned


@pytest.mark.parametrize(
    'invalid',
    [
        {'lagrange_multiplier': -0.1},
        {'lagrange_multiplier': float('nan')},
        {'calibration_bias': (float('inf'),) + (0.0,) * (BUDGET_CURVE_POINTS - 1)},
        {'calibration_weights': ((0.0,) * CALIBRATION_FEATURE_COUNT,) * 9 + ((0.0,) * 4,)},
        {'calibration_weights': ((float('nan'),) * CALIBRATION_FEATURE_COUNT,) * BUDGET_CURVE_POINTS},
    ],
)
def test_policy_rejects_invalid_parameters(invalid: dict[str, object]) -> None:
    payload: dict[str, object] = {
        'lagrange_multiplier': 0.5,
        'calibration_bias': IDENTITY_CALIBRATION_BIAS,
        'calibration_weights': IDENTITY_CALIBRATION_WEIGHTS,
        'apply_learned': True,
    }
    payload.update(invalid)
    with pytest.raises(ValueError):
        SearchBudgetPolicy(**payload)  # type: ignore[arg-type]

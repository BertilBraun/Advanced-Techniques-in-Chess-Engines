from __future__ import annotations

import math

import pytest
from src.search_budget.policy import (
    BASELINE_CURVE_INDEX,
    BUDGET_CURVE_MULTIPLES,
    BUDGET_CURVE_POINTS,
    HALF_DEEP_VISIT_MULTIPLE,
    BudgetSelectionFeatures,
    SearchBudgetPolicy,
    corrected_curve,
    deep_label_visit_limit,
    disabled_policy,
    grid_checkpoint_visits,
    grid_visit_counts,
    half_deep_visit_count,
    log_kl_curve,
    project_non_increasing,
    select_budget_index,
)


def learned_policy(lagrange_multiplier: float = 1.0) -> SearchBudgetPolicy:
    return SearchBudgetPolicy(
        lagrange_multiplier=lagrange_multiplier,
        corrector_path=None,
        corrector_sha256=None,
        apply_learned=True,
    )


def neutral_features(ply: int = 0, baseline_visits: int = 400) -> BudgetSelectionFeatures:
    return BudgetSelectionFeatures(
        top_visit_share=1.0,
        policy_entropy=0.0,
        ply=ply,
        baseline_visits=baseline_visits,
        source_generation=0,
    )


def test_grid_is_the_narrowed_eight_point_grid_ending_at_two_times_baseline() -> None:
    assert BUDGET_CURVE_MULTIPLES == (0.125, 0.2, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5, 2.0)
    assert len(BUDGET_CURVE_MULTIPLES) == BUDGET_CURVE_POINTS == 8
    assert BUDGET_CURVE_MULTIPLES[BASELINE_CURVE_INDEX] == 1.0
    assert tuple(sorted(BUDGET_CURVE_MULTIPLES)) == BUDGET_CURVE_MULTIPLES


@pytest.mark.parametrize('baseline', [200, 300, 400, 500, 600, 700, 800, 1000])
def test_deep_label_limit_is_exactly_eight_times_source_baseline(baseline: int) -> None:
    assert deep_label_visit_limit(baseline) == 8 * baseline


def test_grid_visits_round_half_up_with_a_floor_of_one() -> None:
    assert grid_visit_counts(600) == (75, 120, 200, 300, 400, 600, 900, 1200)
    assert grid_visit_counts(1)[0] == 1


def test_grid_checkpoints_include_the_half_deep_diagnostic_reference() -> None:
    assert HALF_DEEP_VISIT_MULTIPLE == 4
    assert half_deep_visit_count(600) == 2400
    checkpoints = grid_checkpoint_visits(600)
    assert checkpoints == tuple(sorted({*grid_visit_counts(600), 2400}))
    assert 2400 in checkpoints


def test_grid_checkpoints_deduplicate_but_grid_stays_eight_wide() -> None:
    visits = grid_visit_counts(4)
    assert len(visits) == BUDGET_CURVE_POINTS
    checkpoints = grid_checkpoint_visits(4)
    assert checkpoints == tuple(sorted({*visits, half_deep_visit_count(4)}))
    assert len(set(visits)) < BUDGET_CURVE_POINTS
    assert grid_visit_counts(4)[BASELINE_CURVE_INDEX] == 4


def test_curve_label_is_log_of_kl_plus_epsilon() -> None:
    kl_values = tuple(0.1 * (BUDGET_CURVE_POINTS - index) for index in range(BUDGET_CURVE_POINTS))
    assert log_kl_curve(kl_values) == tuple(math.log(value + 1e-6) for value in kl_values)


def test_isotonic_projection_is_running_minimum_from_cheapest_budget_upward() -> None:
    values = (5.0, 1.0, 4.0, 2.0, 3.0, 2.0, 9.0, 0.0)
    assert project_non_increasing(values) == (5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)


def test_a_well_formed_decreasing_curve_survives_the_projection_unchanged() -> None:
    # A suffix minimum would flatten this to its deepest value, reducing selection to a two-point
    # rule keyed on the deepest prediction and discarding the curve the head exists to predict.
    curve = (-1.0, -1.4, -1.9, -2.3, -2.6, -3.0, -3.4, -3.9)
    assert project_non_increasing(curve) == curve


def test_projected_curve_is_non_increasing_for_any_finite_input() -> None:
    import random

    generator = random.Random(20260831)
    for _ in range(200):
        curve = tuple(generator.uniform(-8.0, 4.0) for _ in range(BUDGET_CURVE_POINTS))
        projected = project_non_increasing(curve)
        assert all(later <= earlier for earlier, later in zip(projected, projected[1:], strict=False))
        assert all(value <= original for value, original in zip(projected, curve, strict=True))


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
    assert select_budget_index(curve, learned_policy(lagrange_multiplier=0.1), neutral_features()) == 7


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


def test_a_correction_steering_the_deepest_point_moves_the_argmin() -> None:
    def deep_correction(curve: tuple[float, ...], features: BudgetSelectionFeatures) -> tuple[float, ...]:
        return (*curve[:-1], curve[-1] - 3.0)

    flat = (0.0,) * BUDGET_CURVE_POINTS
    policy = learned_policy(lagrange_multiplier=0.01)
    assert select_budget_index(flat, policy, neutral_features(), deep_correction) == BUDGET_CURVE_POINTS - 1
    assert select_budget_index(flat, policy, neutral_features()) == 0


def test_corrected_curve_applies_the_correction_and_rejects_nonfinite_outputs() -> None:
    def shift(curve: tuple[float, ...], features: BudgetSelectionFeatures) -> tuple[float, ...]:
        return tuple(value + 0.5 for value in curve)

    curve = tuple(float(-index) for index in range(BUDGET_CURVE_POINTS))
    assert corrected_curve(curve, neutral_features(), shift) == tuple(value + 0.5 for value in curve)

    def broken(curve: tuple[float, ...], features: BudgetSelectionFeatures) -> tuple[float, ...]:
        return (float('nan'),) * BUDGET_CURVE_POINTS

    with pytest.raises(ValueError, match='finite'):
        corrected_curve(curve, neutral_features(), broken)


def test_selection_rejects_nonfinite_predictions() -> None:
    prediction = (float('nan'),) + (0.0,) * (BUDGET_CURVE_POINTS - 1)
    with pytest.raises(ValueError, match='finite'):
        select_budget_index(prediction, learned_policy(), neutral_features())


def test_features_reject_nonfinite_and_out_of_range_values() -> None:
    with pytest.raises(ValueError, match='finite'):
        BudgetSelectionFeatures(
            top_visit_share=float('nan'), policy_entropy=0.0, ply=0, baseline_visits=400, source_generation=0
        )
    with pytest.raises(ValueError, match='positive baseline'):
        BudgetSelectionFeatures(top_visit_share=1.0, policy_entropy=0.0, ply=0, baseline_visits=0, source_generation=0)
    with pytest.raises(ValueError, match='source generation'):
        BudgetSelectionFeatures(
            top_visit_share=1.0, policy_entropy=0.0, ply=0, baseline_visits=400, source_generation=-1
        )


def test_disabled_policy_never_applies_the_learned_rule() -> None:
    assert not disabled_policy().apply_learned
    assert disabled_policy().corrector_path is None


@pytest.mark.parametrize(
    'invalid',
    [
        {'lagrange_multiplier': -0.1},
        {'lagrange_multiplier': float('nan')},
        {'corrector_path': 'corrector.jit.pt'},
        {'corrector_sha256': 'a' * 64},
        {'corrector_path': 'corrector.jit.pt', 'corrector_sha256': 'not-a-digest'},
    ],
)
def test_policy_rejects_invalid_parameters(invalid: dict[str, object]) -> None:
    payload: dict[str, object] = {
        'lagrange_multiplier': 0.5,
        'corrector_path': None,
        'corrector_sha256': None,
        'apply_learned': True,
    }
    payload.update(invalid)
    with pytest.raises(ValueError):
        SearchBudgetPolicy(**payload)  # type: ignore[arg-type]


def test_policy_accepts_a_paired_corrector_reference() -> None:
    policy = SearchBudgetPolicy(
        lagrange_multiplier=0.5,
        corrector_path='corrector-generation-00000005.jit.pt',  # type: ignore[arg-type]
        corrector_sha256='0' * 64,
        apply_learned=True,
    )
    assert policy.corrector_sha256 == '0' * 64

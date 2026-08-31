from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip('AlphaZeroCpp')
from AlphaZeroCpp import FirstPlayUrgencyKind, FirstPlayUrgencyParameters, TreeSearchParameters
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.search_budget.policy import BUDGET_CURVE_MULTIPLES, SearchBudgetPolicy
from test_helpers.configuration_paths import TEST_CONFIG_DIRECTORY


def _zero_first_play_urgency() -> FirstPlayUrgencyParameters:
    return FirstPlayUrgencyParameters(FirstPlayUrgencyKind.ZERO)


def test_tree_search_parameters_default_virtual_loss_weight_is_a_full_loss() -> None:
    parameters = TreeSearchParameters(
        exploration_constant=1.5,
        first_play_urgency=_zero_first_play_urgency(),
        forced_playout_coefficient=0.0,
        value_discount_per_ply=1.0,
    )

    assert parameters.virtual_loss_weight == 1.0


def test_tree_search_parameters_accept_a_fractional_virtual_loss_weight() -> None:
    parameters = TreeSearchParameters(
        exploration_constant=1.5,
        first_play_urgency=_zero_first_play_urgency(),
        forced_playout_coefficient=0.0,
        value_discount_per_ply=1.0,
        virtual_loss_weight=0.5,
    )

    assert parameters.virtual_loss_weight == 0.5


@pytest.mark.parametrize('invalid_weight', (-0.5, 1.5, float('nan')))
def test_tree_search_parameters_reject_invalid_virtual_loss_weights(invalid_weight: float) -> None:
    with pytest.raises(ValueError, match='Virtual-loss weight'):
        TreeSearchParameters(
            exploration_constant=1.5,
            first_play_urgency=_zero_first_play_urgency(),
            forced_playout_coefficient=0.0,
            value_discount_per_ply=1.0,
            virtual_loss_weight=invalid_weight,
        )


def test_resolved_virtual_loss_weight_reaches_the_native_search_parameters() -> None:
    configuration = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    assert isinstance(configuration, ChessExperimentConfiguration)
    implementation = ChessImplementation(configuration)
    policy = SearchBudgetPolicy(
        lagrange_multiplier=0.4,
        calibration_bias=tuple(0.01 * index for index in range(10)),
        calibration_weights=tuple(tuple(0.001 * (row + column) for column in range(5)) for row in range(10)),
        apply_learned=True,
    )
    resolved = replace(implementation.self_play_parameters_at(0, policy), virtual_loss_weight=0.25)

    native_parameters = implementation.native_search_parameters(resolved)

    assert native_parameters.tree_search.virtual_loss_weight == pytest.approx(0.25)
    assert native_parameters.baseline_visits == resolved.baseline_visits
    native_policy = native_parameters.search_budget_policy
    assert tuple(native_policy.multiples) == pytest.approx(BUDGET_CURVE_MULTIPLES)
    assert native_policy.lagrange_multiplier == pytest.approx(0.4)
    assert tuple(native_policy.calibration_bias) == pytest.approx(policy.calibration_bias)
    for native_row, python_row in zip(native_policy.calibration_weights, policy.calibration_weights, strict=True):
        assert tuple(native_row) == pytest.approx(python_row)
    assert native_policy.apply_learned is True


def _float32(value: float) -> float:
    import numpy as np

    return float(np.float32(value))


def test_native_and_python_budget_selection_agree_on_identical_inputs() -> None:
    import random

    from AlphaZeroCpp import SearchBudgetPolicy as NativeSearchBudgetPolicy
    from AlphaZeroCpp import SearchBudgetSelectionFeatures
    from AlphaZeroCpp import calibrate_budget_curve as native_calibrate
    from AlphaZeroCpp import select_budget_index as native_select
    from src.search_budget.policy import BudgetSelectionFeatures, calibrate_curve, select_budget_index

    generator = random.Random(20260830)
    for _ in range(250):
        # The native curve is float32, so feed values that are exact in both precisions.
        prediction = tuple(_float32(generator.uniform(-8.0, 4.0)) for _ in range(10))
        bias = tuple(generator.uniform(-0.5, 0.5) for _ in range(10))
        weights = tuple(tuple(generator.uniform(-0.05, 0.05) for _ in range(5)) for _ in range(10))
        lagrange_multiplier = generator.uniform(0.0, 2.0)
        python_policy = SearchBudgetPolicy(
            lagrange_multiplier=lagrange_multiplier,
            calibration_bias=bias,
            calibration_weights=weights,
            apply_learned=True,
        )
        native_policy = NativeSearchBudgetPolicy(
            list(BUDGET_CURVE_MULTIPLES),
            lagrange_multiplier,
            list(bias),
            [list(row) for row in weights],
            True,
        )
        ply = generator.randrange(0, 300)
        baseline_visits = generator.choice((300, 400, 600))
        python_features = BudgetSelectionFeatures(
            top_visit_share=_float32(generator.uniform(0.05, 1.0)),
            policy_entropy=_float32(generator.uniform(0.0, 4.0)),
            ply=ply,
            baseline_visits=baseline_visits,
        )
        native_features = SearchBudgetSelectionFeatures(
            top_visit_share=python_features.top_visit_share,
            policy_entropy=python_features.policy_entropy,
            ply=float(ply),
            baseline_visits=float(baseline_visits),
        )
        assert native_select(native_policy, list(prediction), native_features) == select_budget_index(
            prediction, python_policy, python_features
        )
        native_curve = native_calibrate(native_policy, list(prediction), native_features)
        assert tuple(native_curve) == pytest.approx(calibrate_curve(prediction, python_policy, python_features))

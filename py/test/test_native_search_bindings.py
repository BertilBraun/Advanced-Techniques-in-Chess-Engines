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
        sigma=tuple(0.5 + 0.1 * index for index in range(10)),
        log_tau=-2.25,
        selection_threshold=0.8,
        apply_learned=True,
    )
    resolved = replace(implementation.self_play_parameters_at(0, policy), virtual_loss_weight=0.25)

    native_parameters = implementation.native_search_parameters(resolved)

    assert native_parameters.tree_search.virtual_loss_weight == pytest.approx(0.25)
    assert native_parameters.baseline_visits == resolved.baseline_visits
    native_policy = native_parameters.search_budget_policy
    assert tuple(native_policy.multiples) == pytest.approx(BUDGET_CURVE_MULTIPLES)
    assert tuple(native_policy.sigma) == pytest.approx(policy.sigma)
    assert native_policy.log_tau == pytest.approx(-2.25)
    assert native_policy.selection_threshold == pytest.approx(0.8)
    assert native_policy.apply_learned is True

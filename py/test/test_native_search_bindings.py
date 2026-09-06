from __future__ import annotations

import pytest

pytest.importorskip('AlphaZeroCpp')
from AlphaZeroCpp import FirstPlayUrgencyKind, FirstPlayUrgencyParameters, TreeSearchParameters
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
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


def test_resolved_self_play_parameters_reach_the_native_search_parameters() -> None:
    configuration = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    assert isinstance(configuration, ChessExperimentConfiguration)
    implementation = ChessImplementation(configuration)

    parameters = implementation.self_play_parameters_at(0)
    native_parameters = implementation.native_search_parameters(parameters)

    assert native_parameters.baseline_visits == parameters.baseline_visits
    assert native_parameters.tree_search.exploration_constant == pytest.approx(parameters.exploration_constant)
    assert native_parameters.dirichlet_alpha == pytest.approx(parameters.dirichlet_alpha)
    assert native_parameters.dirichlet_epsilon == pytest.approx(parameters.dirichlet_epsilon)
    implementation.close()


def test_the_search_stop_reason_enum_carries_only_the_fixed_limit_reasons() -> None:
    import AlphaZeroCpp

    assert set(AlphaZeroCpp.SearchStopReason.__members__) == {'FIXED_LIMIT', 'ADDITIONAL_VISITS'}

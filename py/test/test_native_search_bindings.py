from __future__ import annotations

from dataclasses import replace

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


def test_resolved_virtual_loss_weight_reaches_the_native_search_parameters() -> None:
    configuration = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    assert isinstance(configuration, ChessExperimentConfiguration)
    implementation = ChessImplementation(configuration)
    resolved = replace(implementation.self_play_parameters_at(0), virtual_loss_weight=0.25)

    native_parameters = implementation.native_search_parameters(resolved)

    assert native_parameters.tree_search.virtual_loss_weight == pytest.approx(0.25)
    assert native_parameters.parallel_searches == resolved.parallel_searches

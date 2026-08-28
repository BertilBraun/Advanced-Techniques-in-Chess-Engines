from __future__ import annotations

import pytest
from src.evaluation.configuration import EvaluationSearchConfiguration, EvaluationTreeSearchOverrides
from src.experiment.configuration import (
    EXPERIMENT_CONFIGURATION_ADAPTER,
    ExperimentConfiguration,
    load_experiment_configuration,
)
from src.games.chess.training import ChessImplementation
from src.search_budget.curve import analytic_initial_curve, flat_curve
from src.self_play.configuration import (
    FirstPlayUrgencyConfiguration,
    ParentValueFirstPlayUrgencyConfiguration,
    ReducedParentValueFirstPlayUrgencyConfiguration,
    ZeroFirstPlayUrgencyConfiguration,
)
from src.self_play.parameters import (
    FirstPlayUrgencyParameters,
    ParentValueFirstPlayUrgencyParameters,
    ReducedParentValueFirstPlayUrgencyParameters,
    ZeroFirstPlayUrgencyParameters,
)
from src.util.generation_schedule import ConstantSchedule
from test_helpers.configuration_paths import TEST_CONFIG_DIRECTORY


def _experiment() -> ExperimentConfiguration:
    return load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')


def _evaluation_search(experiment: ExperimentConfiguration) -> EvaluationSearchConfiguration:
    definition = next(
        candidate for candidate in experiment.evaluation.definitions if candidate.kind == 'previous_checkpoint'
    )
    return definition.search


def test_evaluation_inherits_the_self_play_first_play_urgency() -> None:
    experiment = _experiment()
    implementation = ChessImplementation(experiment)

    self_play = implementation.self_play_parameters_at(0, analytic_initial_curve())
    evaluation = implementation.evaluation_parameters_at(0, _evaluation_search(experiment))

    assert evaluation.first_play_urgency == self_play.first_play_urgency


def test_evaluation_overrides_only_the_search_shaping_fields() -> None:
    experiment = _experiment()
    implementation = ChessImplementation(experiment)
    search = _evaluation_search(experiment)

    evaluation = implementation.evaluation_parameters_at(0, search)

    assert evaluation.baseline_visits == search.searches_per_move
    assert evaluation.search_budget_curve == flat_curve()
    assert evaluation.exploration_constant == pytest.approx(search.resolved_exploration_constant)
    assert evaluation.forced_playout_coefficient == pytest.approx(0.0)
    assert evaluation.dirichlet_epsilon == pytest.approx(0.0)


def test_a_reduced_parent_value_self_play_urgency_reaches_evaluation() -> None:
    experiment = _experiment()
    payload = experiment.model_dump(mode='json')
    payload['chess']['self_play']['search']['first_play_urgency'] = {
        'kind': 'reduced_parent_value',
        'reduction': 0.2,
    }
    implementation = ChessImplementation(EXPERIMENT_CONFIGURATION_ADAPTER.validate_python(payload))

    evaluation = implementation.evaluation_parameters_at(0, _evaluation_search(experiment))

    assert evaluation.first_play_urgency == ReducedParentValueFirstPlayUrgencyParameters(reduction=0.2)


@pytest.mark.parametrize(
    ('configured', 'expected'),
    (
        (ZeroFirstPlayUrgencyConfiguration(), ZeroFirstPlayUrgencyParameters()),
        (ParentValueFirstPlayUrgencyConfiguration(), ParentValueFirstPlayUrgencyParameters()),
        (
            ReducedParentValueFirstPlayUrgencyConfiguration(reduction=ConstantSchedule[float](value=0.2)),
            ReducedParentValueFirstPlayUrgencyParameters(reduction=0.2),
        ),
    ),
)
def test_evaluation_search_applies_requested_overrides(
    configured: FirstPlayUrgencyConfiguration,
    expected: FirstPlayUrgencyParameters,
) -> None:
    experiment = _experiment()
    implementation = ChessImplementation(experiment)
    overrides = EvaluationTreeSearchOverrides(
        first_play_urgency=configured,
        virtual_loss_weight=0.25,
        value_discount_per_ply=0.98,
    )

    parameters = implementation.evaluation_parameters_at(12, _evaluation_search(experiment), overrides)

    assert parameters.first_play_urgency == expected
    assert parameters.virtual_loss_weight == pytest.approx(0.25)
    assert parameters.value_discount_per_ply == pytest.approx(0.98)

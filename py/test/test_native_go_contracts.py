from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip('AlphaZeroCpp')
from src.experiment.configuration import load_experiment_configuration
from src.games.go.configuration import GoExperimentConfiguration
from src.games.go.training import GoImplementation
from test_helpers.configuration_paths import TEST_CONFIG_DIRECTORY


@pytest.mark.parametrize(
    ('path', 'expected_action_size'),
    (
        (TEST_CONFIG_DIRECTORY / 'go-7x7-experiment.yaml', 50),
        (TEST_CONFIG_DIRECTORY / 'go-9x9-experiment.yaml', 82),
    ),
)
def test_go_root_implementation_owns_action_id_state_and_fixed_target_layout(
    path: Path,
    expected_action_size: int,
) -> None:
    configuration = load_experiment_configuration(path)
    assert isinstance(configuration, GoExperimentConfiguration)
    implementation = GoImplementation(configuration)

    assert implementation.state.action_size == expected_action_size
    assert implementation.state.augmentation_count == 8
    assert implementation.target_layout.action_size == expected_action_size
    assert implementation.target_layout.wdl_size == 3
    assert implementation.target_layout.auxiliary_heads == ()


def test_go_action_permutations_are_cached_and_preserve_pass() -> None:
    configuration = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'go-7x7-experiment.yaml')
    assert isinstance(configuration, GoExperimentConfiguration)
    state = GoImplementation(configuration).state
    permutations = state.action_permutations

    assert permutations is state.action_permutations
    assert not permutations.flags.writeable
    assert tuple(permutations[:, state.pass_action]) == (state.pass_action,) * state.augmentation_count

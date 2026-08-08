from pathlib import Path

import pytest


pytest.importorskip('AlphaZeroCpp')

from src.experiment.configuration import load_experiment_configuration
from src.games.go.configuration import GoExperimentConfiguration
from src.games.go.training import GoImplementation


@pytest.mark.parametrize(
    ('path', 'expected_action_size'),
    (
        (Path('configs/go-7x7-experiment-template.yaml'), 50),
        (Path('configs/go-9x9-experiment-template.yaml'), 82),
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

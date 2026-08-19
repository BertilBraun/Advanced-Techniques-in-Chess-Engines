from pathlib import Path

import pytest


pytest.importorskip('AlphaZeroCpp')
from AlphaZeroCpp import GameSearchVisit

from src.experiment.configuration import load_experiment_configuration
from src.games.go.configuration import GoExperimentConfiguration
from src.games.go.training import GoImplementation
from src.games.contracts import WdlTarget
from src.replay.contracts import EligibleRemainingGameLengthTarget, ReplaySample, SparsePolicyTarget
from src.self_play.completed_game import SearchVisitCounts


@pytest.mark.parametrize(
    ('path', 'expected_action_size'),
    (
        (Path('test/configs/go-7x7-experiment.yaml'), 50),
        (Path('test/configs/go-9x9-experiment.yaml'), 82),
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


def test_go_symmetry_leaves_remaining_game_length_unchanged() -> None:
    configuration = load_experiment_configuration(Path('test/configs/go-7x7-experiment.yaml'))
    assert isinstance(configuration, GoExperimentConfiguration)
    state = GoImplementation(configuration).state
    position = state.initial_position()
    action_id = state.legal_action_ids(position)[0]
    sample = ReplaySample(
        encoded_state=state.encode_network_input(position),
        policy=SparsePolicyTarget(
            visits=SearchVisitCounts.from_native((GameSearchVisit(action_id=action_id, visit_count=1),)),
            legal_action_ids=state.legal_action_ids(position),
        ),
        wdl_target=WdlTarget(win=0.0, draw=1.0, loss=0.0),
        root_value=0.0,
        auxiliary_targets=(EligibleRemainingGameLengthTarget(normalized_length=0.5),),
        sample_weight=1.0,
        source_model_generation=0,
        source_created_at_seconds=1.0,
    )

    transformed = state.transform_replay_targets(sample, augmentation_index=1)

    assert transformed.auxiliary_targets == (EligibleRemainingGameLengthTarget(normalized_length=0.5),)

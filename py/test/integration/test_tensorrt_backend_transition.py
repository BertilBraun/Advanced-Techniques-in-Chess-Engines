from __future__ import annotations

import os
from pathlib import Path

import pytest
from tools.verify_tensorrt_backend_transition import Arguments, run


@pytest.mark.integration
def test_torchscript_bootstrap_transitions_to_tensorrt(tmp_path: Path) -> None:
    configuration = os.environ.get('ALPHAZERO_TENSORRT_TRANSITION_CONFIGURATION')
    checkpoint_directory = os.environ.get('ALPHAZERO_TENSORRT_TRANSITION_CHECKPOINT_DIRECTORY')
    candidate_generation = os.environ.get('ALPHAZERO_TENSORRT_TRANSITION_CANDIDATE_GENERATION')
    if configuration is None or checkpoint_directory is None or candidate_generation is None:
        pytest.skip('TensorRT transition integration artifacts are not configured.')
    report = run(
        Arguments(
            configuration_path=Path(configuration),
            checkpoint_directory=Path(checkpoint_directory),
            bootstrap_generation=0,
            candidate_generation=int(candidate_generation),
            device_id=0,
            maximum_policy_absolute_error=1e-6,
            maximum_value_absolute_error=1e-6,
            output_path=tmp_path / 'transition-report.json',
        )
    )
    assert report.bootstrap_and_candidate_differ
    assert report.all_selected_actions_match
    assert report.maximum_transition_policy_error <= 1e-6
    assert report.maximum_transition_value_error <= 1e-6
    assert report.passed

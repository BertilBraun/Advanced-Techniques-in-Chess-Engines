from __future__ import annotations

import pytest
from src.training.quantization.configuration import (
    QatCheckpointPhase,
    TensorRtInt8QatConfiguration,
    expected_qat_phase,
)


@pytest.mark.parametrize(
    ('completed_optimizer_steps', 'expected'),
    (
        (0, QatCheckpointPhase.PRE_FOLD),
        (999, QatCheckpointPhase.PRE_FOLD),
        (1_000, QatCheckpointPhase.DEPLOYMENT),
        (10_000, QatCheckpointPhase.DEPLOYMENT),
    ),
)
def test_expected_qat_phase_is_stable_across_resume(
    completed_optimizer_steps: int,
    expected: QatCheckpointPhase,
) -> None:
    configuration = TensorRtInt8QatConfiguration(
        fold_after_optimizer_steps=1_000,
        deployment_learning_rate=0.02,
    )

    assert expected_qat_phase(configuration, completed_optimizer_steps) is expected


def test_expected_qat_phase_rejects_negative_progress() -> None:
    with pytest.raises(ValueError, match='nonnegative'):
        expected_qat_phase(TensorRtInt8QatConfiguration(deployment_learning_rate=0.02), -1)

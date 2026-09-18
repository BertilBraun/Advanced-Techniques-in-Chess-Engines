from __future__ import annotations

import pytest
from src.training.quantization.configuration import (
    QatCalibrationSource,
    QatCheckpointPhase,
    QatStateIdentity,
    TensorRtInt8QatConfiguration,
    expected_qat_phase,
    qat_phase_warmup_progress,
)


def test_qat_replay_calibration_source_is_explicit() -> None:
    configuration = TensorRtInt8QatConfiguration(
        calibration_source=QatCalibrationSource.REPLAY,
        calibration_positions=10_000,
        recalibration_interval_generations=5,
        deployment_learning_rate='inherit',
        deployment_warmup_optimizer_steps=0,
    )

    assert configuration.calibration_source is QatCalibrationSource.REPLAY
    assert configuration.calibration_positions == 10_000
    assert configuration.recalibration_interval_generations == 5


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
        deployment_warmup_optimizer_steps=0,
    )

    assert expected_qat_phase(configuration, completed_optimizer_steps) is expected


def test_expected_qat_phase_rejects_negative_progress() -> None:
    with pytest.raises(ValueError, match='nonnegative'):
        expected_qat_phase(
            TensorRtInt8QatConfiguration(deployment_learning_rate=0.02, deployment_warmup_optimizer_steps=0), -1
        )


@pytest.mark.parametrize(
    ('phase', 'completed_optimizer_steps', 'expected_completed_steps'),
    (
        (QatCheckpointPhase.PRE_FOLD, 2_500, 2_500),
        (QatCheckpointPhase.DEPLOYMENT, 5_000, 0),
        (QatCheckpointPhase.DEPLOYMENT, 7_499, 2_499),
        (QatCheckpointPhase.DEPLOYMENT, 10_000, 5_000),
    ),
)
def test_qat_fold_starts_an_independent_deployment_warmup(
    phase: QatCheckpointPhase,
    completed_optimizer_steps: int,
    expected_completed_steps: int,
) -> None:
    configuration = TensorRtInt8QatConfiguration(
        fold_after_optimizer_steps=5_000,
        deployment_learning_rate='inherit',
        deployment_warmup_optimizer_steps=5_000,
        deployment_warmup_start_learning_rate=0.002,
    )
    state = QatStateIdentity(
        phase=phase,
        completed_optimizer_steps=completed_optimizer_steps,
        path='qat-state.pt',
        sha256='0' * 64,
    )

    progress = qat_phase_warmup_progress(5_000, 0.001, configuration, state, completed_optimizer_steps)

    assert progress.warmup_optimizer_steps == 5_000
    assert progress.completed_optimizer_steps == expected_completed_steps
    assert progress.start_learning_rate == (0.001 if phase is QatCheckpointPhase.PRE_FOLD else 0.002)

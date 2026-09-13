from __future__ import annotations

import pytest
from src.training.quantization.configuration import (
    QatCheckpointPhase,
    QatStateIdentity,
    TensorRtInt8QatConfiguration,
    qat_phase_warmup_progress,
)
from src.training.trainer.rank import qat_phase_learning_rate, warmup_scaled_learning_rate
from src.util.generation_schedule import LinearSchedule, ScheduleRounding


@pytest.mark.parametrize(
    ('warmup_optimizer_steps', 'completed_optimizer_steps', 'expected'),
    (
        (0, 0, 0.005),
        (0, 12345, 0.005),
        (1000, 0, 0.005 * 1 / 1000),
        (1000, 499, 0.005 * 500 / 1000),
        (1000, 999, 0.005),
        (1000, 1000, 0.005),
        (1000, 5000, 0.005),
    ),
)
def test_warmup_scales_learning_rate_linearly_until_the_configured_step(
    warmup_optimizer_steps: int,
    completed_optimizer_steps: int,
    expected: float,
) -> None:
    scaled = warmup_scaled_learning_rate(0.005, warmup_optimizer_steps, completed_optimizer_steps)

    assert scaled == pytest.approx(expected)


@pytest.mark.parametrize(
    ('completed_optimizer_steps', 'model_generation', 'phase', 'expected'),
    (
        (500, 1, QatCheckpointPhase.PRE_FOLD, 0.0501),
        (1_000, 2, QatCheckpointPhase.DEPLOYMENT, 0.02),
        (1_500, 3, QatCheckpointPhase.DEPLOYMENT, 0.02 - 0.01 / 998),
    ),
)
def test_qat_deployment_learning_rate_replaces_prefold_warmup_after_optimizer_reset(
    completed_optimizer_steps: int,
    model_generation: int,
    phase: QatCheckpointPhase,
    expected: float,
) -> None:
    configuration = TensorRtInt8QatConfiguration(
        fold_after_optimizer_steps=1_000,
        deployment_warmup_optimizer_steps=0,
        deployment_learning_rate=LinearSchedule[float](
            start_generation=2,
            end_generation=1_000,
            start_value=0.02,
            end_value=0.01,
            rounding=ScheduleRounding.NONE,
        ),
    )
    state = QatStateIdentity(
        phase=phase,
        completed_optimizer_steps=completed_optimizer_steps,
        path='qat-state.pt',
        sha256='0' * 64,
    )
    phase_rate = qat_phase_learning_rate(0.1, configuration, state, model_generation)

    actual = warmup_scaled_learning_rate(phase_rate, 1_000, completed_optimizer_steps)

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ('phase', 'completed_optimizer_steps', 'expected_warmup_steps', 'expected_completed_steps'),
    (
        (QatCheckpointPhase.PRE_FOLD, 2_500, 5_000, 2_500),
        (QatCheckpointPhase.DEPLOYMENT, 5_000, 5_000, 0),
        (QatCheckpointPhase.DEPLOYMENT, 7_499, 5_000, 2_499),
        (QatCheckpointPhase.DEPLOYMENT, 10_000, 5_000, 5_000),
    ),
)
def test_qat_fold_starts_an_independent_deployment_warmup(
    phase: QatCheckpointPhase,
    completed_optimizer_steps: int,
    expected_warmup_steps: int,
    expected_completed_steps: int,
) -> None:
    configuration = TensorRtInt8QatConfiguration(
        fold_after_optimizer_steps=5_000,
        deployment_learning_rate='inherit',
        deployment_warmup_optimizer_steps=5_000,
    )
    state = QatStateIdentity(
        phase=phase,
        completed_optimizer_steps=completed_optimizer_steps,
        path='qat-state.pt',
        sha256='0' * 64,
    )

    progress = qat_phase_warmup_progress(5_000, configuration, state, completed_optimizer_steps)

    assert progress.warmup_optimizer_steps == expected_warmup_steps
    assert progress.completed_optimizer_steps == expected_completed_steps


def test_inherited_deployment_learning_rate_preserves_the_session_rate() -> None:
    configuration = TensorRtInt8QatConfiguration(
        fold_after_optimizer_steps=5_000,
        deployment_learning_rate='inherit',
        deployment_warmup_optimizer_steps=5_000,
    )
    state = QatStateIdentity(
        phase=QatCheckpointPhase.DEPLOYMENT,
        completed_optimizer_steps=5_000,
        path='qat-state.pt',
        sha256='0' * 64,
    )

    assert qat_phase_learning_rate(0.073, configuration, state, model_generation=500) == pytest.approx(0.073)

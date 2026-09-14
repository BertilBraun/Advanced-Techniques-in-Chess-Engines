from __future__ import annotations

import pytest
from tools.run_sgd_replay_screen import ARM_SCHEDULES, ScreenArm, _learning_rate, _phase


@pytest.mark.parametrize(
    ('arm', 'peak_learning_rate', 'fold_step'),
    (
        (ScreenArm.HISTORICAL_FOLD_1000, 0.1, 1_000),
        (ScreenArm.CONTINUOUS_FOLD_1000, 0.02, 1_000),
        (ScreenArm.HISTORICAL_FOLD_3000, 0.1, 3_000),
        (ScreenArm.CONTINUOUS_FOLD_3000, 0.02, 3_000),
    ),
)
def test_prefold_factorial_schedule(arm: ScreenArm, peak_learning_rate: float, fold_step: int) -> None:
    schedule = ARM_SCHEDULES[arm]

    assert schedule.pre_fold_peak_learning_rate == pytest.approx(peak_learning_rate)
    assert schedule.fold_after_optimizer_steps == fold_step
    assert _learning_rate(schedule.pre_fold_warmup_steps, schedule) == pytest.approx(peak_learning_rate)
    assert _learning_rate(fold_step, schedule) == pytest.approx(peak_learning_rate)
    assert _learning_rate(fold_step + 1, schedule) == pytest.approx(0.02)
    assert _phase(fold_step - 1, schedule) == 'pre_fold'
    assert _phase(fold_step, schedule) == 'deployment'


def test_continuous_schedule_has_no_fold_learning_rate_discontinuity() -> None:
    schedule = ARM_SCHEDULES[ScreenArm.CONTINUOUS_FOLD_3000]

    assert _learning_rate(1, schedule) == pytest.approx(0.0001199)
    assert _learning_rate(1_000, schedule) == pytest.approx(0.02)
    assert _learning_rate(3_000, schedule) == pytest.approx(0.02)
    assert _learning_rate(3_001, schedule) == pytest.approx(0.02)


def test_historical_schedule_reproduces_the_v35_fold_drop() -> None:
    schedule = ARM_SCHEDULES[ScreenArm.HISTORICAL_FOLD_1000]

    assert _learning_rate(1, schedule) == pytest.approx(0.0001)
    assert _learning_rate(500, schedule) == pytest.approx(0.05)
    assert _learning_rate(1_000, schedule) == pytest.approx(0.1)
    assert _learning_rate(1_001, schedule) == pytest.approx(0.02)


@pytest.mark.parametrize(
    ('arm', 'target_learning_rate'),
    (
        (ScreenArm.POST_FOLD_002, 0.02),
        (ScreenArm.POST_FOLD_004, 0.04),
        (ScreenArm.POST_FOLD_006, 0.06),
        (ScreenArm.POST_FOLD_008, 0.08),
    ),
)
def test_post_fold_learning_rate_sweep(arm: ScreenArm, target_learning_rate: float) -> None:
    schedule = ARM_SCHEDULES[arm]

    assert schedule.pre_fold_peak_learning_rate == pytest.approx(0.1)
    assert schedule.fold_after_optimizer_steps == 3_000
    assert _learning_rate(3_000, schedule) == pytest.approx(0.1)
    assert _learning_rate(3_001, schedule) == pytest.approx(0.02 + (target_learning_rate - 0.02) / 500)
    assert _learning_rate(3_250, schedule) == pytest.approx(0.02 + (target_learning_rate - 0.02) / 2)
    assert _learning_rate(3_500, schedule) == pytest.approx(target_learning_rate)
    assert _learning_rate(10_000, schedule) == pytest.approx(target_learning_rate)

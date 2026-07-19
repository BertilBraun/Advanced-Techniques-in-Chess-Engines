import pytest

from src.experiment.evaluation_schedule import (
    evaluation_device_for_task,
    select_historical_model_iterations,
)


HISTORICAL_MODELS = tuple(range(20, 481, 20))


def test_evaluation_devices_are_assigned_round_robin() -> None:
    assigned_devices = tuple(evaluation_device_for_task((0, 1, 2, 3), index) for index in range(10))

    assert assigned_devices == (0, 1, 2, 3, 0, 1, 2, 3, 0, 1)


@pytest.mark.parametrize(
    ('device_cycle', 'task_index', 'message'),
    (
        ((), 0, 'cannot be empty'),
        ((0,), -1, 'cannot be negative'),
    ),
)
def test_evaluation_device_assignment_rejects_invalid_inputs(
    device_cycle: tuple[int, ...],
    task_index: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluation_device_for_task(device_cycle, task_index)


@pytest.mark.parametrize(
    ('current_iteration', 'expected_iterations'),
    (
        (142, (40, 140)),
        (143, (60,)),
        (144, (80,)),
        (145, (100,)),
        (146, (20, 120)),
        (342, (40, 140, 240, 340)),
    ),
)
def test_historical_models_rotate_across_five_iteration_buckets(
    current_iteration: int,
    expected_iterations: tuple[int, ...],
) -> None:
    selected = select_historical_model_iterations(
        current_iteration,
        HISTORICAL_MODELS,
        milestone_interval=20,
        rotation_period=5,
    )

    assert selected == expected_iterations


def test_rotation_period_one_preserves_full_historical_sweep() -> None:
    selected = select_historical_model_iterations(
        142,
        HISTORICAL_MODELS,
        milestone_interval=20,
        rotation_period=1,
    )

    assert selected == (20, 40, 60, 80, 100, 120, 140)


@pytest.mark.parametrize(
    ('milestone_interval', 'rotation_period', 'message'),
    (
        (0, 5, 'Milestone interval'),
        (20, 0, 'rotation period'),
    ),
)
def test_historical_model_rotation_rejects_non_positive_periods(
    milestone_interval: int,
    rotation_period: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_historical_model_iterations(
            142,
            HISTORICAL_MODELS,
            milestone_interval,
            rotation_period,
        )

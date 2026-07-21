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
def test_historical_models_rotate_across_five_evaluation_buckets(
    current_iteration: int,
    expected_iterations: tuple[int, ...],
) -> None:
    selected = select_historical_model_iterations(
        current_iteration,
        HISTORICAL_MODELS,
        milestone_interval=20,
        rotation_period=5,
        evaluation_interval=1,
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
    ('current_iteration', 'expected_iterations'),
    (
        (76, (0, 40)),
        (78, (20, 60)),
        (80, (0, 40)),
    ),
)
def test_historical_models_alternate_on_every_evaluation(
    current_iteration: int,
    expected_iterations: tuple[int, ...],
) -> None:
    selected = select_historical_model_iterations(
        current_iteration,
        (0, 20, 40, 60, 80),
        milestone_interval=20,
        rotation_period=2,
        evaluation_interval=2,
    )

    assert selected == expected_iterations


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


def test_historical_model_rotation_rejects_non_positive_evaluation_intervals() -> None:
    with pytest.raises(ValueError, match='Evaluation interval'):
        select_historical_model_iterations(10, HISTORICAL_MODELS, 20, 2, evaluation_interval=0)

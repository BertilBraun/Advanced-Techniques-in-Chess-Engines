import pytest

from src.experiment.training_data_schedule import select_validation_iteration


@pytest.mark.parametrize(
    ('current_iteration', 'current_iteration_has_files', 'expected_iteration'),
    (
        (142, True, 142),
        (142, False, 141),
        (0, True, 0),
    ),
)
def test_validation_iteration_uses_the_newest_available_replay(
    current_iteration: int,
    current_iteration_has_files: bool,
    expected_iteration: int,
) -> None:
    assert select_validation_iteration(current_iteration, current_iteration_has_files) == expected_iteration


@pytest.mark.parametrize(
    ('current_iteration', 'current_iteration_has_files', 'message'),
    (
        (-1, True, 'cannot be negative'),
        (0, False, 'cannot fall back'),
    ),
)
def test_validation_iteration_rejects_invalid_boundaries(
    current_iteration: int,
    current_iteration_has_files: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_validation_iteration(current_iteration, current_iteration_has_files)

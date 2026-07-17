import pytest

from src.self_play.curriculum import curriculum_progress


@pytest.mark.parametrize(
    ('iteration', 'warmup_iterations', 'expected_progress'),
    (
        (0, 0, 1.0),
        (0, 10, 0.0),
        (5, 10, 0.5),
        (10, 10, 1.0),
        (20, 10, 1.0),
    ),
)
def test_curriculum_progress(
    iteration: int,
    warmup_iterations: int,
    expected_progress: float,
) -> None:
    assert curriculum_progress(iteration, warmup_iterations) == pytest.approx(expected_progress)


@pytest.mark.parametrize(
    ('iteration', 'warmup_iterations'),
    ((-1, 10), (0, -1)),
)
def test_curriculum_rejects_negative_values(
    iteration: int,
    warmup_iterations: int,
) -> None:
    with pytest.raises(ValueError):
        curriculum_progress(iteration, warmup_iterations)

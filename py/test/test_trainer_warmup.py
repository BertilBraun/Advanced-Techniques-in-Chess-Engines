import pytest

from src.training.trainer.rank import warmup_scaled_learning_rate


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

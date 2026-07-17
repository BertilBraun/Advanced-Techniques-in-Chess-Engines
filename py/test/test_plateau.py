import pytest

from src.experiment.plateau import (
    CheckpointEvaluation,
    PlateauRule,
    PlateauStatus,
    evaluate_plateau,
)


RULE = PlateauRule(
    minimum_iteration=15,
    evaluation_spacing=5,
    consecutive_evaluations=3,
    minimum_score_gain=0.025,
    maximum_score_interval_width=0.12,
    minimum_policy_top_1_gain=0.002,
    minimum_value_loss_reduction=0.002,
    regression_score_confidence_high=0.45,
)


def checkpoint(
    iteration: int,
    score: float,
    confidence_low: float,
    confidence_high: float,
    policy_top_1: float | None = 0.38,
    value_loss: float | None = 0.80,
) -> CheckpointEvaluation:
    return CheckpointEvaluation(
        iteration=iteration,
        score_vs_archive=score,
        score_confidence_low=confidence_low,
        score_confidence_high=confidence_high,
        policy_top_1=policy_top_1,
        value_loss=value_loss,
    )


def test_plateau_requires_three_powered_consecutive_evaluations() -> None:
    evaluations = (
        checkpoint(0, 0.50, 0.46, 0.54),
        checkpoint(5, 0.51, 0.47, 0.55),
        checkpoint(10, 0.50, 0.46, 0.54),
        checkpoint(15, 0.51, 0.47, 0.55),
    )

    decision = evaluate_plateau(evaluations, RULE)

    assert decision.status == PlateauStatus.PLATEAU


def test_clear_score_gain_continues_training() -> None:
    evaluations = (
        checkpoint(0, 0.50, 0.46, 0.54),
        checkpoint(5, 0.51, 0.47, 0.55),
        checkpoint(10, 0.55, 0.53, 0.58),
        checkpoint(15, 0.56, 0.54, 0.59),
    )

    decision = evaluate_plateau(evaluations, RULE)

    assert decision.status == PlateauStatus.CONTINUE


def test_wide_intervals_cannot_declare_plateau() -> None:
    evaluations = (
        checkpoint(0, 0.50, 0.40, 0.60),
        checkpoint(5, 0.51, 0.40, 0.62),
        checkpoint(10, 0.50, 0.39, 0.61),
        checkpoint(15, 0.51, 0.40, 0.62),
    )

    decision = evaluate_plateau(evaluations, RULE)

    assert decision.status == PlateauStatus.INSUFFICIENT_EVIDENCE


def test_plateau_does_not_require_unavailable_fixed_dataset_metrics() -> None:
    score_only_rule = RULE.model_copy(
        update={
            'minimum_policy_top_1_gain': None,
            'minimum_value_loss_reduction': None,
        }
    )
    evaluations = tuple(
        checkpoint(
            iteration,
            0.50,
            0.46,
            0.54,
            policy_top_1=None,
            value_loss=None,
        )
        for iteration in (0, 5, 10, 15)
    )

    decision = evaluate_plateau(evaluations, score_only_rule)

    assert decision.status == PlateauStatus.PLATEAU


@pytest.mark.parametrize(
    ('iterations', 'expected_status'),
    [
        ((0, 5, 10), PlateauStatus.INSUFFICIENT_EVIDENCE),
        ((0, 4, 10, 15), PlateauStatus.INSUFFICIENT_EVIDENCE),
    ],
)
def test_incomplete_windows_do_not_stop(
    iterations: tuple[int, ...],
    expected_status: PlateauStatus,
) -> None:
    evaluations = tuple(checkpoint(iteration, 0.5, 0.46, 0.54) for iteration in iterations)

    decision = evaluate_plateau(evaluations, RULE)

    assert decision.status == expected_status

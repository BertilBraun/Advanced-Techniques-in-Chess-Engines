from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class PlateauStatus(str, Enum):
    CONTINUE = 'continue'
    PLATEAU = 'plateau'
    REGRESSION = 'regression'
    INSUFFICIENT_EVIDENCE = 'insufficient_evidence'


class CheckpointEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    iteration: int = Field(ge=0)
    score_vs_archive: float = Field(ge=0, le=1)
    score_confidence_low: float = Field(ge=0, le=1)
    score_confidence_high: float = Field(ge=0, le=1)
    policy_top_1: float | None = Field(default=None, ge=0, le=1)
    value_loss: float | None = Field(default=None, ge=0)


class PlateauRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    minimum_iteration: int = Field(ge=0)
    evaluation_spacing: int = Field(gt=0)
    consecutive_evaluations: int = Field(ge=3)
    minimum_score_gain: float = Field(gt=0, lt=1)
    maximum_score_interval_width: float = Field(gt=0, lt=1)
    minimum_policy_top_1_gain: float | None = Field(default=None, gt=0, lt=1)
    minimum_value_loss_reduction: float | None = Field(default=None, gt=0)
    regression_score_confidence_high: float = Field(gt=0, lt=0.5)


class PlateauDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    status: PlateauStatus
    reason: str
    evaluated_iterations: tuple[int, ...]


def evaluate_plateau(
    evaluations: tuple[CheckpointEvaluation, ...],
    rule: PlateauRule,
) -> PlateauDecision:
    ordered = tuple(sorted(evaluations, key=lambda evaluation: evaluation.iteration))
    if len(ordered) < rule.consecutive_evaluations + 1:
        return PlateauDecision(
            status=PlateauStatus.INSUFFICIENT_EVIDENCE,
            reason='A pre-window checkpoint and the required consecutive evaluations are not available.',
            evaluated_iterations=tuple(evaluation.iteration for evaluation in ordered),
        )

    window = ordered[-rule.consecutive_evaluations :]
    baseline_candidates = ordered[: -rule.consecutive_evaluations]
    if window[-1].iteration < rule.minimum_iteration:
        return PlateauDecision(
            status=PlateauStatus.INSUFFICIENT_EVIDENCE,
            reason='The minimum continuation iteration has not been reached.',
            evaluated_iterations=tuple(evaluation.iteration for evaluation in window),
        )

    expected_iterations = tuple(
        window[0].iteration + offset * rule.evaluation_spacing for offset in range(rule.consecutive_evaluations)
    )
    actual_iterations = tuple(evaluation.iteration for evaluation in window)
    if actual_iterations != expected_iterations:
        return PlateauDecision(
            status=PlateauStatus.INSUFFICIENT_EVIDENCE,
            reason='The latest evaluations do not have the predeclared spacing.',
            evaluated_iterations=actual_iterations,
        )

    if any(
        evaluation.score_confidence_high - evaluation.score_confidence_low > rule.maximum_score_interval_width
        for evaluation in window
    ):
        return PlateauDecision(
            status=PlateauStatus.INSUFFICIENT_EVIDENCE,
            reason='At least one score interval is wider than the predeclared power threshold.',
            evaluated_iterations=actual_iterations,
        )

    if all(evaluation.score_confidence_high < rule.regression_score_confidence_high for evaluation in window):
        return PlateauDecision(
            status=PlateauStatus.REGRESSION,
            reason='Every checkpoint in the window is clearly below the archived model.',
            evaluated_iterations=actual_iterations,
        )

    score_baseline = max(
        baseline_candidates,
        key=lambda evaluation: evaluation.score_vs_archive,
    )
    latest_pre_window = baseline_candidates[-1]
    score_progress = any(
        evaluation.score_vs_archive >= score_baseline.score_vs_archive + rule.minimum_score_gain
        for evaluation in window
    )
    window_policy_values = tuple(
        evaluation.policy_top_1 for evaluation in window if evaluation.policy_top_1 is not None
    )
    policy_progress = (
        rule.minimum_policy_top_1_gain is not None
        and latest_pre_window.policy_top_1 is not None
        and len(window_policy_values) == len(window)
        and max(window_policy_values) >= latest_pre_window.policy_top_1 + rule.minimum_policy_top_1_gain
    )
    window_value_losses = tuple(evaluation.value_loss for evaluation in window if evaluation.value_loss is not None)
    value_progress = (
        rule.minimum_value_loss_reduction is not None
        and latest_pre_window.value_loss is not None
        and len(window_value_losses) == len(window)
        and min(window_value_losses) <= latest_pre_window.value_loss - rule.minimum_value_loss_reduction
    )

    if score_progress or policy_progress or value_progress:
        return PlateauDecision(
            status=PlateauStatus.CONTINUE,
            reason='At least one predeclared playing-strength or fixed-dataset metric improved.',
            evaluated_iterations=actual_iterations,
        )
    return PlateauDecision(
        status=PlateauStatus.PLATEAU,
        reason='Three powered checkpoint evaluations showed no predeclared meaningful gain.',
        evaluated_iterations=actual_iterations,
    )

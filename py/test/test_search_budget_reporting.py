from __future__ import annotations

import pytest
import src.training.reporting as reporting_module
from src.search_budget.labeling import DistributionSummary
from src.search_budget.manager import (
    CurvePointReport,
    FailedLabelJobReport,
    GenerationLabelReport,
    LabelManagerEvent,
    SkippedLabelJobReport,
)
from src.search_budget.policy import BUDGET_CURVE_MULTIPLES, BUDGET_CURVE_POINTS
from src.training.coordinator import Coordinator
from src.training.reporting import TrainingReporter


def _distribution(mean: float) -> DistributionSummary:
    return DistributionSummary(
        count=10,
        minimum=0.0,
        maximum=1.0,
        mean=mean,
        variance=0.1,
        p10=0.1,
        p25=0.25,
        median=0.5,
        p75=0.75,
        p90=0.9,
        histogram_counts=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    )


def test_completed_deep_label_report_is_fully_published_to_tensorboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scalars: dict[str, tuple[float, int | None]] = {}
    histograms: dict[str, tuple[int, int | None]] = {}

    def record_scalar(name: str, value: float, step: int | None = None) -> None:
        scalars[name] = (value, step)

    def record_histogram(
        name: str,
        minimum: float,
        maximum: float,
        count: int,
        mean: float,
        variance: float,
        bucket_counts: tuple[int, ...],
        step: int | None = None,
    ) -> None:
        del minimum, maximum, mean, variance, bucket_counts
        histograms[name] = (count, step)

    monkeypatch.setattr(reporting_module, 'log_scalar', record_scalar)
    monkeypatch.setattr(reporting_module, 'log_equal_width_histogram_summary', record_histogram)
    report = GenerationLabelReport(
        source_generation=12,
        model_generation=12,
        inference_model_sha256='1' * 64,
        population_position_count=500,
        selected_position_count=10,
        baseline_raw_kl_distribution=_distribution(0.2),
        predicted_baseline_log_kl_distribution=_distribution(0.45),
        target_baseline_log_kl_distribution=_distribution(0.5),
        curve_points=tuple(
            CurvePointReport(
                curve_index=index,
                multiple=BUDGET_CURVE_MULTIPLES[index],
                grid_visits=75 + index,
                sigma=1.0 + 0.1 * index,
                mean_target_log_kl=-2.0 - 0.1 * index,
                mean_predicted_log_kl=-1.9 - 0.1 * index,
                mean_absolute_error=0.1 * (index + 1),
                selected_count=1,
            )
            for index in range(BUDGET_CURVE_POINTS)
        ),
        replay_samples_written=10,
        replay_write_applied=True,
        prediction_shard_seconds=1.5,
        deep_search_shard_seconds=9.0,
        total_gpu_seconds=10.5,
        prediction_retry_count=1,
        deep_search_retry_count=2,
        completion_generation_lag=1,
        queued_generation_count=2,
        current_validation_gain=0.03,
        ema_validation_gain=0.02,
        lagrange_multiplier=0.4,
        corrector_applied=True,
        corrector_sha256='2' * 64,
        realized_mean_multiple=1.05,
        realized_mean_assigned_visits=630.0,
        flat_mean_assigned_visits=600.0,
        assigned_new_visits_variance=25.0,
        selected_index_counts=(1,) * BUDGET_CURVE_POINTS,
        published_apply_learned=False,
        failed_eligibility_conditions=('warmup',),
        application_generation=14,
        decision_reason='warmup',
    )

    TrainingReporter.record_search_budget_label_event(report)

    assert scalars['search_budget/label/status/completed'] == (1, 12)
    assert scalars['search_budget/label/sample_fraction'] == pytest.approx((0.02, 12))
    assert scalars['search_budget/label/baseline_raw_kl/mean'] == (0.2, 12)
    assert scalars['search_budget/label/predicted_baseline_log_kl/mean'] == (0.45, 12)
    assert scalars['search_budget/label/target_baseline_log_kl/median'] == (0.5, 12)
    assert scalars['search_budget/label/deep_search_retries'] == (2, 12)
    assert scalars['search_budget/calibration/lagrange_multiplier'] == (0.4, 12)
    assert scalars['search_budget/calibration/corrector_applied'] == (1, 12)
    assert scalars['search_budget/calibration/realized_mean_multiple'] == (1.05, 12)
    assert scalars['search_budget/calibration/realized_mean_assigned_visits'] == (630.0, 12)
    assert scalars['search_budget/calibration/flat_mean_assigned_visits'] == (600.0, 12)
    assert scalars['search_budget/calibration/assigned_new_visits_variance'] == (25.0, 12)
    assert scalars['search_budget/calibration/published_apply_learned'] == (0, 12)
    assert scalars['search_budget/calibration/current_validation_gain'] == (0.03, 12)
    assert scalars['search_budget/calibration/ema_validation_gain'] == (0.02, 12)
    assert scalars['search_budget/calibration/application_generation'] == (14, 12)
    assert scalars['search_budget/calibration/decision_reason/warmup'] == (1, 12)
    assert scalars['search_budget/calibration/selected_index_7'] == (1, 12)
    prefix = 'search_budget/curve_point_7'
    assert scalars[f'{prefix}/sigma'] == pytest.approx((1.7, 12))
    assert scalars[f'{prefix}/grid_visits'] == (82, 12)
    assert scalars[f'{prefix}/mean_target_log_kl'] == pytest.approx((-2.7, 12))
    assert scalars[f'{prefix}/mean_predicted_log_kl'] == pytest.approx((-2.6, 12))
    assert scalars[f'{prefix}/mean_absolute_error'] == pytest.approx((0.8, 12))
    assert scalars[f'{prefix}/selected_count'] == (1, 12)
    assert scalars['search_budget/calibration/failed_eligibility/warmup'] == (1, 12)
    assert histograms == {
        'search_budget/label/baseline_raw_kl': (10, 12),
        'search_budget/label/predicted_baseline_log_kl': (10, 12),
        'search_budget/label/target_baseline_log_kl': (10, 12),
    }


def test_failed_and_skipped_label_reports_publish_health_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    scalars: dict[str, tuple[float, int | None]] = {}

    def record_scalar(name: str, value: float, step: int | None = None) -> None:
        scalars[name] = (value, step)

    monkeypatch.setattr(reporting_module, 'log_scalar', record_scalar)

    TrainingReporter.record_search_budget_label_event(
        FailedLabelJobReport(
            source_generation=4,
            failure='RuntimeError: failed shard',
            published_apply_learned=False,
            application_generation=6,
            decision_reason='terminal_failure',
        )
    )
    TrainingReporter.record_search_budget_label_event(
        SkippedLabelJobReport(
            source_generation=5,
            population_position_count=40,
            selected_position_count=0,
            reason='generation population produces zero positions at the configured sample fraction',
        )
    )

    assert scalars['search_budget/label/status/failed'] == (1, 4)
    assert scalars['search_budget/calibration/published_apply_learned'] == (0, 4)
    assert scalars['search_budget/calibration/decision_reason/terminal_failure'] == (1, 4)
    assert scalars['search_budget/label/status/skipped'] == (1, 5)
    assert scalars['search_budget/label/sample_fraction'] == (0.0, 5)
    assert scalars['search_budget/label/skip_reason/zero_position_sample'] == (1, 5)


def test_coordinator_forwards_every_label_manager_event_to_the_reporter() -> None:
    event = SkippedLabelJobReport(
        source_generation=5,
        population_position_count=40,
        selected_position_count=0,
        reason='generation population produces zero positions at the configured sample fraction',
    )

    class LabelManager:
        def poll(self) -> tuple[LabelManagerEvent, ...]:
            return (event,)

    class Reporter:
        def __init__(self) -> None:
            self.events: list[LabelManagerEvent] = []

        def record_search_budget_label_event(self, reported: LabelManagerEvent) -> None:
            self.events.append(reported)

    coordinator = Coordinator.__new__(Coordinator)
    reporter = Reporter()
    coordinator.search_budget_label_manager = LabelManager()  # type: ignore[assignment]
    coordinator.reporter = reporter  # type: ignore[assignment]

    coordinator._collect_search_budget_label_jobs()

    assert reporter.events == [event]

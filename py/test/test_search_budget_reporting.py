from __future__ import annotations

import pytest
import src.training.reporting as reporting_module
from src.search_budget.labeling import DistributionSummary
from src.search_budget.manager import (
    BucketFinalizationReport,
    FailedLabelJobReport,
    GenerationLabelReport,
    LabelManagerEvent,
    SkippedLabelJobReport,
)
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

    def record_scalar(name: str, value: float, step: int | None = None) -> None:
        scalars[name] = (value, step)

    monkeypatch.setattr(reporting_module, 'log_scalar', record_scalar)
    report = GenerationLabelReport(
        source_generation=12,
        model_generation=12,
        inference_model_sha256='1' * 64,
        population_position_count=500,
        selected_position_count=10,
        prediction_distribution=_distribution(0.45),
        target_distribution=_distribution(0.5),
        raw_kl_distribution=_distribution(0.2),
        buckets=tuple(
            BucketFinalizationReport(
                bucket_index=index,
                sample_count=1,
                current_generation_utility=0.01 * index,
                ema_utility=0.02 * index,
                shadow_multiplier=1.0,
                pending_multiplier=1.0,
                published_multiplier=1.0,
                raw_log_update=0.001 * index,
                projection_adjustment=0.0,
                lower_mean_visits=550.0,
                upper_mean_visits=650.0,
                checkpoint_deduplication_count=0,
            )
            for index in range(10)
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
        candidate_mean_assigned_new_visits=600.0,
        candidate_assigned_new_visits_variance=25.0,
        candidate_mean_kl_from_deep=0.15,
        candidate_exact_spend_residual=0,
        previous_published_curve=(1.0,) * 10,
        validated_curve=(1.0,) * 10,
        shadow_curve=(1.0,) * 10,
        pending_curve=(1.0,) * 10,
        published_curve=(1.0,) * 10,
        minimum_published_multiplier=1.0,
        maximum_published_multiplier=1.0,
        failed_eligibility_conditions=('warmup',),
        application_generation=14,
        decision_reason='warmup',
    )

    TrainingReporter.record_search_budget_label_event(report)

    assert scalars['search_budget/label/status/completed'] == (1, 12)
    assert scalars['search_budget/label/sample_fraction'] == pytest.approx((0.02, 12))
    assert scalars['search_budget/label/raw_kl/mean'] == (0.2, 12)
    assert scalars['search_budget/label/target_quantile/median'] == (0.5, 12)
    assert scalars['search_budget/label/prediction_quantile/histogram_bin_9'] == (1, 12)
    assert scalars['search_budget/label/deep_search_retries'] == (2, 12)
    assert scalars['search_budget/calibration/published_mean_multiplier'] == (1.0, 12)
    assert scalars['search_budget/calibration/current_validation_gain'] == (0.03, 12)
    assert scalars['search_budget/calibration/ema_validation_gain'] == (0.02, 12)
    assert scalars['search_budget/calibration/candidate_exact_spend_residual'] == (0, 12)
    assert scalars['search_budget/calibration/application_generation'] == (14, 12)
    assert scalars['search_budget/calibration/decision_reason/warmup'] == (1, 12)
    prefix = 'search_budget/calibration/bucket_9'
    assert scalars[f'{prefix}/generation_marginal_utility'] == (0.09, 12)
    assert scalars[f'{prefix}/ema_marginal_utility'] == (0.18, 12)
    assert scalars[f'{prefix}/validated_multiplier'] == (1.0, 12)
    assert scalars[f'{prefix}/published_multiplier'] == (1.0, 12)
    assert scalars['search_budget/calibration/failed_eligibility/warmup'] == (1, 12)


def test_failed_and_skipped_label_reports_publish_health_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    scalars: dict[str, tuple[float, int | None]] = {}

    def record_scalar(name: str, value: float, step: int | None = None) -> None:
        scalars[name] = (value, step)

    monkeypatch.setattr(reporting_module, 'log_scalar', record_scalar)

    TrainingReporter.record_search_budget_label_event(
        FailedLabelJobReport(
            source_generation=4,
            failure='RuntimeError: failed shard',
            published_curve=(1.0,) * 10,
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

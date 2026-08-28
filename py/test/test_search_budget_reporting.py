from __future__ import annotations

from decimal import Decimal

import pytest
import src.training.reporting as reporting_module
from src.search_budget.labeling import DistributionSummary
from src.search_budget.manager import (
    CandidateFinalizationReport,
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
        candidates=(
            CandidateFinalizationReport(
                blend=Decimal('0.1'),
                current_generation_gain=0.03,
                ema_gain=0.02,
                mean_assigned_new_visits=600.0,
                assigned_new_visits_variance=25.0,
                mean_kl_from_deep=0.18,
                exact_spend_residual=0,
                floor_share=0.4,
                ceiling_share=0.05,
                failed_eligibility_conditions=('warmup',),
            ),
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
        previous_blend=Decimal('0.0'),
        selected_blend=Decimal('0.1'),
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
    assert scalars['search_budget/calibration/published_blend'] == (0.1, 12)
    assert scalars['search_budget/calibration/application_generation'] == (14, 12)
    assert scalars['search_budget/calibration/decision_reason/warmup'] == (1, 12)
    prefix = 'search_budget/calibration/candidate_0_1'
    assert scalars[f'{prefix}/current_generation_gain'] == (0.03, 12)
    assert scalars[f'{prefix}/ema_gain'] == (0.02, 12)
    assert scalars[f'{prefix}/exact_spend_residual'] == (0, 12)
    assert scalars[f'{prefix}/eligible'] == (0, 12)
    assert scalars[f'{prefix}/failed_eligibility/warmup'] == (1, 12)


def test_failed_and_skipped_label_reports_publish_health_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    scalars: dict[str, tuple[float, int | None]] = {}

    def record_scalar(name: str, value: float, step: int | None = None) -> None:
        scalars[name] = (value, step)

    monkeypatch.setattr(reporting_module, 'log_scalar', record_scalar)

    TrainingReporter.record_search_budget_label_event(
        FailedLabelJobReport(
            source_generation=4,
            failure='RuntimeError: failed shard',
            published_blend=Decimal('0.0'),
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

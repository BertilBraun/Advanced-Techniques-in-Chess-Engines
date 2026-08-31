from __future__ import annotations

import pytest
import src.training.reporting as reporting_module
from src.experiment.configuration import load_experiment_configuration
from src.search_budget.policy import BUDGET_CURVE_POINTS
from src.training.reporting import TrainingReporter
from src.training.search_budget_tensorboard import search_budget_tensorboard_categories
from src.training.targets import LegalMovesHeadLayout, SearchBudgetHeadLayout
from src.util.tensorboard import TensorboardCustomScalarCategory
from test_helpers.configuration_paths import REPOSITORY_CONFIG_DIRECTORY


def test_search_budget_dashboard_is_compact_and_uses_dynamic_auxiliary_index() -> None:
    categories = search_budget_tensorboard_categories(
        (
            LegalMovesHeadLayout(kind='legal_moves', action_size=64),
            SearchBudgetHeadLayout(kind='search_budget'),
        )
    )

    assert tuple(category.title for category in categories) == (
        'Adaptive Search Overview',
        'Adaptive Search - Curve Head Diagnostics',
        'Adaptive Search - Policy Calibration',
        'Adaptive Search - Label Pipeline',
        'Adaptive Search - Production Diagnostics',
    )
    overview = categories[0]
    assert tuple(chart.title for chart in overview.charts) == (
        'Validation gain (nats)',
        'Curve-head learning',
        'Spend tracking',
        'Realized mean multiple',
        'Dual variable',
        'Corrector applied',
        'Gate',
        'Exact spend residual',
        'Label job outcome',
        'Label and replay volume',
    )
    assert overview.charts[1].tags == (
        'training_auxiliary/1-search-budget/loss',
        'training_auxiliary/1-search-budget/absolute_error_mean',
    )
    calibration = categories[2]
    sigma = next(chart for chart in calibration.charts if chart.title == 'Sigma by grid point')
    assert sigma.tags == tuple(f'search_budget/curve_point_{index}/sigma' for index in range(BUDGET_CURVE_POINTS))


def test_search_budget_dashboard_is_absent_without_search_budget_head() -> None:
    assert search_budget_tensorboard_categories((LegalMovesHeadLayout(kind='legal_moves', action_size=64),)) == ()


def test_training_reporter_publishes_adaptive_overview_with_evaluation_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = load_experiment_configuration(
        REPOSITORY_CONFIG_DIRECTORY / 'validation' / 'vast-chess-4day-production-v10.yaml'
    )
    published_categories: list[tuple[str, ...]] = []

    def record_layout(categories: tuple[TensorboardCustomScalarCategory, ...]) -> None:
        published_categories.append(tuple(category.title for category in categories))

    def skip_scheduled_settings(reporter: TrainingReporter, generation: int) -> None:
        del reporter, generation

    monkeypatch.setattr(reporting_module, 'log_custom_scalar_layout', record_layout)
    monkeypatch.setattr(TrainingReporter, '_record_scheduled_settings', skip_scheduled_settings)
    reporter = TrainingReporter(
        configuration,
        (
            LegalMovesHeadLayout(kind='legal_moves', action_size=64),
            SearchBudgetHeadLayout(kind='search_budget'),
        ),
    )

    reporter.record_initial_settings(0)

    assert len(published_categories) == 1
    titles = published_categories[0]
    assert titles[0] == 'Adaptive Search Overview'
    assert 'Evaluation matches' in titles

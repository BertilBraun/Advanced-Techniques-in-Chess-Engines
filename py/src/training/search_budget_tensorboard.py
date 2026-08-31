from __future__ import annotations

from src.search_budget.calibration import BudgetDecisionReason, BudgetEligibilityFailure
from src.search_budget.policy import BUDGET_CURVE_POINTS
from src.training.targets import AuxiliaryHeadLayout, search_budget_auxiliary_index
from src.util.tensorboard import TensorboardCustomScalarCategory, TensorboardMultilineChart


def search_budget_tensorboard_categories(
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...],
) -> tuple[TensorboardCustomScalarCategory, ...]:
    auxiliary_prefix = _search_budget_auxiliary_prefix(auxiliary_heads)
    if auxiliary_prefix is None:
        return ()
    return (
        _overview_category(auxiliary_prefix),
        _curve_head_category(auxiliary_prefix),
        _calibration_category(),
        _label_pipeline_category(),
        _production_category(),
    )


def _overview_category(auxiliary_prefix: str) -> TensorboardCustomScalarCategory:
    return TensorboardCustomScalarCategory(
        title='Adaptive Search Overview',
        charts=(
            _chart(
                'Validation gain (nats)',
                'search_budget/calibration/current_validation_gain',
                'search_budget/calibration/ema_validation_gain',
            ),
            _chart(
                'Curve-head learning',
                f'{auxiliary_prefix}/loss',
                f'{auxiliary_prefix}/absolute_error_mean',
            ),
            _chart(
                'Spend tracking',
                'search_budget/calibration/realized_mean_assigned_visits',
                'search_budget/calibration/flat_mean_assigned_visits',
                'search_budget/production/baseline_visits_mean',
                'search_budget/production/assigned_additional_visits_mean',
            ),
            _chart('Realized mean multiple', 'search_budget/calibration/realized_mean_multiple'),
            _chart('Dual variable', 'search_budget/calibration/lagrange_multiplier'),
            _chart('Corrector applied', 'search_budget/calibration/corrector_applied'),
            _chart('Gate', 'search_budget/calibration/published_apply_learned'),
            _chart(
                'Exact spend residual',
                'search_budget/production/exact_generation_spend_residual',
            ),
            _chart(
                'Label job outcome',
                'search_budget/label/status/completed',
                'search_budget/label/status/failed',
                'search_budget/label/status/skipped',
                'search_budget/label/replay_write_applied',
            ),
            _chart(
                'Label and replay volume',
                'search_budget/label/selected_positions',
                'search_budget/label/replay_samples_written',
            ),
        ),
    )


def _curve_head_category(auxiliary_prefix: str) -> TensorboardCustomScalarCategory:
    return TensorboardCustomScalarCategory(
        title='Adaptive Search - Curve Head Diagnostics',
        charts=(
            _chart('Training loss and error', f'{auxiliary_prefix}/loss', f'{auxiliary_prefix}/absolute_error_mean'),
            _chart(
                'Training target and prediction mean',
                f'{auxiliary_prefix}/target_mean',
                f'{auxiliary_prefix}/prediction_mean',
            ),
            _chart(
                'Head-batch target and prediction mean',
                'search_budget/head_batch/target_mean',
                'search_budget/head_batch/prediction_mean',
            ),
            _chart(
                'Head-batch spread',
                'search_budget/head_batch/target_standard_deviation',
                'search_budget/head_batch/prediction_standard_deviation',
            ),
            _chart(
                'Head-batch error',
                'search_budget/head_batch/loss',
                'search_budget/head_batch/absolute_error_mean',
            ),
            _chart(
                'Head-batch supply',
                'search_budget/head_batch/labelled_pool_rows',
                'search_budget/head_batch/labelled_batches',
            ),
            _chart('Head-batch skipped', 'search_budget/head_batch/skipped'),
            _curve_point_chart('Curve residual by grid point', 'mean_absolute_error'),
            _curve_point_chart('Target log KL by grid point', 'mean_target_log_kl'),
            _curve_point_chart('Predicted log KL by grid point', 'mean_predicted_log_kl'),
            _chart(
                'Baseline log KL mean',
                'search_budget/label/predicted_baseline_log_kl/mean',
                'search_budget/label/target_baseline_log_kl/mean',
            ),
            _chart(
                'Baseline log KL variance',
                'search_budget/label/predicted_baseline_log_kl/variance',
                'search_budget/label/target_baseline_log_kl/variance',
            ),
            _chart(
                'Baseline raw KL summary',
                'search_budget/label/baseline_raw_kl/mean',
                'search_budget/label/baseline_raw_kl/median',
                'search_budget/label/baseline_raw_kl/p90',
            ),
        ),
    )


def _calibration_category() -> TensorboardCustomScalarCategory:
    return TensorboardCustomScalarCategory(
        title='Adaptive Search - Policy Calibration',
        charts=(
            _chart(
                'Validation gain (nats)',
                'search_budget/calibration/current_validation_gain',
                'search_budget/calibration/ema_validation_gain',
            ),
            _curve_point_chart('Sigma by grid point', 'sigma'),
            _curve_point_chart('Selected count by grid point', 'selected_count'),
            _chart('Dual variable', 'search_budget/calibration/lagrange_multiplier'),
            _chart('Corrector applied', 'search_budget/calibration/corrector_applied'),
            _chart(
                'Spend tracking',
                'search_budget/calibration/realized_mean_assigned_visits',
                'search_budget/calibration/flat_mean_assigned_visits',
            ),
            _chart('Realized mean multiple', 'search_budget/calibration/realized_mean_multiple'),
            _chart(
                'Assigned-visit variance',
                'search_budget/calibration/assigned_new_visits_variance',
            ),
            _chart(
                'Shadow selected-index histogram',
                *tuple(f'search_budget/calibration/selected_index_{index}' for index in range(BUDGET_CURVE_POINTS)),
            ),
            _chart(
                'Publication decision',
                *tuple(f'search_budget/calibration/decision_reason/{reason.value}' for reason in BudgetDecisionReason),
            ),
            _chart(
                'Failed eligibility conditions',
                *tuple(
                    f'search_budget/calibration/failed_eligibility/{failure.value}'
                    for failure in BudgetEligibilityFailure
                ),
            ),
        ),
    )


def _label_pipeline_category() -> TensorboardCustomScalarCategory:
    return TensorboardCustomScalarCategory(
        title='Adaptive Search - Label Pipeline',
        charts=(
            _chart(
                'Job outcome',
                'search_budget/label/status/completed',
                'search_budget/label/status/failed',
                'search_budget/label/status/skipped',
            ),
            _chart(
                'Population and sample',
                'search_budget/label/population_positions',
                'search_budget/label/selected_positions',
            ),
            _chart('Sample fraction', 'search_budget/label/sample_fraction'),
            _chart(
                'Replay write-back',
                'search_budget/label/replay_samples_written',
                'search_budget/label/replay_write_applied',
            ),
            _chart(
                'Shard retries',
                'search_budget/label/prediction_retries',
                'search_budget/label/deep_search_retries',
            ),
            _chart(
                'Phase duration (seconds)',
                'search_budget/label/prediction_shard_seconds',
                'search_budget/label/deep_search_shard_seconds',
                'search_budget/label/total_gpu_seconds',
            ),
            _chart(
                'Generation lag and queue',
                'search_budget/label/completion_generation_lag',
                'search_budget/label/queued_generations',
            ),
        ),
    )


def _production_category() -> TensorboardCustomScalarCategory:
    return TensorboardCustomScalarCategory(
        title='Adaptive Search - Production Diagnostics',
        charts=(
            _chart(
                'Visit budget',
                'search_budget/production/baseline_visits_mean',
                'search_budget/production/assigned_additional_visits_mean',
                'search_budget/production/final_visits_mean',
            ),
            _chart(
                'Predicted baseline log KL',
                'search_budget/production/predicted_baseline_log_kl_mean',
                'search_budget/label/predicted_baseline_log_kl/mean',
            ),
            _chart(
                'Selected-index histogram',
                *tuple(f'search_budget/production/selected_index_{index}' for index in range(BUDGET_CURVE_POINTS)),
            ),
            _chart('Mean selected index', 'search_budget/production/selected_index_mean'),
            _chart('Parallel searches', 'search_budget/production/parallel_searches_mean'),
            _chart(
                'Residual accounting',
                'search_budget/production/exact_generation_spend_residual',
            ),
            _chart(
                'Retained-root visits',
                'search_budget/production/starting_visits_mean',
                'search_budget/production/final_visits_mean',
            ),
            _chart(
                'Search corrections',
                'search_budget/production/policy_correction_mean',
                'search_budget/production/value_correction_mean',
            ),
        ),
    )


def _curve_point_chart(title: str, metric: str) -> TensorboardMultilineChart:
    return _chart(
        title,
        *tuple(f'search_budget/curve_point_{index}/{metric}' for index in range(BUDGET_CURVE_POINTS)),
    )


def _chart(title: str, *tags: str) -> TensorboardMultilineChart:
    return TensorboardMultilineChart(title=title, tags=tags)


def _search_budget_auxiliary_prefix(auxiliary_heads: tuple[AuxiliaryHeadLayout, ...]) -> str | None:
    index = search_budget_auxiliary_index(auxiliary_heads)
    return None if index is None else f'training_auxiliary/{index}-search-budget'

from __future__ import annotations

from src.search_budget.calibration import CurveDecisionReason, CurveEligibilityFailure
from src.search_budget.curve import CURVE_BUCKET_COUNT
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
        _scalar_head_category(auxiliary_prefix),
        _curve_category(),
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
                'Relative validation gain (%)',
                'search_budget/calibration/current_relative_validation_gain_percent',
                'search_budget/calibration/ema_relative_validation_gain_percent',
            ),
            _chart(
                'Scalar-head learning',
                f'{auxiliary_prefix}/loss',
                f'{auxiliary_prefix}/absolute_error_mean',
            ),
            _chart(
                'Label quantile mean',
                'search_budget/label/prediction_quantile/mean',
                'search_budget/label/target_quantile/mean',
            ),
            _chart(
                'Label quantile variance',
                'search_budget/label/prediction_quantile/variance',
                'search_budget/label/target_quantile/variance',
            ),
            _chart(
                'Visit allocation',
                'search_budget/production/baseline_visits_mean',
                'search_budget/production/assigned_additional_visits_mean',
                'search_budget/production/final_visits_mean',
                'search_budget/calibration/candidate_mean_assigned_new_visits',
            ),
            _chart(
                'Exact spend residual',
                'search_budget/production/exact_generation_spend_residual',
                'search_budget/calibration/candidate_exact_spend_residual',
            ),
            _chart(
                'Curve multiplier range',
                'search_budget/calibration/minimum_published_multiplier',
                'search_budget/calibration/maximum_published_multiplier',
                'search_budget/calibration/minimum_pending_multiplier',
                'search_budget/calibration/maximum_pending_multiplier',
                'search_budget/calibration/minimum_shadow_multiplier',
                'search_budget/calibration/maximum_shadow_multiplier',
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


def _scalar_head_category(auxiliary_prefix: str) -> TensorboardCustomScalarCategory:
    return TensorboardCustomScalarCategory(
        title='Adaptive Search - Scalar Head Diagnostics',
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
            _chart(
                'Head-batch skipped',
                'search_budget/head_batch/skipped',
            ),
            _chart(
                'Generation-wide quantile mean',
                'search_budget/label/target_quantile/mean',
                'search_budget/label/prediction_quantile/mean',
            ),
            _chart(
                'Generation-wide quantile variance',
                'search_budget/label/target_quantile/variance',
                'search_budget/label/prediction_quantile/variance',
            ),
            _chart(
                'Prediction quantile range',
                'search_budget/label/prediction_quantile/minimum',
                'search_budget/label/prediction_quantile/maximum',
            ),
            _chart(
                'Prediction quantiles',
                'search_budget/label/prediction_quantile/p10',
                'search_budget/label/prediction_quantile/p25',
                'search_budget/label/prediction_quantile/median',
                'search_budget/label/prediction_quantile/p75',
                'search_budget/label/prediction_quantile/p90',
            ),
            _chart(
                'Prediction histogram bins',
                *tuple(f'search_budget/label/prediction_quantile/histogram_bin_{index}' for index in range(10)),
            ),
            _chart(
                'Target histogram bins',
                *tuple(f'search_budget/label/target_quantile/histogram_bin_{index}' for index in range(10)),
            ),
            _chart(
                'Raw KL summary',
                'search_budget/label/raw_kl/mean',
                'search_budget/label/raw_kl/median',
                'search_budget/label/raw_kl/p90',
            ),
        ),
    )


def _curve_category() -> TensorboardCustomScalarCategory:
    return TensorboardCustomScalarCategory(
        title='Adaptive Search - Live Curve Calibration',
        charts=(
            _chart(
                'Validation gain (nats)',
                'search_budget/calibration/current_validation_gain',
                'search_budget/calibration/ema_validation_gain',
            ),
            _chart(
                'Relative validation gain (%)',
                'search_budget/calibration/current_relative_validation_gain_percent',
                'search_budget/calibration/ema_relative_validation_gain_percent',
            ),
            _bucket_chart('Published curve', 'published_multiplier'),
            _bucket_chart('Pending curve', 'pending_multiplier'),
            _bucket_chart('Shadow curve', 'shadow_multiplier'),
            _bucket_chart('Validated curve', 'validated_multiplier'),
            _bucket_chart('Generation marginal utility', 'generation_marginal_utility'),
            _bucket_chart('EMA marginal utility', 'ema_marginal_utility'),
            _bucket_chart('Bucket sample counts', 'sample_count'),
            _bucket_chart('Raw log updates', 'raw_log_update'),
            _bucket_chart('Projection adjustments', 'projection_adjustment'),
            _bucket_chart('Lower probe visits', 'lower_mean_visits'),
            _bucket_chart('Upper probe visits', 'upper_mean_visits'),
            _chart(
                'Publication decision',
                *tuple(f'search_budget/calibration/decision_reason/{reason.value}' for reason in CurveDecisionReason),
            ),
            _chart(
                'Failed eligibility conditions',
                *tuple(
                    f'search_budget/calibration/failed_eligibility/{failure.value}'
                    for failure in CurveEligibilityFailure
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
                'Predicted quantile',
                'search_budget/production/predicted_quantile_mean',
                'search_budget/label/prediction_quantile/mean',
            ),
            _chart('Parallel searches', 'search_budget/production/parallel_searches_mean'),
            _chart(
                'Residual accounting',
                'search_budget/production/exact_generation_spend_residual',
                'search_budget/calibration/candidate_exact_spend_residual',
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


def _bucket_chart(title: str, metric: str) -> TensorboardMultilineChart:
    return _chart(
        title,
        *tuple(f'search_budget/calibration/bucket_{index}/{metric}' for index in range(CURVE_BUCKET_COUNT)),
    )


def _chart(title: str, *tags: str) -> TensorboardMultilineChart:
    return TensorboardMultilineChart(title=title, tags=tags)


def _search_budget_auxiliary_prefix(auxiliary_heads: tuple[AuxiliaryHeadLayout, ...]) -> str | None:
    index = search_budget_auxiliary_index(auxiliary_heads)
    return None if index is None else f'training_auxiliary/{index}-search-budget'

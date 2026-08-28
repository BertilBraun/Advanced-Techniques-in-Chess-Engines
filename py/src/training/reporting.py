from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from src.evaluation.tensorboard import evaluation_tensorboard_categories
from src.experiment.configuration import ExperimentConfiguration
from src.replay.description import ReplayDescription
from src.replay.manager import IngestedCompletedGame
from src.search_budget.labeling import DistributionSummary
from src.search_budget.manager import (
    FailedLabelJobReport,
    GenerationLabelReport,
    LabelManagerEvent,
    SkippedLabelJobReport,
)
from src.self_play.resignation import ResignationDiagnostics
from src.training.credit_ledger import CreditLedgerState
from src.training.distributions import (
    LegalMovesTrainingDistribution,
    NextPolicyTrainingDistribution,
    PolicyTrainingDistribution,
    RemainingGameLengthTrainingDistribution,
    ScalarAuxiliaryTrainingDistribution,
    TrainingDistributionSnapshot,
)
from src.training.search_budget_tensorboard import search_budget_tensorboard_categories
from src.training.session import (
    FixedTrainingSessionResult,
    ModelTrainingResult,
    ProgressiveTrainingSessionResult,
    TrainingPublication,
    TrainingSessionResult,
)
from src.training.targets import (
    AuxiliaryHeadLayout,
    FutureSearchValueHeadLayout,
    IrreversibleProgressHeadLayout,
    LegalMovesHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    SearchBudgetHeadLayout,
)
from src.training.telemetry import (
    completed_game_length_telemetry,
    search_budget_telemetry,
    training_lifecycle_telemetry,
)
from src.training.tensorboard import scheduled_settings_at
from src.training.trainer import TrainingStatistics
from src.util.log import log
from src.util.tensorboard import log_custom_scalar_layout, log_equal_width_histogram_summary, log_histogram, log_scalar


@dataclass(frozen=True)
class ReplayIngestionTelemetry:
    ingest_seconds: float
    inbox_depth: int
    staging_depth: int
    materialization_failures: int
    rejection_rate: float


class TrainingReporter:
    def __init__(
        self,
        configuration: ExperimentConfiguration,
        auxiliary_heads: tuple[AuxiliaryHeadLayout, ...],
    ) -> None:
        self.configuration = configuration
        self.auxiliary_heads = auxiliary_heads

    def record_initial_settings(self, generation: int) -> None:
        log_custom_scalar_layout(
            (
                *search_budget_tensorboard_categories(self.auxiliary_heads),
                *evaluation_tensorboard_categories(self.configuration.evaluation),
            )
        )
        self._record_scheduled_settings(generation)

    def record_training_outcome(
        self,
        outcome: TrainingSessionResult,
        credit_wait_seconds: float,
        ledger_state: CreditLedgerState,
        replay: ReplayDescription,
        completed_games: tuple[IngestedCompletedGame, ...],
        ingestion: ReplayIngestionTelemetry,
    ) -> None:
        match outcome:
            case FixedTrainingSessionResult(publication=publication, statistics=statistics):
                self._record_training_statistics(
                    publication,
                    statistics,
                    credit_wait_seconds,
                    ledger_state,
                    replay,
                )
            case ProgressiveTrainingSessionResult(
                publication=publication,
                active_model_id=active_model_id,
                active_model_index=active_model_index,
                model_results=model_results,
            ):
                active_result = next(
                    (item.result for item in model_results if item.model_id == active_model_id),
                    None,
                )
                if active_result is not None:
                    self._record_training_statistics(
                        publication,
                        active_result.statistics,
                        credit_wait_seconds,
                        ledger_state,
                        replay,
                    )
                log_scalar(
                    'progressive/active_model_index',
                    active_model_index,
                    publication.checkpoint.generation,
                )
                self._record_progressive_model_statistics(model_results, publication.checkpoint.generation)
                log(
                    f'Published progressive model {active_model_id} at generation '
                    f'{publication.checkpoint.generation} after {credit_wait_seconds:.1f}s of credit wait.'
                )
        self._record_ingestion(ingestion, outcome.publication.checkpoint.generation)
        self._record_completed_game_lengths(completed_games, outcome.publication.checkpoint.generation)
        self._record_search_budget(completed_games, outcome.publication.checkpoint.generation)
        self._record_scheduled_settings(outcome.publication.checkpoint.generation)

    @staticmethod
    def _record_ingestion(ingestion: ReplayIngestionTelemetry, generation: int) -> None:
        log_scalar('replay/ingest_seconds', ingestion.ingest_seconds, generation)
        log_scalar('replay/inbox_depth', ingestion.inbox_depth, generation)
        log_scalar('replay/staging_depth', ingestion.staging_depth, generation)
        log_scalar('replay/materialization_failures', ingestion.materialization_failures, generation)
        log_scalar('replay/rejection_rate', ingestion.rejection_rate, generation)

    def record_resignation(self, diagnostics: ResignationDiagnostics, generation: int) -> None:
        if diagnostics.selected_threshold is not None:
            log_scalar('resignation/selected_threshold', diagnostics.selected_threshold, generation)
        log_scalar('resignation/selected_threshold_safe', int(diagnostics.selected_threshold_safe), generation)
        log_scalar('resignation/continuation_games', diagnostics.continuation_games, generation)
        log_scalar('resignation/audit_triggers', diagnostics.audit_triggers, generation)
        log_scalar('resignation/false_nonlosses', diagnostics.false_nonlosses, generation)
        log_scalar('resignation/false_nonloss_rate', diagnostics.false_nonloss_rate, generation)
        log_scalar('resignation/false_nonloss_upper_bound', diagnostics.false_nonloss_upper_bound, generation)
        log_scalar('resignation/actual_resignations', diagnostics.actual_resignations, generation)
        if diagnostics.average_trigger_ply is not None:
            log_scalar('resignation/average_trigger_ply', diagnostics.average_trigger_ply, generation)
        if diagnostics.average_saved_plies is not None:
            log_scalar('resignation/average_saved_plies', diagnostics.average_saved_plies, generation)

    @staticmethod
    def record_search_budget_label_event(event: LabelManagerEvent) -> None:
        generation = event.source_generation
        match event:
            case GenerationLabelReport():
                log_scalar('search_budget/label/status/completed', 1, generation)
                log_scalar('search_budget/label/model_generation', event.model_generation, generation)
                log_scalar('search_budget/label/population_positions', event.population_position_count, generation)
                log_scalar('search_budget/label/selected_positions', event.selected_position_count, generation)
                log_scalar(
                    'search_budget/label/sample_fraction',
                    event.selected_position_count / event.population_position_count,
                    generation,
                )
                log_scalar('search_budget/label/replay_samples_written', event.replay_samples_written, generation)
                log_scalar('search_budget/label/replay_write_applied', int(event.replay_write_applied), generation)
                log_scalar('search_budget/label/prediction_shard_seconds', event.prediction_shard_seconds, generation)
                log_scalar('search_budget/label/deep_search_shard_seconds', event.deep_search_shard_seconds, generation)
                log_scalar('search_budget/label/total_gpu_seconds', event.total_gpu_seconds, generation)
                log_scalar('search_budget/label/prediction_retries', event.prediction_retry_count, generation)
                log_scalar('search_budget/label/deep_search_retries', event.deep_search_retry_count, generation)
                log_scalar('search_budget/label/completion_generation_lag', event.completion_generation_lag, generation)
                log_scalar('search_budget/label/queued_generations', event.queued_generation_count, generation)
                log_scalar('search_budget/calibration/application_generation', event.application_generation, generation)
                if event.current_validation_gain is not None:
                    log_scalar(
                        'search_budget/calibration/current_validation_gain', event.current_validation_gain, generation
                    )
                if event.ema_validation_gain is not None:
                    log_scalar('search_budget/calibration/ema_validation_gain', event.ema_validation_gain, generation)
                _record_relative_validation_gain(event, generation)
                if event.candidate_mean_assigned_new_visits is not None:
                    log_scalar(
                        'search_budget/calibration/candidate_mean_assigned_new_visits',
                        event.candidate_mean_assigned_new_visits,
                        generation,
                    )
                if event.candidate_assigned_new_visits_variance is not None:
                    log_scalar(
                        'search_budget/calibration/candidate_assigned_new_visits_variance',
                        event.candidate_assigned_new_visits_variance,
                        generation,
                    )
                if event.candidate_mean_kl_from_deep is not None:
                    log_scalar(
                        'search_budget/calibration/candidate_mean_kl_from_deep',
                        event.candidate_mean_kl_from_deep,
                        generation,
                    )
                if event.candidate_exact_spend_residual is not None:
                    log_scalar(
                        'search_budget/calibration/candidate_exact_spend_residual',
                        event.candidate_exact_spend_residual,
                        generation,
                    )
                log_scalar(
                    'search_budget/calibration/minimum_published_multiplier',
                    event.minimum_published_multiplier,
                    generation,
                )
                log_scalar(
                    'search_budget/calibration/maximum_published_multiplier',
                    event.maximum_published_multiplier,
                    generation,
                )
                log_scalar(
                    'search_budget/calibration/published_mean_multiplier',
                    sum(event.published_curve) / len(event.published_curve),
                    generation,
                )
                log_scalar(
                    'search_budget/calibration/previous_published_mean_multiplier',
                    sum(event.previous_published_curve) / len(event.previous_published_curve),
                    generation,
                )
                log_scalar(
                    'search_budget/calibration/shadow_mean_multiplier',
                    sum(event.shadow_curve) / len(event.shadow_curve),
                    generation,
                )
                _record_curve_range('shadow', event.shadow_curve, generation)
                if event.pending_curve is not None:
                    log_scalar(
                        'search_budget/calibration/pending_mean_multiplier',
                        sum(event.pending_curve) / len(event.pending_curve),
                        generation,
                    )
                    _record_curve_range('pending', event.pending_curve, generation)
                if event.validated_curve is not None:
                    log_scalar(
                        'search_budget/calibration/validated_mean_multiplier',
                        sum(event.validated_curve) / len(event.validated_curve),
                        generation,
                    )
                    _record_curve_range('validated', event.validated_curve, generation)
                log_scalar(
                    f'search_budget/calibration/decision_reason/{_metric_tag(event.decision_reason)}', 1, generation
                )
                _record_search_budget_distribution(
                    'search_budget/label/prediction_quantile', event.prediction_distribution, generation
                )
                _record_search_budget_distribution(
                    'search_budget/label/target_quantile', event.target_distribution, generation
                )
                _record_search_budget_distribution('search_budget/label/raw_kl', event.raw_kl_distribution, generation)
                for bucket in event.buckets:
                    prefix = f'search_budget/calibration/bucket_{bucket.bucket_index}'
                    log_scalar(f'{prefix}/sample_count', bucket.sample_count, generation)
                    log_scalar(f'{prefix}/empty', int(bucket.sample_count == 0), generation)
                    if bucket.current_generation_utility is not None:
                        log_scalar(
                            f'{prefix}/generation_marginal_utility', bucket.current_generation_utility, generation
                        )
                    if bucket.ema_utility is not None:
                        log_scalar(f'{prefix}/ema_marginal_utility', bucket.ema_utility, generation)
                    log_scalar(f'{prefix}/shadow_multiplier', bucket.shadow_multiplier, generation)
                    if bucket.pending_multiplier is not None:
                        log_scalar(f'{prefix}/pending_multiplier', bucket.pending_multiplier, generation)
                    if event.validated_curve is not None:
                        log_scalar(
                            f'{prefix}/validated_multiplier', event.validated_curve[bucket.bucket_index], generation
                        )
                    log_scalar(
                        f'{prefix}/previous_published_multiplier',
                        event.previous_published_curve[bucket.bucket_index],
                        generation,
                    )
                    log_scalar(f'{prefix}/published_multiplier', bucket.published_multiplier, generation)
                    log_scalar(f'{prefix}/raw_log_update', bucket.raw_log_update, generation)
                    log_scalar(f'{prefix}/projection_adjustment', bucket.projection_adjustment, generation)
                    if bucket.lower_mean_visits is not None:
                        log_scalar(f'{prefix}/lower_mean_visits', bucket.lower_mean_visits, generation)
                    if bucket.upper_mean_visits is not None:
                        log_scalar(f'{prefix}/upper_mean_visits', bucket.upper_mean_visits, generation)
                    log_scalar(
                        f'{prefix}/checkpoint_deduplication_count',
                        bucket.checkpoint_deduplication_count,
                        generation,
                    )
                log_scalar(
                    'search_budget/calibration/floor_share',
                    event.prediction_distribution.histogram_counts[0] / event.prediction_distribution.count,
                    generation,
                )
                log_scalar(
                    'search_budget/calibration/ceiling_share',
                    event.prediction_distribution.histogram_counts[-1] / event.prediction_distribution.count,
                    generation,
                )
                for condition in event.failed_eligibility_conditions:
                    log_scalar(f'search_budget/calibration/failed_eligibility/{_metric_tag(condition)}', 1, generation)
            case FailedLabelJobReport():
                log_scalar('search_budget/label/status/failed', 1, generation)
                log_scalar(
                    'search_budget/calibration/published_mean_multiplier',
                    sum(event.published_curve) / len(event.published_curve),
                    generation,
                )
                log_scalar('search_budget/calibration/application_generation', event.application_generation, generation)
                log_scalar(
                    f'search_budget/calibration/decision_reason/{_metric_tag(event.decision_reason)}', 1, generation
                )
            case SkippedLabelJobReport():
                log_scalar('search_budget/label/status/skipped', 1, generation)
                log_scalar('search_budget/label/population_positions', event.population_position_count, generation)
                log_scalar('search_budget/label/selected_positions', event.selected_position_count, generation)
                sample_fraction = (
                    event.selected_position_count / event.population_position_count
                    if event.population_position_count > 0
                    else 0.0
                )
                log_scalar('search_budget/label/sample_fraction', sample_fraction, generation)
                log_scalar(f'search_budget/label/skip_reason/{_skip_reason_tag(event.reason)}', 1, generation)

    def _record_training_statistics(
        self,
        publication: TrainingPublication,
        statistics: TrainingStatistics,
        credit_wait_seconds: float,
        ledger_state: CreditLedgerState,
        replay: ReplayDescription,
    ) -> None:
        generation = publication.checkpoint.generation
        training = self.configuration.training
        lifecycle = training_lifecycle_telemetry(
            ledger_state,
            training.lifecycle.credit,
            replay,
            training.trainer.global_batch_size,
        )
        log_scalar('training/policy_loss', statistics.policy_loss, generation)
        log_scalar('training/wdl_loss', statistics.wdl_loss, generation)
        log_scalar('training/total_loss', statistics.total_loss, generation)
        log_scalar('training/gradient_norm', statistics.gradient_norm, generation)
        for index, (head, auxiliary_loss) in enumerate(
            zip(self.auxiliary_heads, statistics.auxiliary_losses, strict=True)
        ):
            log_scalar(f'training_auxiliary/{_auxiliary_name(index, head)}/loss', auxiliary_loss, generation)
        if statistics.term_trunk_gradients:
            policy_gradient, wdl_gradient, *auxiliary_gradients = statistics.term_trunk_gradients
            log_scalar('training_trunk_gradient/policy', policy_gradient, generation)
            log_scalar('training_trunk_gradient/wdl', wdl_gradient, generation)
            main_gradient = max(policy_gradient, wdl_gradient)
            for index, (head, auxiliary_gradient) in enumerate(
                zip(self.auxiliary_heads, auxiliary_gradients, strict=True)
            ):
                name = _auxiliary_name(index, head)
                log_scalar(f'training_trunk_gradient/{name}', auxiliary_gradient, generation)
                if main_gradient > 0.0:
                    log_scalar(
                        f'training_trunk_gradient/{name}_share_of_main',
                        auxiliary_gradient / main_gradient,
                        generation,
                    )
        _record_training_distributions(statistics.distributions, self.auxiliary_heads, generation)
        log_scalar('throughput/training_samples_per_second', statistics.training_samples_per_second, generation)
        log_scalar('training/optimizer_steps', publication.completed_optimizer_steps, generation)
        log_scalar('training/learning_rate', training.trainer.learning_rate.value_at(generation - 1), generation)
        log_scalar('training/quantum_duration_seconds', statistics.elapsed_seconds, generation)
        log_scalar('credit/wait_seconds', credit_wait_seconds, generation)
        log_scalar('credit/configured_replay_ratio', lifecycle.configured_replay_ratio, generation)
        log_scalar('credit/observed_replay_ratio', lifecycle.observed_replay_ratio, generation)
        log_scalar('credit/materialized_samples', lifecycle.materialized_samples, generation)
        log_scalar('credit/consumed_presentations', lifecycle.consumed_presentations, generation)
        log_scalar('credit/available_presentations', lifecycle.available_presentations, generation)
        log_scalar(
            'credit/required_presentations_per_quantum',
            lifecycle.required_presentations_per_quantum,
            generation,
        )
        log_scalar('credit/available_quantum_fraction', lifecycle.available_quantum_fraction, generation)
        log_scalar('replay/live_rows', lifecycle.live_replay_rows, generation)
        log_scalar('replay/logical_capacity', lifecycle.logical_replay_capacity, generation)
        log_scalar('replay/fill_fraction', lifecycle.replay_fill_fraction, generation)
        log(
            f'Completed generation {generation}: loss={statistics.total_loss:.4f}, '
            f'training={statistics.training_samples_per_second:.0f} samples/s, '
            f'credit-wait={credit_wait_seconds:.1f}s, '
            f'available-presentations={lifecycle.available_presentations:.0f}, '
            f'observed-replay-ratio={lifecycle.observed_replay_ratio:.3f}, '
            f'replay={lifecycle.live_replay_rows}/{lifecycle.logical_replay_capacity}'
        )

    @staticmethod
    def _record_progressive_model_statistics(
        model_results: tuple[ModelTrainingResult, ...],
        generation: int,
    ) -> None:
        for model_result in model_results:
            result = model_result.result
            prefix = f'progressive_models/{model_result.model_id}'
            log_scalar(f'{prefix}/policy_loss', result.statistics.policy_loss, generation)
            log_scalar(f'{prefix}/wdl_loss', result.statistics.wdl_loss, generation)
            log_scalar(f'{prefix}/total_loss', result.statistics.total_loss, generation)
            log_scalar(f'{prefix}/gradient_norm', result.statistics.gradient_norm, generation)
            log_scalar(f'{prefix}/optimizer_steps', result.completed_optimizer_steps, generation)
            log_scalar(f'{prefix}/quantum_duration_seconds', result.statistics.elapsed_seconds, generation)

    @staticmethod
    def _record_completed_game_lengths(
        games: tuple[IngestedCompletedGame, ...],
        generation: int,
    ) -> None:
        telemetry = completed_game_length_telemetry(games)
        if telemetry is None:
            return
        lengths = np.asarray(telemetry.lengths_plies, dtype=np.int32)
        log_histogram('self_play/game_length_plies', lengths, generation)
        log_scalar('self_play/completed_games', len(telemetry.lengths_plies), generation)
        log_scalar('self_play/game_length_plies_mean', telemetry.mean_plies, generation)
        log_scalar('self_play/game_length_plies_median', telemetry.median_plies, generation)
        log_scalar('self_play/game_length_plies_p90', telemetry.p90_plies, generation)
        log_scalar('self_play/game_length_plies_p99', telemetry.p99_plies, generation)
        log_scalar('self_play/game_length_plies_maximum', telemetry.maximum_plies, generation)
        for termination in telemetry.terminations:
            prefix = f'self_play/termination/{termination.reason.value}'
            log_scalar(f'{prefix}/completed_games', termination.completed_games, generation)
            log_scalar(f'{prefix}/fraction', termination.fraction, generation)
            if termination.mean_plies is not None:
                log_scalar(f'{prefix}/game_length_plies_mean', termination.mean_plies, generation)

    @staticmethod
    def _record_search_budget(
        games: tuple[IngestedCompletedGame, ...],
        generation: int,
    ) -> None:
        telemetry = search_budget_telemetry(games)
        if telemetry is None:
            return
        log_scalar('search_budget/production/positions', len(telemetry.final_visits), generation)
        _log_values('search_budget/production/baseline_visits', telemetry.baseline_visits, generation, log_mean=True)
        _log_values('search_budget/production/final_visits', telemetry.final_visits, generation, log_mean=True)
        _log_values(
            'search_budget/production/assigned_additional_visits',
            telemetry.assigned_additional_visits,
            generation,
            log_mean=True,
        )
        _log_values('search_budget/production/prediction_logit', telemetry.search_budget_logits, generation)
        _log_values(
            'search_budget/production/predicted_quantile',
            telemetry.predicted_search_budgets,
            generation,
            log_mean=True,
        )
        _log_values(
            'search_budget/production/parallel_searches', telemetry.parallel_searches, generation, log_mean=True
        )
        _log_values('search_budget/production/spend_residual', telemetry.spend_residuals, generation)
        _log_values('search_budget/production/starting_visits', telemetry.starting_visits, generation, log_mean=True)
        _log_values(
            'search_budget/production/policy_correction',
            telemetry.policy_corrections,
            generation,
            log_mean=True,
        )
        _log_values(
            'search_budget/production/value_correction',
            telemetry.value_corrections,
            generation,
            log_mean=True,
        )
        for reason, count in telemetry.stop_reasons:
            log_scalar(f'search_budget/production/stop_reason/{reason.value}', count, generation)

    def _record_scheduled_settings(self, generation: int) -> None:
        for setting in scheduled_settings_at(self.configuration, generation):
            log_scalar(setting.tag, setting.value, generation)


def _record_training_distributions(
    distributions: TrainingDistributionSnapshot,
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...],
    generation: int,
) -> None:
    _record_policy_distribution('training_diagnostics/policy', distributions.policy, generation)
    _log_values('training_diagnostics/wdl_loss', distributions.wdl_loss, generation)
    _log_values('training_diagnostics/root_value', distributions.root_value, generation, log_mean=True)
    _log_values('training_diagnostics/terminal_value', distributions.terminal_value, generation, log_mean=True)
    _log_values('training_diagnostics/predicted_value', distributions.predicted_value, generation, log_mean=True)
    _log_values(
        'training_diagnostics/value_absolute_error',
        distributions.value_absolute_error,
        generation,
        log_mean=True,
    )
    _log_values('training_diagnostics/sample_weight', distributions.sample_weight, generation)
    _log_values(
        'replay_diagnostics/generation_age',
        distributions.replay_generation_age,
        generation,
        log_mean=True,
    )
    _log_values('replay_diagnostics/age_seconds', distributions.replay_age_seconds, generation, log_mean=True)
    for index, (head, auxiliary) in enumerate(zip(auxiliary_heads, distributions.auxiliary, strict=True)):
        prefix = f'training_auxiliary/{_auxiliary_name(index, head)}'
        match auxiliary:
            case NextPolicyTrainingDistribution(policy=policy):
                _record_policy_distribution(prefix, policy, generation)
            case RemainingGameLengthTrainingDistribution(
                target=target,
                prediction=prediction,
                absolute_error=absolute_error,
            ):
                _log_values(f'{prefix}/target', target, generation, log_mean=True)
                _log_values(f'{prefix}/prediction', prediction, generation, log_mean=True)
                _log_values(f'{prefix}/absolute_error', absolute_error, generation, log_mean=True)
            case ScalarAuxiliaryTrainingDistribution(
                target=target,
                prediction=prediction,
                absolute_error=absolute_error,
            ):
                _log_values(f'{prefix}/target', target, generation, log_mean=True)
                _log_values(f'{prefix}/prediction', prediction, generation, log_mean=True)
                _log_values(f'{prefix}/absolute_error', absolute_error, generation, log_mean=True)
            case LegalMovesTrainingDistribution(
                legal_probability=legal_probability,
                illegal_probability=illegal_probability,
            ):
                _log_values(f'{prefix}/legal_probability', legal_probability, generation, log_mean=True)
                _log_values(f'{prefix}/illegal_probability', illegal_probability, generation, log_mean=True)


def _record_policy_distribution(prefix: str, policy: PolicyTrainingDistribution, generation: int) -> None:
    _log_values(f'{prefix}/loss', policy.loss, generation)
    _log_values(f'{prefix}/target_top1_mass', policy.target_top1_mass, generation, log_mean=True)
    _log_values(f'{prefix}/target_top2_mass', policy.target_top2_mass, generation, log_mean=True)
    _log_values(f'{prefix}/target_top3_mass', policy.target_top3_mass, generation, log_mean=True)
    _log_values(f'{prefix}/target_entropy', policy.target_entropy, generation, log_mean=True)
    _log_values(f'{prefix}/prediction_entropy', policy.prediction_entropy, generation, log_mean=True)


def _log_values(name: str, values: tuple[float, ...], generation: int, log_mean: bool = False) -> None:
    if not values:
        return
    array = np.asarray(values, dtype=np.float32)
    log_histogram(name, array, generation)
    if log_mean:
        log_scalar(f'{name}_mean', float(array.mean()), generation)


def _record_search_budget_distribution(prefix: str, distribution: DistributionSummary, generation: int) -> None:
    log_equal_width_histogram_summary(
        prefix,
        distribution.minimum,
        distribution.maximum,
        distribution.count,
        distribution.mean,
        distribution.variance,
        distribution.histogram_counts,
        generation,
    )
    log_scalar(f'{prefix}/count', distribution.count, generation)
    log_scalar(f'{prefix}/minimum', distribution.minimum, generation)
    log_scalar(f'{prefix}/maximum', distribution.maximum, generation)
    log_scalar(f'{prefix}/mean', distribution.mean, generation)
    log_scalar(f'{prefix}/variance', distribution.variance, generation)
    log_scalar(f'{prefix}/p10', distribution.p10, generation)
    log_scalar(f'{prefix}/p25', distribution.p25, generation)
    log_scalar(f'{prefix}/median', distribution.median, generation)
    log_scalar(f'{prefix}/p75', distribution.p75, generation)
    log_scalar(f'{prefix}/p90', distribution.p90, generation)
    for index, count in enumerate(distribution.histogram_counts):
        log_scalar(f'{prefix}/histogram_bin_{index}', count, generation)


def _record_relative_validation_gain(event: GenerationLabelReport, generation: int) -> None:
    if event.current_validation_gain is None or event.candidate_mean_kl_from_deep is None:
        return
    flat_mean_kl = event.candidate_mean_kl_from_deep + event.current_validation_gain
    if flat_mean_kl <= 0.0:
        return
    log_scalar(
        'search_budget/calibration/current_relative_validation_gain_percent',
        100.0 * event.current_validation_gain / flat_mean_kl,
        generation,
    )
    if event.ema_validation_gain is not None:
        log_scalar(
            'search_budget/calibration/ema_relative_validation_gain_percent',
            100.0 * event.ema_validation_gain / flat_mean_kl,
            generation,
        )


def _record_curve_range(name: str, curve: tuple[float, ...], generation: int) -> None:
    log_scalar(f'search_budget/calibration/minimum_{name}_multiplier', min(curve), generation)
    log_scalar(f'search_budget/calibration/maximum_{name}_multiplier', max(curve), generation)


def _metric_tag(value: str) -> str:
    return ''.join(character if character.isalnum() else '_' for character in value).strip('_')


def _skip_reason_tag(reason: str) -> str:
    if reason.startswith('unstarted source-generation lag'):
        return 'unstarted_generation_lag'
    if reason == 'generation population produces zero positions at the configured sample fraction':
        return 'zero_position_sample'
    return 'other'


def _auxiliary_name(index: int, head: AuxiliaryHeadLayout) -> str:
    match head:
        case NextPolicyHeadLayout(ply_offset=ply_offset):
            return f'{index}-next-policy-ply-{ply_offset}'
        case RemainingGameLengthHeadLayout():
            return f'{index}-remaining-game-length'
        case FutureSearchValueHeadLayout(ply_offset=ply_offset):
            return f'{index}-future-search-value-ply-{ply_offset}'
        case IrreversibleProgressHeadLayout(horizon_plies=horizon_plies):
            return f'{index}-irreversible-progress-{horizon_plies}'
        case LegalMovesHeadLayout():
            return f'{index}-legal-moves'
        case SearchBudgetHeadLayout():
            return f'{index}-search-budget'

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from src.experiment.configuration import ExperimentConfiguration
from src.replay.description import ReplayDescription
from src.replay.manager import IngestedCompletedGame
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
    SearchCorrectionHeadLayout,
)
from src.training.telemetry import (
    adaptive_search_telemetry,
    completed_game_length_telemetry,
    training_lifecycle_telemetry,
)
from src.training.tensorboard import scheduled_settings_at
from src.training.trainer import TrainingStatistics
from src.util.log import log
from src.util.tensorboard import log_histogram, log_scalar


@dataclass(frozen=True)
class ReplayIngestionTelemetry:
    ingest_seconds: float
    inbox_depth: int
    staging_depth: int
    materialization_failures: int


class TrainingReporter:
    def __init__(
        self,
        configuration: ExperimentConfiguration,
        auxiliary_heads: tuple[AuxiliaryHeadLayout, ...],
    ) -> None:
        self.configuration = configuration
        self.auxiliary_heads = auxiliary_heads

    def record_initial_settings(self, generation: int) -> None:
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
        self._record_adaptive_search(completed_games, outcome.publication.checkpoint.generation)
        self._record_scheduled_settings(outcome.publication.checkpoint.generation)

    @staticmethod
    def _record_ingestion(ingestion: ReplayIngestionTelemetry, generation: int) -> None:
        log_scalar('replay/ingest_seconds', ingestion.ingest_seconds, generation)
        log_scalar('replay/inbox_depth', ingestion.inbox_depth, generation)
        log_scalar('replay/staging_depth', ingestion.staging_depth, generation)
        log_scalar('replay/materialization_failures', ingestion.materialization_failures, generation)

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
        _record_training_distributions(statistics.distributions, self.auxiliary_heads, generation)
        log_scalar('throughput/replay_rows_per_second', statistics.replay_rows_per_second, generation)
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
            f'replay={statistics.replay_rows_per_second:.0f} rows/s, '
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
    def _record_adaptive_search(
        games: tuple[IngestedCompletedGame, ...],
        generation: int,
    ) -> None:
        telemetry = adaptive_search_telemetry(games)
        if telemetry is None:
            return
        log_scalar('adaptive_search/full_searches', len(telemetry.final_visits), generation)
        _log_values('adaptive_search/final_visits', telemetry.final_visits, generation, log_mean=True)
        _log_values('adaptive_search/new_simulations', telemetry.new_simulations, generation, log_mean=True)
        _log_values(
            'adaptive_search/search_correction_target',
            telemetry.search_correction_targets,
            generation,
            log_mean=True,
        )
        _log_values(
            'adaptive_search/search_correction_prediction',
            telemetry.search_correction_predictions,
            generation,
            log_mean=True,
        )
        _log_values(
            'adaptive_search/policy_correction',
            telemetry.policy_corrections,
            generation,
            log_mean=True,
        )
        _log_values(
            'adaptive_search/value_correction',
            telemetry.value_corrections,
            generation,
            log_mean=True,
        )
        if telemetry.checkpoint_visits:
            _log_values('adaptive_search/checkpoint_visits', telemetry.checkpoint_visits, generation, log_mean=True)
            _log_values(
                'adaptive_search/checkpoint_top_visit_share',
                telemetry.checkpoint_top_visit_shares,
                generation,
                log_mean=True,
            )
            _log_values(
                'adaptive_search/checkpoint_top_two_margin',
                telemetry.checkpoint_top_two_margins,
                generation,
                log_mean=True,
            )
            _log_values(
                'adaptive_search/checkpoint_root_value_delta',
                telemetry.checkpoint_root_value_deltas,
                generation,
                log_mean=True,
            )
        log_scalar('adaptive_search/leader_changes', telemetry.leader_changes, generation)
        log_scalar('adaptive_search/learned_gate/evaluations', telemetry.learned_gate_evaluations, generation)
        log_scalar('adaptive_search/learned_gate/denials', telemetry.learned_gate_denials, generation)
        log_scalar('adaptive_search/learned_gate/unlocks', telemetry.learned_gate_unlocks, generation)
        for reason, count in telemetry.stop_reasons:
            log_scalar(f'adaptive_search/stop_reason/{reason.value}', count, generation)

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
        case SearchCorrectionHeadLayout():
            return f'{index}-search-correction'

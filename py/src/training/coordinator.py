from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from src.evaluation.manager import EvaluationManager
from src.experiment.base_configuration import initial_generation
from src.games.implementation import GameImplementation
from src.replay.layout import ReplayLayout
from src.replay.manager import IngestedCompletedGame, ReplayManager
from src.self_play.protocol import (
    RunningSelfPlayState,
    StatisticsLevel,
)
from src.self_play.resignation import PublishedResignationPolicy, ResignationCalibrator
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.persistence import import_checkpoint, publish_checkpoint
from src.training.checkpoint.retention import CheckpointRetention
from src.training.credit_ledger import CreditLedger
from src.training.self_play_group import SelfPlayGroup
from src.training.distributions import (
    NextPolicyTrainingDistribution,
    PolicyTrainingDistribution,
    RemainingGameLengthTrainingDistribution,
    TrainingDistributionSnapshot,
)
from src.training.targets import AuxiliaryHeadLayout, NextPolicyHeadLayout, RemainingGameLengthHeadLayout
from src.training.telemetry import completed_game_length_telemetry, training_lifecycle_telemetry
from src.training.tensorboard import scheduled_settings_at
from src.training.trainer import TrainerGroup, TrainingQuantumResult
from src.training.progress import TrainingProgress
from src.training.progressive import (
    CompletedCandidateTraining,
    ProgressiveTrainingStateStore,
    retain_progressive_candidate_checkpoints,
)
from src.training.run_limits import RunLimitMonitor
from src.util.log import log
from src.util.tensorboard import log_histogram, log_scalar


class Coordinator:
    def __init__(self, game: GameImplementation, run_started_at: float) -> None:
        self.game = game
        self.configuration = game.configuration
        self.run_started_at = run_started_at
        training = game.training
        run_path = Path(training.save_path)
        self.run_limit_monitor = RunLimitMonitor(training.limits, run_path, run_started_at)
        starting_checkpoint = CheckpointReference.load(
            run_path,
            initial_generation(self.configuration.run.resume),
        )
        self.ledger = CreditLedger(
            run_path,
            training.lifecycle.credit,
            training.trainer.global_batch_size,
            starting_checkpoint,
            adopt_completed_quantum=training.progressive_model_sizing is None,
        )
        progressive_configuration = training.progressive_model_sizing
        self.progressive_training = (
            None
            if progressive_configuration is None
            else ProgressiveTrainingStateStore(run_path / 'progressive-training.json', progressive_configuration)
        )
        if self.progressive_training is not None:
            self._recover_published_progressive_checkpoint(run_path)
        resignation_configuration = game.resignation_configuration
        self.resignation_calibrator = (
            None
            if resignation_configuration is None
            else ResignationCalibrator(run_path / 'resignation' / 'calibration.json', resignation_configuration)
        )
        if self.resignation_calibrator is not None:
            self.resignation_calibrator.advance_generation(self.ledger.model_generation)
        replay_layout = ReplayLayout(
            packed_planes=game.state.packed_plane_layout,
            targets=game.target_layout,
            maximum_policy_entries=training.lifecycle.replay.maximum_policy_entries,
        )
        self.replay_manager = ReplayManager.open(
            run_path,
            game.state,
            replay_layout,
            training.lifecycle.replay,
            self.ledger.model_generation,
            game.value_discount_per_ply,
            self.resignation_calibrator,
        )
        self.trainer_group = (
            TrainerGroup(self.configuration, game, self.ledger.state.active_checkpoint)
            if self.progressive_training is None
            else None
        )
        self.self_play_group = SelfPlayGroup(game)
        self.evaluation_manager = EvaluationManager(self.configuration, self.ledger.state.active_checkpoint)
        self.checkpoint_retention = CheckpointRetention(run_path, training.lifecycle)
        self._apply_checkpoint_retention()
        self.latest_completed_model_version = self.ledger.model_generation
        self.final_stop_reason: str | None = None
        self._backpressure_pause_requested = False
        self._credit_wait_started_at = time.perf_counter()
        self._completed_games_since_last_quantum: list[IngestedCompletedGame] = []
        self._record_scheduled_settings(self.ledger.model_generation)

    def run(self) -> None:
        try:
            if self.progressive_training is not None and self.progressive_training.state.pending_quantum is not None:
                self._train_progressive_quantum(self_play_started=False)
            self._start_self_play()
            while not self.ledger.training_complete:
                self.evaluation_manager.collect_completed_jobs()
                if self.progressive_training is None:
                    self.evaluation_manager.schedule_due_jobs(self.ledger.state.active_checkpoint)
                restarted_workers = self.self_play_group.restart_exited_workers(
                    self.ledger.state.active_checkpoint,
                    self._resignation_policy(),
                )
                for worker_id in restarted_workers:
                    log(f'Restarted self-play worker {worker_id} at generation {self.ledger.model_generation}.')
                self.final_stop_reason = self.run_limit_monitor.stop_reason()
                if self.final_stop_reason is not None:
                    break
                self._apply_self_play_backpressure()
                self._ingest_available_games()
                self._apply_self_play_backpressure()
                if not self.ledger.can_train_quantum(self.replay_manager.live_samples):
                    if self.progressive_training is not None:
                        self.evaluation_manager.schedule_due_jobs(self.ledger.state.active_checkpoint)
                    time.sleep(0.1)
                    continue
                self._train_quantum()
                if self.progressive_training is not None:
                    self.evaluation_manager.schedule_due_jobs(self.ledger.state.active_checkpoint)
        finally:
            self.evaluation_manager.close()
            self.self_play_group.close()
            if self.trainer_group is not None:
                self.trainer_group.close()
            self.replay_manager.close()
            self.ledger.save()

    def _start_self_play(self) -> None:
        checkpoint = self.ledger.state.active_checkpoint
        responses = self.self_play_group.apply(
            tuple(
                RunningSelfPlayState(checkpoint=checkpoint, resignation_policy=self._resignation_policy())
                for _ in range(self.self_play_group.worker_count)
            )
        )
        if any(response.kind != 'running' for response in responses):
            raise RuntimeError('Self-play workers did not enter the running state.')
        self._credit_wait_started_at = time.perf_counter()

    def _ingest_available_games(self) -> None:
        generation = self.ledger.model_generation
        ingestion = self.replay_manager.ingest_available_games(generation)
        self.ledger.add_samples(ingestion.samples_added, generation)
        self._completed_games_since_last_quantum.extend(ingestion.completed_games)
        if ingestion.games_ingested:
            self._record_resignation_diagnostics(generation)

    def _train_quantum(self) -> None:
        if self.progressive_training is not None:
            self._train_progressive_quantum(self_play_started=True)
            return
        credit_wait_seconds = time.perf_counter() - self._credit_wait_started_at
        paused_worker_ids = self._training_pause_worker_ids()
        self.self_play_group.request_pause(paused_worker_ids)
        assert self.trainer_group is not None
        result = self.trainer_group.train_quantum(self.replay_manager.description(), self.ledger.progress)
        self.ledger.commit_quantum(result)
        self.latest_completed_model_version = self.ledger.model_generation
        self._ingest_available_games()
        if self.resignation_calibrator is not None:
            self.resignation_calibrator.advance_generation(self.ledger.model_generation)
            self._record_resignation_diagnostics(self.ledger.model_generation)
        self._record_training_statistics(result, credit_wait_seconds)
        detailed_workers = self._detailed_statistics_workers()
        desired_states = tuple(
            RunningSelfPlayState(
                checkpoint=result.checkpoint,
                resignation_policy=self._resignation_policy(),
                completed_generation_statistics=(
                    StatisticsLevel.DETAILED if worker_id < detailed_workers else StatisticsLevel.BASIC
                ),
            )
            for worker_id in range(self.self_play_group.worker_count)
        )
        applied = self.self_play_group.apply(desired_states)
        if any(response.kind != 'running' for response in applied):
            raise RuntimeError('Self-play workers did not apply the trained checkpoint.')
        self._backpressure_pause_requested = False
        self._apply_self_play_backpressure()
        self._apply_checkpoint_retention()
        self._credit_wait_started_at = time.perf_counter()

    def _train_progressive_quantum(self, self_play_started: bool) -> None:
        progressive = self.progressive_training
        configuration = self.configuration.training.progressive_model_sizing
        assert progressive is not None
        assert configuration is not None
        credit_wait_seconds = time.perf_counter() - self._credit_wait_started_at
        if self_play_started:
            self.self_play_group.request_pause(tuple(range(self.self_play_group.worker_count)))
        replay = self.replay_manager.description()
        credit = self.configuration.training.lifecycle.credit
        pending = progressive.begin_quantum(
            self.evaluation_manager.elapsed_seconds,
            replay,
            self.ledger.progress.completed_optimizer_steps,
            credit.optimizer_steps_per_quantum,
        )
        run_path = Path(self.configuration.training.save_path)
        while pending.next_model_id is not None:
            model_id = pending.next_model_id
            definition = configuration.model(model_id)
            candidate = progressive.candidate(model_id)
            model_path = run_path / 'models' / model_id
            if candidate.checkpoint is None and model_id == progressive.state.active_model_id:
                imported = import_checkpoint(
                    self.ledger.state.active_checkpoint.manifest_path,
                    self.ledger.state.active_checkpoint.generation,
                    model_path,
                )
                progressive.initialize_candidate(
                    model_id,
                    self.ledger.progress.completed_optimizer_steps,
                    imported,
                )
                candidate = progressive.candidate(model_id)
            model_progress = TrainingProgress(
                completed_optimizer_steps=candidate.completed_optimizer_steps,
                optimizer_steps_per_generation=credit.optimizer_steps_per_quantum,
            )
            trainer = TrainerGroup(
                self.configuration,
                self.game,
                candidate.checkpoint,
                network=definition.network,
                save_path=str(model_path),
            )
            try:
                result = trainer.train_quantum(
                    replay,
                    model_progress,
                    replay_source_optimizer_steps=pending.replay_batch.source_optimizer_steps,
                )
            finally:
                trainer.close()
            progressive.record_candidate(
                CompletedCandidateTraining(
                    model_id=model_id,
                    completed_optimizer_steps=result.completed_optimizer_steps,
                    checkpoint=result.checkpoint,
                    comparable_total_loss=result.statistics.total_loss,
                )
            )
            self._record_model_training_statistics(model_id, result)
            pending = progressive.state.pending_quantum
            assert pending is not None

        published_model_id = progressive.preview_active_model_id()
        published_training = progressive.completed_result(published_model_id)
        published_checkpoint = publish_checkpoint(
            published_training.checkpoint,
            self.ledger.model_generation + 1,
            run_path,
        )
        progressive.complete_quantum()
        self.ledger.commit_checkpoint(pending.target_global_optimizer_steps, published_checkpoint)
        retain_progressive_candidate_checkpoints(run_path, progressive.state)
        self.latest_completed_model_version = self.ledger.model_generation
        self._ingest_available_games()
        self._finish_progressive_quantum(
            published_checkpoint,
            published_model_id,
            credit_wait_seconds,
            self_play_started,
        )

    def _finish_progressive_quantum(
        self,
        checkpoint: CheckpointReference,
        model_id: str,
        credit_wait_seconds: float,
        self_play_started: bool,
    ) -> None:
        if self.resignation_calibrator is not None:
            self.resignation_calibrator.advance_generation(self.ledger.model_generation)
            self._record_resignation_diagnostics(self.ledger.model_generation)
        log_scalar('progressive/active_model_index', self._progressive_model_index(model_id), checkpoint.generation)
        if self_play_started:
            detailed_workers = self._detailed_statistics_workers()
            desired_states = tuple(
                RunningSelfPlayState(
                    checkpoint=checkpoint,
                    resignation_policy=self._resignation_policy(),
                    completed_generation_statistics=(
                        StatisticsLevel.DETAILED if worker_id < detailed_workers else StatisticsLevel.BASIC
                    ),
                )
                for worker_id in range(self.self_play_group.worker_count)
            )
            applied = self.self_play_group.apply(desired_states)
            if any(response.kind != 'running' for response in applied):
                raise RuntimeError('Self-play workers did not apply the promoted progressive checkpoint.')
        self._backpressure_pause_requested = False
        self._apply_self_play_backpressure()
        self._apply_checkpoint_retention()
        self._credit_wait_started_at = time.perf_counter()
        log(
            f'Published progressive model {model_id} at generation {checkpoint.generation} '
            f'after {credit_wait_seconds:.1f}s of credit wait.'
        )

    def _record_model_training_statistics(self, model_id: str, result: TrainingQuantumResult) -> None:
        generation = self.ledger.model_generation + 1
        prefix = f'progressive_models/{model_id}'
        log_scalar(f'{prefix}/policy_loss', result.statistics.policy_loss, generation)
        log_scalar(f'{prefix}/wdl_loss', result.statistics.wdl_loss, generation)
        log_scalar(f'{prefix}/total_loss', result.statistics.total_loss, generation)
        log_scalar(f'{prefix}/gradient_norm', result.statistics.gradient_norm, generation)
        log_scalar(f'{prefix}/optimizer_steps', result.completed_optimizer_steps, generation)
        log_scalar(f'{prefix}/quantum_duration_seconds', result.statistics.elapsed_seconds, generation)

    def _progressive_model_index(self, model_id: str) -> int:
        configuration = self.configuration.training.progressive_model_sizing
        assert configuration is not None
        return tuple(model.model_id for model in configuration.models).index(model_id)

    def _recover_published_progressive_checkpoint(self, run_path: Path) -> None:
        assert self.progressive_training is not None
        if self.progressive_training.state.pending_quantum is not None:
            return
        generation = self.ledger.model_generation + 1
        manifest_path = run_path / f'checkpoint_{generation}.json'
        if not manifest_path.exists():
            return
        checkpoint = CheckpointReference.load(run_path, generation)
        completed_steps = self.ledger.progress.completed_optimizer_steps
        completed_steps += self.configuration.training.lifecycle.credit.optimizer_steps_per_quantum
        self.ledger.commit_checkpoint(completed_steps, checkpoint)

    def _training_pause_worker_ids(self) -> tuple[int, ...]:
        if self._backpressure_pause_requested:
            return ()
        if self._self_play_backpressure_required():
            return tuple(range(self.self_play_group.worker_count))
        return self.configuration.training.topology.self_play.node_ids_to_pause_during_training

    def _apply_self_play_backpressure(self) -> None:
        if not self._self_play_backpressure_required() or self._backpressure_pause_requested:
            return
        self.self_play_group.request_pause(tuple(range(self.self_play_group.worker_count)))
        self._backpressure_pause_requested = True

    def _self_play_backpressure_required(self) -> bool:
        training = self.configuration.training
        return training.lifecycle.credit.requires_self_play_backpressure(
            self.ledger.state.available_credits,
            training.trainer.global_batch_size,
        )

    def _resignation_policy(self) -> PublishedResignationPolicy:
        if self.resignation_calibrator is None:
            return PublishedResignationPolicy()
        return self.resignation_calibrator.published_policy(self.ledger.model_generation)

    def _record_resignation_diagnostics(self, generation: int) -> None:
        if self.resignation_calibrator is None:
            return
        diagnostics = self.resignation_calibrator.diagnostics()
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

    def _apply_checkpoint_retention(self) -> None:
        self.checkpoint_retention.apply(
            self.ledger.model_generation,
            self.evaluation_manager.required_checkpoint_generations,
        )

    def _detailed_statistics_workers(self) -> int:
        configured = self.game.self_play_configuration.detailed_statistics_workers
        return min(configured, self.self_play_group.worker_count)

    def _record_training_statistics(self, result: TrainingQuantumResult, credit_wait_seconds: float) -> None:
        generation = result.checkpoint.generation
        source_generation = generation - 1
        statistics = result.statistics
        training = self.configuration.training
        lifecycle = training_lifecycle_telemetry(
            self.ledger.state,
            training.lifecycle.credit,
            self.replay_manager.description(),
            training.trainer.global_batch_size,
        )
        log_scalar('training/policy_loss', statistics.policy_loss, generation)
        log_scalar('training/wdl_loss', statistics.wdl_loss, generation)
        log_scalar('training/total_loss', statistics.total_loss, generation)
        log_scalar('training/gradient_norm', statistics.gradient_norm, generation)
        for index, (head, auxiliary_loss) in enumerate(
            zip(self.game.target_layout.auxiliary_heads, statistics.auxiliary_losses, strict=True)
        ):
            log_scalar(f'training_auxiliary/{_auxiliary_name(index, head)}/loss', auxiliary_loss, generation)
        _record_training_distributions(
            statistics.distributions,
            self.game.target_layout.auxiliary_heads,
            generation,
        )
        log_scalar('throughput/replay_rows_per_second', statistics.replay_rows_per_second, generation)
        log_scalar('throughput/training_samples_per_second', statistics.training_samples_per_second, generation)
        log_scalar('training/optimizer_steps', result.completed_optimizer_steps, generation)
        log_scalar('training/learning_rate', training.trainer.learning_rate.value_at(source_generation), generation)
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
        self._record_completed_game_lengths(generation)
        self._record_scheduled_settings(generation)
        log(
            f'Completed generation {generation}: loss={statistics.total_loss:.4f}, '
            f'replay={statistics.replay_rows_per_second:.0f} rows/s, '
            f'training={statistics.training_samples_per_second:.0f} samples/s, '
            f'credit-wait={credit_wait_seconds:.1f}s, '
            f'available-presentations={lifecycle.available_presentations:.0f}, '
            f'observed-replay-ratio={lifecycle.observed_replay_ratio:.3f}, '
            f'replay={lifecycle.live_replay_rows}/{lifecycle.logical_replay_capacity}'
        )

    def _record_completed_game_lengths(self, generation: int) -> None:
        telemetry = completed_game_length_telemetry(tuple(self._completed_games_since_last_quantum))
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
        self._completed_games_since_last_quantum.clear()

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


def _record_policy_distribution(
    prefix: str,
    policy: PolicyTrainingDistribution,
    generation: int,
) -> None:
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

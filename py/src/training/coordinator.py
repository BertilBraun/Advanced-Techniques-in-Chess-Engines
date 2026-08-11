from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from src.evaluation.manager import EvaluationManager
from src.games.implementation import GameImplementation
from src.replay.layout import ReplayLayout
from src.replay.manager import ReplayManager
from src.self_play.protocol import (
    PausedSelfPlayState,
    RunningSelfPlayState,
    StatisticsLevel,
)
from src.training.checkpoint import CheckpointReference
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
from src.training.telemetry import training_lifecycle_telemetry
from src.training.trainer import TrainerGroup, TrainingQuantumResult
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
        generation_zero = CheckpointReference.load(run_path, 0)
        self.ledger = CreditLedger(
            run_path,
            training.lifecycle.credit,
            training.trainer.global_batch_size,
            generation_zero,
        )
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
        )
        self.trainer_group = TrainerGroup(self.configuration, game, self.ledger.state.active_checkpoint)
        self.self_play_group = SelfPlayGroup(game)
        self.evaluation_manager = EvaluationManager(self.configuration, self.ledger.state.active_checkpoint)
        self.checkpoint_retention = CheckpointRetention(run_path, training.lifecycle)
        self._apply_checkpoint_retention()
        self.latest_completed_model_version = self.ledger.model_generation
        self.final_stop_reason: str | None = None
        self._credit_wait_started_at = time.perf_counter()

    def run(self) -> None:
        try:
            self._start_self_play()
            while not self.ledger.training_complete:
                self.evaluation_manager.collect_completed_jobs()
                self.evaluation_manager.schedule_due_jobs(self.ledger.state.active_checkpoint)
                restarted_workers = self.self_play_group.restart_exited_workers(self.ledger.state.active_checkpoint)
                for worker_id in restarted_workers:
                    log(f'Restarted self-play worker {worker_id} at generation {self.ledger.model_generation}.')
                self.final_stop_reason = self.run_limit_monitor.stop_reason()
                if self.final_stop_reason is not None:
                    break
                self._ingest_available_games()
                if not self.ledger.can_train_quantum(self.replay_manager.live_samples):
                    time.sleep(min(0.1, self.evaluation_manager.seconds_until_next_boundary()))
                    continue
                self._train_quantum()
        finally:
            self.evaluation_manager.close()
            self.self_play_group.close()
            self.trainer_group.close()
            self.replay_manager.close()
            self.ledger.save()

    def _start_self_play(self) -> None:
        checkpoint = self.ledger.state.active_checkpoint
        responses = self.self_play_group.apply(
            tuple(RunningSelfPlayState(checkpoint=checkpoint) for _ in range(self.self_play_group.worker_count))
        )
        if any(response.kind != 'running' for response in responses):
            raise RuntimeError('Self-play workers did not enter the running state.')
        self._credit_wait_started_at = time.perf_counter()

    def _ingest_available_games(self) -> None:
        generation = self.ledger.model_generation
        ingestion = self.replay_manager.ingest_available_games(generation)
        self.ledger.add_samples(ingestion.samples_added, generation)

    def _train_quantum(self) -> None:
        credit_wait_seconds = time.perf_counter() - self._credit_wait_started_at
        paused_worker_ids = self.configuration.training.topology.self_play.node_ids_to_pause_during_training
        paused = self.self_play_group.apply_to_workers(
            paused_worker_ids,
            PausedSelfPlayState(),
        )
        if any(response.kind != 'paused' for response in paused):
            raise RuntimeError('Selected self-play workers did not enter the paused state.')
        self._ingest_available_games()
        if not self.ledger.can_train_quantum(self.replay_manager.live_samples):
            resumed = self.self_play_group.apply_to_workers(
                paused_worker_ids,
                RunningSelfPlayState(checkpoint=self.ledger.state.active_checkpoint),
            )
            if any(response.kind != 'running' for response in resumed):
                raise RuntimeError('Selected self-play workers did not resume after a cancelled training quantum.')
            return
        result = self.trainer_group.train_quantum(self.replay_manager.description(), self.ledger.progress)
        self.ledger.commit_quantum(result)
        self.latest_completed_model_version = self.ledger.model_generation
        self._record_training_statistics(result, credit_wait_seconds)
        detailed_workers = self._detailed_statistics_workers()
        desired_states = tuple(
            RunningSelfPlayState(
                checkpoint=result.checkpoint,
                completed_generation_statistics=(
                    StatisticsLevel.DETAILED if worker_id < detailed_workers else StatisticsLevel.BASIC
                ),
            )
            for worker_id in range(self.self_play_group.worker_count)
        )
        applied = self.self_play_group.apply(desired_states)
        if any(response.kind != 'running' for response in applied):
            raise RuntimeError('Self-play workers did not apply the trained checkpoint.')
        self._apply_checkpoint_retention()
        self._credit_wait_started_at = time.perf_counter()

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
        self_play = self.game.self_play_parameters_at(source_generation)
        log_scalar('training/policy_loss', statistics.policy_loss, generation)
        log_scalar('training/wdl_loss', statistics.wdl_loss, generation)
        log_scalar('training/total_loss', statistics.total_loss, generation)
        log_scalar('training/gradient_norm', statistics.gradient_norm, generation)
        for index, (head, auxiliary_loss) in enumerate(
            zip(self.game.target_layout.auxiliary_heads, statistics.auxiliary_losses, strict=True)
        ):
            log_scalar(f'training/auxiliary/{_auxiliary_name(index, head)}/loss', auxiliary_loss, generation)
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
        log_scalar('self_play/full_searches', self_play.full_searches, generation)
        log_scalar('self_play/fast_searches', self_play.fast_searches, generation)
        log_scalar('self_play/full_search_probability', self_play.full_search_probability, generation)
        log_scalar(
            'training/root_value_blend',
            self.game.training_objective_at(source_generation).root_value_blend,
            generation,
        )
        log(
            f'Completed generation {generation}: loss={statistics.total_loss:.4f}, '
            f'replay={statistics.replay_rows_per_second:.0f} rows/s, '
            f'training={statistics.training_samples_per_second:.0f} samples/s, '
            f'credit-wait={credit_wait_seconds:.1f}s, '
            f'available-presentations={lifecycle.available_presentations:.0f}, '
            f'observed-replay-ratio={lifecycle.observed_replay_ratio:.3f}, '
            f'replay={lifecycle.live_replay_rows}/{lifecycle.logical_replay_capacity}'
        )


def _record_training_distributions(
    distributions: TrainingDistributionSnapshot,
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...],
    generation: int,
) -> None:
    _record_policy_distribution('training/distribution/policy', distributions.policy, generation)
    _log_values('training/distribution/wdl_loss', distributions.wdl_loss, generation)
    _log_values('training/distribution/root_value', distributions.root_value, generation, log_mean=True)
    _log_values('training/distribution/terminal_value', distributions.terminal_value, generation, log_mean=True)
    _log_values('training/distribution/predicted_value', distributions.predicted_value, generation, log_mean=True)
    _log_values(
        'training/distribution/value_absolute_error',
        distributions.value_absolute_error,
        generation,
        log_mean=True,
    )
    _log_values('training/distribution/sample_weight', distributions.sample_weight, generation)
    _log_values(
        'replay/distribution/generation_age',
        distributions.replay_generation_age,
        generation,
        log_mean=True,
    )
    _log_values('replay/distribution/age_seconds', distributions.replay_age_seconds, generation, log_mean=True)
    for index, (head, auxiliary) in enumerate(zip(auxiliary_heads, distributions.auxiliary, strict=True)):
        prefix = f'training/auxiliary/{_auxiliary_name(index, head)}'
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

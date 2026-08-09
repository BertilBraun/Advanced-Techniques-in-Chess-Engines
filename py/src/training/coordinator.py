from __future__ import annotations

from pathlib import Path
import time

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
from src.training.checkpoint_retention import CheckpointRetention
from src.training.credit_ledger import CreditLedger
from src.training.self_play_group import SelfPlayGroup
from src.training.trainer_group import TrainerGroup, TrainingQuantumResult
from src.training.run_limits import RunLimitMonitor
from src.util.log import log
from src.util.tensorboard import log_scalar


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

    def _ingest_available_games(self) -> None:
        generation = self.ledger.model_generation
        ingestion = self.replay_manager.ingest_available_games(generation)
        self.ledger.add_samples(ingestion.samples_added, generation)

    def _train_quantum(self) -> None:
        paused = self.self_play_group.apply(
            tuple(PausedSelfPlayState() for _ in range(self.self_play_group.worker_count))
        )
        if any(response.kind != 'paused' for response in paused):
            raise RuntimeError('Self-play workers did not enter the paused state.')
        self._ingest_available_games()
        if not self.ledger.can_train_quantum(self.replay_manager.live_samples):
            self._start_self_play()
            return
        result = self.trainer_group.train_quantum(self.replay_manager.description(), self.ledger.progress)
        self.ledger.commit_quantum(result)
        self.latest_completed_model_version = self.ledger.model_generation
        self._record_training_statistics(result)
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

    def _apply_checkpoint_retention(self) -> None:
        self.checkpoint_retention.apply(
            self.ledger.model_generation,
            self.evaluation_manager.required_checkpoint_generations,
        )

    def _detailed_statistics_workers(self) -> int:
        configured = self.game.self_play_configuration.detailed_statistics_workers
        return min(configured, self.self_play_group.worker_count)

    def _record_training_statistics(self, result: TrainingQuantumResult) -> None:
        generation = result.checkpoint.generation
        statistics = result.statistics
        log_scalar('training/policy_loss', statistics.policy_loss, generation)
        log_scalar('training/wdl_loss', statistics.wdl_loss, generation)
        log_scalar('training/total_loss', statistics.total_loss, generation)
        log_scalar('training/gradient_norm', statistics.gradient_norm, generation)
        log_scalar('throughput/replay_rows_per_second', statistics.replay_rows_per_second, generation)
        log_scalar('throughput/training_samples_per_second', statistics.training_samples_per_second, generation)
        log(
            f'Completed generation {generation}: loss={statistics.total_loss:.4f}, '
            f'replay={statistics.replay_rows_per_second:.0f} rows/s, '
            f'training={statistics.training_samples_per_second:.0f} samples/s'
        )

from __future__ import annotations

import time
from pathlib import Path

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
from src.training.checkpoint.retention import CheckpointRetention
from src.training.credit_ledger import CreditLedger
from src.training.reporting import TrainingReporter
from src.training.run_limits import RunLimitMonitor
from src.training.self_play_group import SelfPlayGroup
from src.training.session import TrainingSessionQuantum, create_training_session
from src.util.log import log

INGESTION_SLICE_SECONDS = 10.0
INGESTION_PAUSE_BACKLOG_GAMES = 100


class Coordinator:
    def __init__(self, game: GameImplementation, run_started_at: float) -> None:
        self.game = game
        self.configuration = game.configuration
        self.reporter = TrainingReporter(self.configuration, game.target_layout.auxiliary_heads)
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
            adopt_completed_quantum=not training.progressive_model_sizing.is_progressive,
        )
        self.training_session = create_training_session(
            self.configuration,
            game,
            self.ledger.state.active_checkpoint,
        )
        recovered_checkpoint = self.training_session.recover_published_checkpoint(self.ledger.progress)
        if recovered_checkpoint is not None:
            self.ledger.commit_checkpoint(
                self.ledger.progress.next_generation.completed_optimizer_steps,
                recovered_checkpoint,
            )
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
            maximum_legal_actions=game.state.maximum_legal_action_count,
        )
        self.replay_manager = ReplayManager.open(
            run_path,
            game.state,
            replay_layout,
            training.lifecycle.replay,
            self.ledger.model_generation,
            game.value_discount_per_ply,
            game.terminal_oracle,
            self.resignation_calibrator,
            self.configuration,
            censor_remaining_game_length_on_cut_games=game.censor_remaining_game_length_on_cut_games,
        )
        self.self_play_group = SelfPlayGroup(game)
        self.evaluation_manager = EvaluationManager(self.configuration, self.ledger.state.active_checkpoint)
        self.checkpoint_retention = CheckpointRetention(run_path, training.lifecycle)
        self._apply_checkpoint_retention()
        self.latest_completed_model_version = self.ledger.model_generation
        self.final_stop_reason: str | None = None
        self._backpressure_pause_requested = False
        self._ingestion_pause_requested = False
        self._credit_wait_started_at = time.perf_counter()
        self._completed_games_since_last_quantum: list[IngestedCompletedGame] = []
        self.reporter.record_initial_settings(self.ledger.model_generation)

    def run(self) -> None:
        try:
            if self.training_session.has_pending_quantum:
                self._train_quantum(self_play_started=False)
            self.evaluation_manager.start()
            self._start_self_play()
            while not self.ledger.training_complete:
                self.evaluation_manager.collect_completed_jobs()
                restarted_workers = self.self_play_group.restart_exited_workers(
                    self.ledger.state.active_checkpoint,
                    self._resignation_policy(),
                )
                for worker_id in restarted_workers:
                    log(f'Restarted self-play worker {worker_id} at generation {self.ledger.model_generation}.')
                self.final_stop_reason = self.run_limit_monitor.stop_reason()
                if self.final_stop_reason is not None:
                    break
                if self.ledger.can_train_quantum(self.replay_manager.live_samples):
                    self._train_quantum(self_play_started=True)
                    self.evaluation_manager.schedule_due_jobs(self.ledger.state.active_checkpoint)
                    continue
                self._apply_self_play_backpressure()
                self._ingest_toward_next_quantum()
                self._apply_self_play_backpressure()
                self.final_stop_reason = self.run_limit_monitor.stop_reason()
                if self.final_stop_reason is not None:
                    break
                if not self.ledger.can_train_quantum(self.replay_manager.live_samples):
                    self.evaluation_manager.schedule_due_jobs(self.ledger.state.active_checkpoint)
                    time.sleep(0.1)
                    continue
                self._train_quantum(self_play_started=True)
                self.evaluation_manager.schedule_due_jobs(self.ledger.state.active_checkpoint)
        finally:
            self.evaluation_manager.close()
            self.self_play_group.close()
            self.training_session.close()
            self.replay_manager.close()
            self.game.close()
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

    def _ingest_toward_next_quantum(self) -> None:
        generation = self.ledger.model_generation
        maximum_samples = self.ledger.samples_needed_for_quantum(self.replay_manager.live_samples)
        if maximum_samples == 0:
            return
        if (
            not self._ingestion_pause_requested
            and self.replay_manager.available_game_count >= INGESTION_PAUSE_BACKLOG_GAMES
        ):
            self.self_play_group.request_pause(self._topology_pause_worker_ids())
            self._ingestion_pause_requested = True
        ingestion = self.replay_manager.ingest_available_games(
            generation,
            maximum_samples,
            maximum_elapsed_seconds=INGESTION_SLICE_SECONDS,
        )
        self.ledger.add_samples(ingestion.samples_added, generation)
        self._completed_games_since_last_quantum.extend(ingestion.completed_games)
        if ingestion.games_ingested:
            log(
                f'Ingested {ingestion.games_ingested} games and {ingestion.samples_added} samples in '
                f'{ingestion.elapsed_seconds:.1f}s ({ingestion.samples_per_second:.0f} samples/s).'
            )
            self._record_resignation_diagnostics(generation)
        if (
            self._ingestion_pause_requested
            and self.replay_manager.available_game_count == 0
            and not self.ledger.can_train_quantum(self.replay_manager.live_samples)
        ):
            self._resume_ingestion_paused_workers()

    def _train_quantum(self, self_play_started: bool) -> None:
        credit_wait_seconds = time.perf_counter() - self._credit_wait_started_at
        if self_play_started:
            self.self_play_group.request_pause(self._training_pause_worker_ids())
        outcome = self.training_session.train_quantum(
            TrainingSessionQuantum(
                replay=self.replay_manager.description(),
                progress=self.ledger.progress,
                active_checkpoint=self.ledger.state.active_checkpoint,
                elapsed_seconds=self.evaluation_manager.elapsed_seconds,
            )
        )
        publication = outcome.publication
        self.ledger.commit_checkpoint(publication.completed_optimizer_steps, publication.checkpoint)
        self.latest_completed_model_version = self.ledger.model_generation
        if self.resignation_calibrator is not None:
            self.resignation_calibrator.advance_generation(self.ledger.model_generation)
        checkpoint_activation_started_at = time.perf_counter()
        if self_play_started:
            detailed_workers = self._detailed_statistics_workers()
            desired_states = tuple(
                RunningSelfPlayState(
                    checkpoint=publication.checkpoint,
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
        checkpoint_activation_seconds = time.perf_counter() - checkpoint_activation_started_at
        if self_play_started:
            self._backpressure_pause_requested = False
            self._ingestion_pause_requested = False
            self._apply_self_play_backpressure()
        if self.resignation_calibrator is not None:
            self._record_resignation_diagnostics(self.ledger.model_generation)
        reporting_started_at = time.perf_counter()
        self.reporter.record_training_outcome(
            outcome,
            credit_wait_seconds,
            self.ledger.state,
            self.replay_manager.description(),
            tuple(self._completed_games_since_last_quantum),
        )
        reporting_seconds = time.perf_counter() - reporting_started_at
        log(
            f'Finalized generation {publication.checkpoint.generation}: '
            f'checkpoint-activation={checkpoint_activation_seconds:.1f}s, reporting={reporting_seconds:.1f}s.'
        )
        self._completed_games_since_last_quantum.clear()
        self._apply_checkpoint_retention()
        self._credit_wait_started_at = time.perf_counter()

    def _training_pause_worker_ids(self) -> tuple[int, ...]:
        if self.training_session.pauses_all_self_play_workers:
            return tuple(range(self.self_play_group.worker_count))
        if self._backpressure_pause_requested:
            return ()
        if self._ingestion_pause_requested:
            return ()
        if self._self_play_backpressure_required():
            return tuple(range(self.self_play_group.worker_count))
        return self._topology_pause_worker_ids()

    def _topology_pause_worker_ids(self) -> tuple[int, ...]:
        return self.configuration.training.topology.self_play.node_ids_to_pause_during_training

    def _resume_ingestion_paused_workers(self) -> None:
        state = RunningSelfPlayState(
            checkpoint=self.ledger.state.active_checkpoint,
            resignation_policy=self._resignation_policy(),
        )
        applied = self.self_play_group.apply_to_workers(self._topology_pause_worker_ids(), state)
        if any(response.kind != 'running' for response in applied):
            raise RuntimeError('Self-play workers did not resume after replay ingestion.')
        self._ingestion_pause_requested = False

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
        self.reporter.record_resignation(self.resignation_calibrator.diagnostics(), generation)

    def _apply_checkpoint_retention(self) -> None:
        self.checkpoint_retention.apply(
            self.ledger.model_generation,
            self.evaluation_manager.required_checkpoint_generations,
        )

    def _detailed_statistics_workers(self) -> int:
        configured = self.game.self_play_configuration.detailed_statistics_workers
        return min(configured, self.self_play_group.worker_count)

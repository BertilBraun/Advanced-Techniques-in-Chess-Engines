from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from src.evaluation.manager import EvaluationManager
from src.experiment.base_configuration import initial_generation
from src.experiment.configuration import experiment_configuration_sha256
from src.games.implementation import GameImplementation
from src.replay.batch_loader import build_training_batch
from src.replay.layout import ReplayLayout
from src.replay.manager import IngestedCompletedGame, ReplayManager
from src.replay.store import ReplayStore
from src.search_budget.labeling import ExperimentReplaySampleProvider
from src.search_budget.manager import (
    FailedLabelJobReport,
    GenerationLabelReport,
    SearchBudgetLabelManager,
    SkippedLabelJobReport,
)
from src.search_budget.worker import ConfiguredLabelWorkerRuntimeFactory
from src.self_play.protocol import (
    RunningSelfPlayState,
    StatisticsLevel,
)
from src.self_play.resignation import PublishedResignationPolicy, ResignationCalibrator
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.persistence import load_model
from src.training.checkpoint.retention import CheckpointRetention
from src.training.credit_ledger import CreditLedger
from src.training.initialization_guard import (
    assert_healthy_policy_initialization,
    probe_policy_initialization,
)
from src.training.reporting import ReplayIngestionTelemetry, TrainingReporter
from src.training.run_limits import RunLimitMonitor
from src.training.self_play_group import SelfPlayGroup
from src.training.self_play_health import SelfPlayHealthMonitor
from src.training.session import TrainingSessionQuantum, TrainingSessionResult, create_training_session
from src.util.log import log, warn
from src.util.tensorboard import log_scalar

IDLE_WAIT_SECONDS = 1.0


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
        # Durable replay state is the credit ground truth across callbacks and restarts.
        self.ledger.reconcile_materialized_samples(self.replay_manager.total_materialized_samples())
        label_sample_provider = ExperimentReplaySampleProvider(self.configuration.model_dump_json())
        self.search_budget_label_manager = SearchBudgetLabelManager(
            run_path=run_path,
            configuration_sha256=experiment_configuration_sha256(self.configuration),
            device_ids=training.topology.trainer.ddp_device_ids,
            runtime_factory=ConfiguredLabelWorkerRuntimeFactory(self.configuration.model_dump_json()),
            action_size=game.state.action_size,
            maximum_policy_entries=training.lifecycle.replay.maximum_policy_entries,
            sample_provider=label_sample_provider,
            replay_writer=self.replay_manager.append_labelled_samples,
            initial_first_unstarted_production_generation=self.ledger.model_generation + 1,
            configuration=training.lifecycle.search_budget,
        )
        self._recover_search_budget_label_sources()
        self.self_play_group = SelfPlayGroup(game)
        self.self_play_health = SelfPlayHealthMonitor(self.self_play_group.worker_count)
        self.evaluation_manager = EvaluationManager(self.configuration, self.ledger.state.active_checkpoint)
        self.checkpoint_retention = CheckpointRetention(run_path, training.lifecycle)
        self._apply_checkpoint_retention()
        self.latest_completed_model_version = self.ledger.model_generation
        self.final_stop_reason: str | None = None
        self._backpressure_pause_requested = False
        self._credit_wait_started_at = time.perf_counter()
        self._completed_games_since_last_quantum: list[IngestedCompletedGame] = []
        self._ingest_seconds_since_last_quantum = 0.0
        self._initialization_guard_pending = self.ledger.progress.completed_optimizer_steps == 0
        self.reporter.record_initial_settings(self.ledger.model_generation)

    def run(self) -> None:
        try:
            if self.training_session.has_pending_quantum:
                self._train_quantum(self_play_started=False)
            self.evaluation_manager.start()
            self._start_self_play()
            self.replay_manager.start_materialization()
            while not self.ledger.training_complete:
                self.replay_manager.raise_if_materialization_failed()
                # Appending must not wait for a full quantum of credits, and must not sit behind worker
                # management: sealed shards occupy claim slots, and crediting stalls permanently once all
                # slots are sealed.
                self._append_staged_games()
                self._apply_self_play_backpressure()
                self.evaluation_manager.collect_completed_jobs()
                self._collect_search_budget_label_jobs()
                self._supervise_self_play()
                self.final_stop_reason = self.run_limit_monitor.stop_reason() or self._self_play_stop_reason()
                if self.final_stop_reason is not None:
                    break
                if not self.ledger.has_quantum_credits:
                    self.evaluation_manager.schedule_due_jobs(self.ledger.state.active_checkpoint)
                    time.sleep(IDLE_WAIT_SECONDS)
                    continue
                if not self.ledger.can_train_quantum(self.replay_manager.live_samples):
                    self.evaluation_manager.schedule_due_jobs(self.ledger.state.active_checkpoint)
                    time.sleep(IDLE_WAIT_SECONDS)
                    continue
                self._train_quantum(self_play_started=True)
                self.evaluation_manager.schedule_due_jobs(self.ledger.state.active_checkpoint)
        finally:
            self.evaluation_manager.close()
            self.self_play_group.close()
            self.training_session.close()
            self.search_budget_label_manager.close()
            self.replay_manager.close()
            self.game.close()
            self.ledger.save()

    def _start_self_play(self) -> None:
        checkpoint = self.ledger.state.active_checkpoint
        search_budget = self.search_budget_label_manager.publication_for_starting_generation(checkpoint.generation)
        responses = self.self_play_group.apply(
            tuple(
                RunningSelfPlayState(
                    checkpoint=checkpoint,
                    search_budget=search_budget,
                    resignation_policy=self._resignation_policy(),
                )
                for _ in range(self.self_play_group.worker_count)
            )
        )
        if len(responses) != self.self_play_group.worker_count or any(
            response.kind != 'running' for response in responses
        ):
            raise RuntimeError('Self-play workers did not enter the running state.')
        self._credit_wait_started_at = time.perf_counter()

    def _supervise_self_play(self) -> None:
        supervision = self.self_play_group.supervise(
            self.ledger.state.active_checkpoint,
            self.search_budget_label_manager.publication_for_generation(self.ledger.model_generation),
            self._resignation_policy(),
        )
        for worker_id in supervision.restarted_worker_ids:
            log(f'Restarted self-play worker {worker_id} at generation {self.ledger.model_generation}.')
        for worker_id in supervision.failed_worker_ids:
            warn(f'Self-play worker {worker_id} failed to restart; retrying after backoff.')

    def _self_play_stop_reason(self) -> str | None:
        return self.self_play_health.stop_reason(self.self_play_group.live_worker_count, time.monotonic())

    def _append_staged_games(self) -> None:
        generation = self.ledger.model_generation
        ingestion = self.replay_manager.append_staged_games(generation)
        # Credit is earned strictly after the append and its flush: late credit costs one loop
        # iteration, early credit lets the ledger over-earn against the store.
        self.ledger.reconcile_materialized_samples(self.replay_manager.total_materialized_samples())
        self._completed_games_since_last_quantum.extend(ingestion.completed_games)
        self._ingest_seconds_since_last_quantum += ingestion.elapsed_seconds
        if ingestion.games_ingested:
            log(
                f'Appended {ingestion.games_ingested} staged games and {ingestion.samples_added} samples in '
                f'{ingestion.elapsed_seconds:.1f}s ({ingestion.samples_per_second:.0f} samples/s).'
            )
            self._record_resignation_diagnostics(generation)

    def _run_initialization_guard(self) -> None:
        training = self.configuration.training
        model = load_model(
            self.ledger.state.active_checkpoint.model_path,
            training.initial_model.network,
            torch.device('cpu'),
            self.game.network_dimensions,
            self.game.target_layout.auxiliary_heads,
        )
        description = self.replay_manager.description()
        store = ReplayStore.open(description.path, description.layout, writable=False)
        try:
            batch_size = min(training.trainer.global_batch_size, description.size)
            batch = build_training_batch(
                store,
                self.game.state,
                tuple(range(batch_size)),
                np.zeros(batch_size, dtype=np.int64),
            )
        finally:
            store.close()
        probe = probe_policy_initialization(model, batch.states, batch.policy_legal_action_ids)
        log_scalar('init/policy_logit_std', probe.policy_logit_std, 0)
        log_scalar('init/policy_entropy_ratio', probe.policy_entropy_ratio, 0)
        log(
            f'Initialization guard at generation 0: policy_logit_std={probe.policy_logit_std:.3f}, '
            f'policy_entropy_ratio={probe.policy_entropy_ratio:.3f}.'
        )
        assert_healthy_policy_initialization(probe)

    def _train_quantum(self, self_play_started: bool) -> None:
        if self._initialization_guard_pending:
            self._run_initialization_guard()
            self._initialization_guard_pending = False
        credit_wait_seconds = time.perf_counter() - self._credit_wait_started_at
        source_checkpoint = self.ledger.state.active_checkpoint
        self.replay_manager.ensure_label_source_cohort(source_checkpoint.generation)
        if self_play_started:
            self.self_play_group.request_pause(self._training_pause_worker_ids())
        with self.replay_manager.training_snapshot() as replay:
            outcome = self.training_session.train_quantum(
                TrainingSessionQuantum(
                    replay=replay,
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
            search_budget = self.search_budget_label_manager.publication_for_starting_generation(
                publication.checkpoint.generation
            )
            desired_states = tuple(
                RunningSelfPlayState(
                    checkpoint=publication.checkpoint,
                    search_budget=search_budget,
                    resignation_policy=self._resignation_policy(),
                    completed_generation_statistics=(
                        StatisticsLevel.DETAILED if worker_id < detailed_workers else StatisticsLevel.BASIC
                    ),
                )
                for worker_id in range(self.self_play_group.worker_count)
            )
            responses = self.self_play_group.apply(desired_states)
            spend_residual = sum(
                response.completed_generation_statistics.search_budget_spend_residual
                for response in responses
                if response.kind == 'running' and response.completed_generation_statistics is not None
            )
            log_scalar(
                'search_budget/production/exact_generation_spend_residual',
                spend_residual,
                source_checkpoint.generation,
            )
        checkpoint_activation_seconds = time.perf_counter() - checkpoint_activation_started_at
        if self_play_started:
            self._backpressure_pause_requested = False
            self._apply_self_play_backpressure()
        if self.resignation_calibrator is not None:
            self._record_resignation_diagnostics(self.ledger.model_generation)
        reporting_started_at = time.perf_counter()
        self._report_training_outcome(outcome, credit_wait_seconds)
        reporting_seconds = time.perf_counter() - reporting_started_at
        log(
            f'Finalized generation {publication.checkpoint.generation}: '
            f'checkpoint-activation={checkpoint_activation_seconds:.1f}s, reporting={reporting_seconds:.1f}s.'
        )
        self._apply_checkpoint_retention()
        self._enqueue_search_budget_label_source(source_checkpoint)
        self._credit_wait_started_at = time.perf_counter()

    def _report_training_outcome(self, outcome: TrainingSessionResult, credit_wait_seconds: float) -> None:
        self.reporter.record_training_outcome(
            outcome,
            credit_wait_seconds,
            self.ledger.state,
            self.replay_manager.description(),
            tuple(self._completed_games_since_last_quantum),
            ReplayIngestionTelemetry(
                ingest_seconds=self._ingest_seconds_since_last_quantum,
                inbox_depth=self.replay_manager.inbox_depth,
                staging_depth=self.replay_manager.staging_depth,
                materialization_failures=self.replay_manager.materialization_failures,
                rejection_rate=self.replay_manager.rejection_rate,
            ),
        )
        self._completed_games_since_last_quantum.clear()
        self._ingest_seconds_since_last_quantum = 0.0

    def _training_pause_worker_ids(self) -> tuple[int, ...]:
        if self.training_session.pauses_all_self_play_workers:
            return tuple(range(self.self_play_group.worker_count))
        return self._topology_pause_worker_ids()

    def _topology_pause_worker_ids(self) -> tuple[int, ...]:
        return self.configuration.training.topology.self_play.node_ids_to_pause_during_training

    def _apply_self_play_backpressure(self) -> None:
        required = self._self_play_backpressure_required()
        if required and not self._backpressure_pause_requested:
            self.self_play_group.request_pause(self._topology_pause_worker_ids())
            self._backpressure_pause_requested = True
        elif not required and self._backpressure_pause_requested:
            self._resume_backpressure_paused_workers()

    def _resume_backpressure_paused_workers(self) -> None:
        checkpoint = self.ledger.state.active_checkpoint
        state = RunningSelfPlayState(
            checkpoint=checkpoint,
            search_budget=self.search_budget_label_manager.publication_for_generation(checkpoint.generation),
            resignation_policy=self._resignation_policy(),
        )
        self.self_play_group.apply_to_workers(self._topology_pause_worker_ids(), state)
        self._backpressure_pause_requested = False

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
        required_generations = tuple(
            sorted(
                set(self.evaluation_manager.required_checkpoint_generations)
                | set(self.search_budget_label_manager.required_checkpoint_generations)
            )
        )
        self.checkpoint_retention.apply(
            self.ledger.model_generation,
            required_generations,
        )

    def _collect_search_budget_label_jobs(self) -> None:
        for event in self.search_budget_label_manager.poll():
            match event:
                case GenerationLabelReport():
                    log(
                        f'Finalized search-budget label generation {event.source_generation}: '
                        f'{event.replay_samples_written} replay samples, blend={event.selected_blend} '
                        f'for production generation {event.application_generation}.'
                    )
                    log_scalar(
                        'search_budget/calibration/published_blend',
                        float(event.selected_blend),
                        event.source_generation,
                    )
                case FailedLabelJobReport():
                    warn(f'Search-budget label generation {event.source_generation} failed closed: {event.failure}')
                case SkippedLabelJobReport():
                    log(f'Search-budget label generation {event.source_generation} skipped: {event.reason}.')

    def _recover_search_budget_label_sources(self) -> None:
        active_generation = self.ledger.model_generation
        for source_generation in self.replay_manager.pending_label_source_generations:
            if source_generation >= active_generation:
                continue
            checkpoint = CheckpointReference.load(Path(self.configuration.training.save_path), source_generation)
            self._enqueue_search_budget_label_source(checkpoint)

    def _enqueue_search_budget_label_source(self, checkpoint: CheckpointReference) -> None:
        source_generation = checkpoint.generation
        cohort = self.replay_manager.finalize_label_source_cohort(source_generation)
        baseline_visits = self.game.self_play_configuration.search.baseline_visits.value_at(source_generation)
        enqueue = self.search_budget_label_manager.enqueue_replay_generation(
            source_generation=source_generation,
            label_source_games=cohort.games,
            checkpoint=checkpoint,
            baseline_new_visits=baseline_visits,
            run_seed=self.configuration.training.random_seed,
        )
        self.replay_manager.acknowledge_label_source_cohort(source_generation)
        status = 'queued' if enqueue.accepted else 'skipped'
        log(f'Search-budget label generation {source_generation} {status}.')

    def _detailed_statistics_workers(self) -> int:
        configured = self.game.self_play_configuration.detailed_statistics_workers
        return min(configured, self.self_play_group.worker_count)

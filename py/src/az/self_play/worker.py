from __future__ import annotations

import io
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid5

import torch
import az_go_native as native
from src.az.config.search import (
    ConstantTemperature,
    DirichletRootExploration,
    DisabledRootExploration,
    PlyTemperatureSchedule,
    RootExplorationConfiguration,
    TemperatureConfiguration,
)
from src.az.games.api import GameIdentifier
from src.az.games.go.configuration import (
    GO_PAYLOAD_SCHEMA_VERSION,
)
from src.az.games.go.model import ResidualGoModel
from src.az.games.go.replay_codec import GoReplayCodec
from src.az.games.go.samples import PendingGoSearchSample, finalize_sample, pending_sample_from_native
from src.az.replay.envelope import (
    GameTermination,
    ReplayEnvelope,
    ReplayRecord,
    RootDiagnostics,
    SearchBudgetClass,
    SearchStopReason,
    SearchStrategy,
    SelfPlaySeedLineage,
    derive_self_play_seed_lineage,
)
from src.az.runtime.messages import (
    IpcReplayRecord,
    WorkerFailure,
    WorkerModelRefreshed,
    WorkerReady,
    WorkerRecords,
    WorkerProgress,
    WorkerResourceSample,
    WorkerStopped,
)
from src.az.runtime.telemetry import sample_process_resources
from src.az.runtime.ipc import ByteQueue, StopSignal
from src.az.self_play.go_adapter import GoInferenceBatchBroker
from src.az.self_play.configuration import GoWorkerSpecification
from src.az.self_play.model_refresh import load_newer_model_checkpoint
from src.az.self_play.scheduling import LogicalWorkerGameScheduler
from src.az.training.checkpoints import CheckpointRepository


@dataclass(frozen=True)
class _PendingPosition:
    ply: int
    player_is_black: bool
    sample: PendingGoSearchSample
    actual_simulations: int
    entropy: float
    top_two_margin: float
    seed_lineage: SelfPlaySeedLineage


class _SearchCancelled(Exception):
    pass


def _send(
    message_queue: ByteQueue,
    message: (
        WorkerReady
        | WorkerProgress
        | WorkerResourceSample
        | WorkerRecords
        | WorkerModelRefreshed
        | WorkerStopped
        | WorkerFailure
    ),
) -> None:
    message_queue.put(message.model_dump_json().encode())


def run_go_worker(
    serialized_specification: bytes,
    stop_event: StopSignal,
    message_queue: ByteQueue,
) -> None:
    specification: GoWorkerSpecification | None = None
    completed_games = 0
    emitted_positions = 0
    last_progress = 0.0
    last_resource_sample = 0.0
    try:
        specification = GoWorkerSpecification.model_validate_json(serialized_specification)
        device = torch.device(specification.device)
        repository = CheckpointRepository(
            Path(specification.checkpoint_directory),
            specification.run_id,
            specification.resolved_configuration_sha256,
        )
        model, model_version, checkpoint_id = _load_initial_model(
            specification,
            repository,
            device,
        )
        _send(
            message_queue,
            WorkerReady(
                kind='worker_ready',
                worker_index=specification.worker_index,
                process_id=os.getpid(),
                model_version=model_version,
            ),
        )
        scheduler = LogicalWorkerGameScheduler(
            specification.logical_worker_start_index,
            specification.logical_worker_count,
        )
        maximum_active_games = specification.logical_worker_count * specification.maximum_active_searches_per_worker
        while not stop_event.is_set():
            broker = GoInferenceBatchBroker(
                model=model,
                configuration=specification.game_configuration,
                device=device,
                maximum_batch_size=specification.maximum_batch_size,
                maximum_wait_microseconds=specification.maximum_wait_microseconds,
                maximum_pending_batches=specification.maximum_pending_batches,
                cache_capacity=specification.inference_cache_capacity,
            )
            with broker:
                with ThreadPoolExecutor(max_workers=maximum_active_games) as executor:
                    active: dict[Future[tuple[ReplayRecord, ...] | None], int] = {}
                    for _ in range(maximum_active_games):
                        _submit_game(
                            active,
                            executor,
                            specification,
                            scheduler,
                            checkpoint_id,
                            broker,
                            stop_event,
                        )
                    refresh_requested = False
                    while active:
                        completed, _ = wait(active, timeout=1.0, return_when=FIRST_COMPLETED)
                        for future in completed:
                            active.pop(future)
                            try:
                                records = future.result()
                            except Exception:
                                stop_event.set()
                                raise
                            if records is None:
                                continue
                            _send(
                                message_queue,
                                WorkerRecords(
                                    kind='worker_records',
                                    worker_index=specification.worker_index,
                                    records=tuple(IpcReplayRecord.from_record(record) for record in records),
                                ),
                            )
                            completed_games += 1
                            emitted_positions += len(records)
                        now = time.monotonic()
                        if now - last_progress >= specification.telemetry_write_every_seconds:
                            telemetry = broker.take_telemetry()
                            _send(
                                message_queue,
                                WorkerProgress(
                                    kind='worker_progress',
                                    worker_index=specification.worker_index,
                                    completed_games_total=completed_games,
                                    emitted_positions_total=emitted_positions,
                                    model_version=model_version,
                                    interval_inference_batches=telemetry.batches,
                                    interval_inference_requests=telemetry.requests,
                                    interval_maximum_inference_batch_size=telemetry.maximum_batch_size,
                                    interval_total_inference_wait_microseconds=telemetry.total_wait_microseconds,
                                    interval_inference_cache_hits=telemetry.cache_hits,
                                    monotonic_seconds=now,
                                ),
                            )
                            last_progress = now
                        if now - last_resource_sample >= specification.resource_sample_every_seconds:
                            resources = sample_process_resources()
                            _send(
                                message_queue,
                                WorkerResourceSample(
                                    kind='worker_resource_sample',
                                    worker_index=specification.worker_index,
                                    monotonic_seconds=resources.monotonic_seconds,
                                    cpu_time_seconds=resources.cpu_time_seconds,
                                    device_memory_bytes=(
                                        torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
                                    ),
                                ),
                            )
                            last_resource_sample = now
                        available_version = repository.current_model_version()
                        refresh_requested = (
                            refresh_requested or available_version is not None and available_version > model_version
                        )
                        if stop_event.is_set() or refresh_requested:
                            continue
                        for _ in completed:
                            _submit_game(
                                active,
                                executor,
                                specification,
                                scheduler,
                                checkpoint_id,
                                broker,
                                stop_event,
                            )
            refreshed = _refresh_model(
                specification,
                repository,
                device,
                model_version,
            )
            if refreshed is not None:
                refreshed_model, refreshed_version, refreshed_id = refreshed
                previous_model_version = model_version
                model = refreshed_model
                model_version = refreshed_version
                checkpoint_id = refreshed_id
                _send(
                    message_queue,
                    WorkerModelRefreshed(
                        kind='worker_model_refreshed',
                        worker_index=specification.worker_index,
                        previous_model_version=previous_model_version,
                        model_version=model_version,
                        checkpoint_id=checkpoint_id,
                    ),
                )
        _send(
            message_queue,
            WorkerStopped(
                kind='worker_stopped',
                worker_index=specification.worker_index,
                completed_games=completed_games,
                emitted_positions=emitted_positions,
            ),
        )
    except Exception as error:
        worker_index = 0 if specification is None else specification.worker_index
        _send(
            message_queue,
            WorkerFailure(
                kind='worker_failure',
                worker_index=worker_index,
                error_type=type(error).__name__,
                message=str(error),
            ),
        )
        raise


def _submit_game(
    active: dict[Future[tuple[ReplayRecord, ...] | None], int],
    executor: ThreadPoolExecutor,
    specification: GoWorkerSpecification,
    scheduler: LogicalWorkerGameScheduler,
    checkpoint_id: str,
    broker: GoInferenceBatchBroker,
    stop_event: StopSignal,
) -> None:
    scheduled = scheduler.next_game()
    future = executor.submit(
        _play_game,
        specification,
        scheduled.logical_worker_index,
        scheduled.game_index,
        checkpoint_id,
        broker,
        stop_event,
    )
    active[future] = scheduled.logical_worker_index


def _new_model(
    specification: GoWorkerSpecification,
    device: torch.device,
) -> ResidualGoModel:
    with torch.random.fork_rng(devices=()):
        torch.manual_seed(specification.model_initialization_seed)
        model = ResidualGoModel(
            specification.game_configuration,
            specification.model_configuration,
        )
    return model.to(device).eval()


def _load_initial_model(
    specification: GoWorkerSpecification,
    repository: CheckpointRepository,
    device: torch.device,
) -> tuple[ResidualGoModel, int, str]:
    model = _new_model(specification, device)
    if repository.current_model_version() is None:
        return model, 0, 'initial-model'
    checkpoint = repository.load_current_model()
    model.load_state_dict(torch.load(io.BytesIO(checkpoint.model_artifact), map_location=device, weights_only=True))
    return (
        model,
        checkpoint.manifest.model_version,
        checkpoint.manifest.checkpoint_id.hex,
    )


def _refresh_model(
    specification: GoWorkerSpecification,
    repository: CheckpointRepository,
    device: torch.device,
    current_version: int,
) -> tuple[ResidualGoModel, int, str] | None:
    checkpoint = load_newer_model_checkpoint(repository, current_version)
    if checkpoint is None:
        return None
    model = _new_model(specification, device)
    model.load_state_dict(
        torch.load(
            io.BytesIO(checkpoint.model_artifact),
            map_location=device,
            weights_only=True,
        )
    )
    return model, checkpoint.manifest.model_version, checkpoint.manifest.checkpoint_id.hex


def _play_game(
    specification: GoWorkerSpecification,
    logical_worker_index: int,
    game_index: int,
    checkpoint_id: str,
    broker: GoInferenceBatchBroker,
    stop_event: StopSignal,
) -> tuple[ReplayRecord, ...] | None:
    game_configuration = specification.game_configuration
    rules = native.GoRules(
        game_configuration.board_size,
        game_configuration.komi_half_points,
        game_configuration.safety_ply_cap,
        game_configuration.history_length,
    )
    state = native.GoState(rules)
    game_id = uuid5(
        specification.run_id,
        f'{specification.process_index}:{logical_worker_index}:{game_index}',
    )
    pending: list[_PendingPosition] = []
    while not state.is_terminal:
        if stop_event.is_set():
            return None
        ply = state.ply
        lineage = derive_self_play_seed_lineage(
            specification.root_seed,
            specification.process_index,
            logical_worker_index,
            game_index,
            ply,
        )
        encoding = state.canonical_encoding()
        legal_actions = tuple(state.legal_actions())
        search_configuration = native.FixedPuctConfiguration(
            simulation_cap=specification.search.simulation_cap,
            exploration_constant=specification.search.exploration_constant,
            backup_discount=specification.search.backup_discount,
            no_visited_child_value=specification.search.no_visited_child_value,
            action_temperature=_temperature(specification.search.temperature, ply),
            root_noise_seed=lineage.root_noise_seed,
            action_sampling_seed=lineage.action_sampling_seed,
            root_noise=_root_noise(specification.search.root_exploration),
            tree_reuse=False,
        )

        def evaluate(request: native.GoInferenceRequest) -> native.InferenceResult:
            if stop_event.is_set():
                raise _SearchCancelled
            return broker.evaluate(request)

        try:
            result = native.search_go_fixed(state, evaluate, search_configuration)
        except _SearchCancelled:
            return None
        if stop_event.is_set():
            return None
        if result.selected_action is None:
            raise AssertionError('A nonterminal Go search must select an action.')
        pending.append(
            _PendingPosition(
                ply=ply,
                player_is_black=state.current_player == native.Player.BLACK,
                sample=pending_sample_from_native(
                    encoding,
                    legal_actions,
                    result,
                    game_configuration,
                ),
                actual_simulations=result.telemetry.actual_simulations,
                entropy=result.telemetry.root_entropy,
                top_two_margin=result.telemetry.top_two_visit_margin,
                seed_lineage=lineage,
            )
        )
        state.apply(result.selected_action)
    terminal_result = state.terminal_result()
    termination = (
        GameTermination.SAFETY_PLY_CAP
        if terminal_result.reason == native.TerminationReason.SAFETY_PLY_CAP
        else GameTermination.TWO_CONSECUTIVE_PASSES
    )
    codec = GoReplayCodec(game_configuration, GO_PAYLOAD_SCHEMA_VERSION)
    records: list[ReplayRecord] = []
    for position in pending:
        value_target = _value_target(position.player_is_black, terminal_result.winner)
        sample = finalize_sample(
            position.sample,
            value_target,
            specification.value_target_weight,
            termination,
        )
        sample_id = uuid5(game_id, f'sample:{position.ply}')
        envelope = ReplayEnvelope(
            run_id=specification.run_id,
            game_identifier=GameIdentifier.GO,
            payload_schema_version=GO_PAYLOAD_SCHEMA_VERSION,
            sample_id=sample_id,
            game_id=game_id,
            seed_lineage=position.seed_lineage,
            created_at=datetime.now(timezone.utc),
            ply=position.ply,
            checkpoint_id=checkpoint_id,
            search_strategy=SearchStrategy.FIXED,
            budget_class=SearchBudgetClass.FIXED,
            configured_simulation_cap=specification.search.simulation_cap,
            actual_simulations=position.actual_simulations,
            stop_reason=SearchStopReason.FULL_BUDGET,
            policy_target_eligible=position.sample.policy_weight > 0,
            policy_target_weight=position.sample.policy_weight,
            value_target_eligible=termination is not GameTermination.SAFETY_PLY_CAP,
            value_target_weight=sample.value_weight,
            root_diagnostics=RootDiagnostics(
                visit_count=position.actual_simulations,
                entropy=position.entropy,
                top_two_margin=position.top_two_margin,
                prefix_full_policy_disagreement=None,
                prefix_full_value_disagreement=None,
            ),
            termination=termination,
            replay_credit_id=uuid5(sample_id, 'replay-credit'),
        )
        records.append(ReplayRecord(envelope=envelope, payload=codec.encode(sample)))
    return tuple(records)


def _value_target(player_is_black: bool, winner: native.Player | None) -> float:
    if winner is None:
        return 0.0
    winner_is_black = winner == native.Player.BLACK
    return 1.0 if winner_is_black == player_is_black else -1.0


def _temperature(configuration: TemperatureConfiguration, ply: int) -> float:
    match configuration:
        case ConstantTemperature(temperature=temperature):
            return temperature
        case PlyTemperatureSchedule(stages=stages, final_temperature=final):
            for stage in stages:
                if ply < stage.maximum_ply_exclusive:
                    return stage.temperature
            return final


def _root_noise(
    configuration: RootExplorationConfiguration,
) -> native.RootNoiseConfiguration:
    match configuration:
        case DisabledRootExploration():
            return native.RootNoiseConfiguration(False, 1.0, 0.0)
        case DirichletRootExploration(alpha=alpha, exploration_fraction=fraction):
            return native.RootNoiseConfiguration(True, alpha, fraction)

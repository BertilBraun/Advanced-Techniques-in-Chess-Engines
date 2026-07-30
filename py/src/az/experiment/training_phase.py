from __future__ import annotations

import multiprocessing
import queue
import time
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Protocol
from uuid import UUID, uuid5

import torch
import torch.distributed as distributed

from src.az.config.model import FixedModelSchedule
from src.az.config.root import ResolvedRunConfiguration
from src.az.config.seeds import ModelInitializationSeedCoordinates, SeedPurpose, derive_seed
from src.az.config.serialization import load_resolved_configuration
from src.az.evaluation.checkpoints import EvaluationModelArtifactRepository
from src.az.experiment.lifecycle import (
    ExperimentPhase,
    ExperimentRunRepository,
    ExperimentRunState,
    RunArtifact,
    RunArtifactKind,
    require_exact_artifact_files,
)
from src.az.experiment.phase_support import (
    ScheduledCheckpointClaim,
    begin_phase,
    complete_phase,
    interrupt_phase,
)
from src.az.experiment.artifact_retention import apply_checkpoint_retention
from src.az.experiment.environment import inspect_hardware
from src.az.experiment.commit_journal import ReplayCommitJournal
from src.az.games.api import GameIdentifier
from src.az.games.go.configuration import (
    GoObjectiveConfiguration,
)
from src.az.games.go.module import create_go_training_module
from src.az.replay.credits import ReplayCreditJournal, ReplayCreditState
from src.az.replay.envelope import ReplayRecord
from src.az.replay.storage import ReplayShardStorage, ShardMetadata
from src.az.runtime.factory import RuntimeBuildEnvironment, build_runtime_plan
from src.az.runtime.orchestrator import RuntimeOrchestrator
from src.az.self_play.worker import run_go_worker
from src.az.training.checkpoints import (
    CheckpointManifest,
    CheckpointPointer,
    CheckpointRepository,
    DistributedCheckpointManifest,
)
from src.az.training.distributed import TrainingRank
from src.az.training.distributed import DistributedBackend
from src.az.training.trainer import CreditTrainer


class _ProcessSignal(Protocol):
    def is_set(self) -> bool: ...

    def set(self) -> None: ...

    def wait(self) -> None: ...


class _ProcessLock(Protocol):
    def __enter__(self) -> bool: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class _ByteQueue(Protocol):
    def put(self, contents: bytes) -> None: ...


def run_training_window(repository: ExperimentRunRepository) -> ExperimentRunState:
    state = begin_phase(repository, ExperimentPhase.TRAINING_RUN)
    configuration = load_resolved_configuration(repository.configuration_path)
    storage = _replay_storage(repository, configuration)
    commit_journal = ReplayCommitJournal((repository.directory / 'replay-commits.azc').resolve())
    commit_journal.commit(tuple(storage.records()))
    checkpoint_directory = (repository.directory / 'checkpoints').resolve()
    checkpoint_repository = CheckpointRepository(
        checkpoint_directory,
        state.run_id,
        state.resolved_configuration_sha256,
    )
    allow_cpu_smoke = configuration.hardware.profile_name == 'local-cpu-smoke'
    if allow_cpu_smoke:
        torch.set_num_threads(1)
    trainer = (
        _create_trainer(configuration, storage, checkpoint_repository, torch.device('cpu'), 0, 1)
        if allow_cpu_smoke
        else None
    )
    model_artifacts = EvaluationModelArtifactRepository((repository.directory / 'evaluation-models').resolve())
    claims_directory = repository.directory / 'checkpoint-claims'
    claims_directory.mkdir(exist_ok=True)
    observed_hardware = inspect_hardware(repository.directory)
    plan = build_runtime_plan(
        configuration,
        RuntimeBuildEnvironment(
            run_id=state.run_id,
            resolved_configuration_sha256=state.resolved_configuration_sha256,
            output_directory=repository.directory,
            checkpoint_directory=checkpoint_directory,
            startup_timeout_seconds=30,
            shutdown_grace_seconds=10,
            visible_cuda_models=tuple(torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())),
            logical_cpu_count=observed_hardware.logical_cpu_count,
            ram_gib=observed_hardware.ram_gib,
            free_disk_gib=observed_hardware.free_disk_gib,
            allow_cpu_smoke=allow_cpu_smoke,
            logical_worker_next_game_indices=commit_journal.next_game_indices(
                len(configuration.topology.self_play.device_ids) * configuration.topology.self_play_workers_per_device
            ),
        ),
    )
    already_elapsed = state.self_play_elapsed_seconds or 0
    remaining_seconds = max(0.001, configuration.experiment.duration_seconds - already_elapsed)
    next_sequence = max((shard.sequence for shard in storage.shards()), default=-1) + 1
    retained_checkpoint_manifest_sha256: str | None = None
    context = multiprocessing.get_context(
        'forkserver' if 'forkserver' in multiprocessing.get_all_start_methods() else 'spawn'
    )
    trainer_stop = context.Event()
    trainer_start = context.Event()
    replay_lock = context.Lock()
    trainer_errors = context.Queue()
    trainer_ready = context.Queue()
    distributed_init_path = (repository.directory / 'distributed-init').resolve()
    if distributed_init_path.exists():
        distributed_init_path.unlink()
    trainer_processes = (
        ()
        if allow_cpu_smoke
        else tuple(
            context.Process(
                target=_run_distributed_trainer,
                args=(
                    configuration.model_dump_json().encode(),
                    str(repository.directory),
                    str(state.run_id),
                    state.resolved_configuration_sha256,
                    rank,
                    len(configuration.topology.trainer.device_ids),
                    configuration.topology.trainer.device_ids[rank],
                    str(distributed_init_path),
                    trainer_stop,
                    trainer_start,
                    replay_lock,
                    trainer_errors,
                    trainer_ready,
                ),
                name=f'az-trainer-{rank}',
            )
            for rank in range(len(configuration.topology.trainer.device_ids))
        )
    )
    for process in trainer_processes:
        process.start()

    def shutdown_trainers() -> tuple[str, ...]:
        trainer_start.set()
        trainer_stop.set()
        for process in trainer_processes:
            process.join(timeout=30)
        for process in trainer_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        failures: list[str] = []
        while True:
            try:
                failures.append(trainer_errors.get_nowait().decode())
            except queue.Empty:
                break
        failures.extend(
            f'exit code {process.exitcode}' for process in trainer_processes if process.exitcode not in (0, None)
        )
        trainer_errors.close()
        trainer_errors.join_thread()
        trainer_ready.close()
        trainer_ready.join_thread()
        return tuple(failures)

    ready_ranks: set[int] = set()
    while len(ready_ranks) < len(trainer_processes):
        try:
            ready_ranks.add(int(trainer_ready.get(timeout=30).decode()))
        except queue.Empty as error:
            shutdown_trainers()
            raise TimeoutError('Distributed trainer ranks did not become ready.') from error

    def publish(records: tuple[ReplayRecord, ...]) -> ShardMetadata:
        nonlocal next_sequence
        with replay_lock:
            metadata = storage.publish(next_sequence, records)
            commit_journal.commit(records)
        next_sequence += 1
        return metadata

    def claim_crossed_checkpoints(current_window_elapsed: float) -> None:
        cumulative_elapsed = already_elapsed + current_window_elapsed
        if not checkpoint_repository.has_current():
            return
        for requested in configuration.experiment.checkpoint_elapsed_seconds:
            path = claims_directory / f'elapsed-{requested:010d}.json'
            if requested > cumulative_elapsed or path.exists():
                continue
            candidate = model_artifacts.claim(checkpoint_repository.load_current_model())
            claim = ScheduledCheckpointClaim(
                run_id=state.run_id,
                resolved_configuration_sha256=state.resolved_configuration_sha256,
                requested_elapsed_seconds=requested,
                published_elapsed_seconds=cumulative_elapsed,
                candidate=candidate,
            )
            path.write_text(claim.model_dump_json(indent=2) + '\n', encoding='utf-8', newline='\n')

    def apply_retention_if_advanced() -> None:
        nonlocal retained_checkpoint_manifest_sha256
        if not checkpoint_repository.has_current():
            return
        with replay_lock:
            pointer = CheckpointPointer.model_validate_json(checkpoint_repository.pointer_path.read_bytes())
            if pointer.manifest_sha256 == retained_checkpoint_manifest_sha256:
                return
            apply_checkpoint_retention(
                checkpoint_directory,
                configuration.retention,
            )
            retained_checkpoint_manifest_sha256 = pointer.manifest_sha256

    def runtime_tick(current_window_elapsed: float) -> None:
        if trainer is not None:
            _train_one_available(trainer, storage, configuration, repository)
        apply_retention_if_advanced()
        claim_crossed_checkpoints(current_window_elapsed)

    try:
        result = RuntimeOrchestrator(
            worker_entrypoint=run_go_worker,
            worker_specifications=tuple(
                specification.model_dump_json().encode() for specification in plan.worker_specifications
            ),
            wall_clock_seconds=remaining_seconds,
            startup_timeout_seconds=plan.startup_timeout_seconds,
            shutdown_grace_seconds=plan.shutdown_grace_seconds,
            start_method=context.get_start_method(),
            replay_publisher=publish,
            topology=plan.topology,
            games_per_shard=plan.games_per_shard,
            telemetry_path=plan.telemetry_path,
            telemetry_write_every_seconds=plan.telemetry_write_every_seconds,
            external_stop_requested=repository.stop_requested,
            runtime_tick=runtime_tick,
            experiment_started=trainer_start.set,
            elapsed_offset_seconds=already_elapsed,
        ).run()
    finally:
        failures = shutdown_trainers()
        if failures:
            raise RuntimeError(f'Distributed trainer failed: {failures}.')
    elapsed = already_elapsed + min(result.elapsed_seconds, remaining_seconds)
    apply_retention_if_advanced()
    claim_crossed_checkpoints(min(result.elapsed_seconds, remaining_seconds))

    def immutable_evidence(include_final_checkpoint: bool) -> tuple[RunArtifact, ...]:
        claim_paths = tuple(
            claims_directory / f'elapsed-{requested:010d}.json'
            for requested in configuration.experiment.checkpoint_elapsed_seconds
            if (claims_directory / f'elapsed-{requested:010d}.json').is_file()
        )
        require_exact_artifact_files(claims_directory, '*.json', claim_paths)
        claims = tuple(ScheduledCheckpointClaim.model_validate_json(path.read_bytes()) for path in claim_paths)
        evaluation_model_paths = tuple(model_artifacts.path(claim.candidate) for claim in claims)
        require_exact_artifact_files(
            repository.directory / 'evaluation-models',
            '*.pt',
            evaluation_model_paths,
        )
        trace_paths = tuple(
            path
            for sample_id in sorted(commit_journal.sample_ids, key=lambda identity: identity.hex)
            if (
                path := repository.directory / 'search-traces' / f'trace-{uuid5(sample_id, "search-trace").hex}.json'
            ).is_file()
        )
        require_exact_artifact_files(
            repository.directory / 'search-traces',
            'trace-*.json',
            trace_paths,
        )
        checkpoint_artifacts = (
            _checkpoint_artifacts(repository, checkpoint_repository) if include_final_checkpoint else ()
        )
        return (
            *checkpoint_artifacts,
            *tuple(repository.artifact(RunArtifactKind.CHECKPOINT_CLAIM, path) for path in claim_paths),
            *tuple(repository.artifact(RunArtifactKind.EVALUATION_MODEL, path) for path in evaluation_model_paths),
            *tuple(repository.artifact(RunArtifactKind.SEARCH_TRACE, path) for path in trace_paths),
        )

    if repository.stop_requested() and elapsed < configuration.experiment.duration_seconds:
        return interrupt_phase(repository, state, elapsed, immutable_evidence(False))
    if not checkpoint_repository.has_current():
        raise RuntimeError('Training window ended without a publishable model checkpoint.')
    missing_claims = tuple(
        requested
        for requested in configuration.experiment.checkpoint_elapsed_seconds
        if not (claims_directory / f'elapsed-{requested:010d}.json').is_file()
    )
    if missing_claims:
        raise RuntimeError(f'Training window ended without scheduled checkpoint claims: {missing_claims}.')
    immutable_artifacts = immutable_evidence(True)
    all_artifacts = (
        *immutable_artifacts,
        repository.artifact(RunArtifactKind.REPLAY_COMMIT_JOURNAL, commit_journal.path),
        repository.artifact(RunArtifactKind.RUNTIME_TELEMETRY, plan.telemetry_path),
    )
    if repository.stop_requested():
        return repository.complete_training_at_stop(
            state,
            all_artifacts,
            configuration.experiment.duration_seconds,
            elapsed,
        )
    return complete_phase(
        repository,
        state,
        ExperimentPhase.TRAINING_RUN,
        all_artifacts,
        self_play_elapsed_seconds=float(configuration.experiment.duration_seconds),
        checkpoint_published_elapsed_seconds=elapsed,
    )


def _create_trainer(
    configuration: ResolvedRunConfiguration,
    storage: ReplayShardStorage,
    checkpoint_repository: CheckpointRepository,
    device: torch.device,
    rank: int,
    world_size: int,
) -> CreditTrainer:
    match configuration.model.schedule:
        case FixedModelSchedule(architecture=architecture):
            pass
        case _:
            raise ValueError('The current Go trainer requires a fixed model schedule.')
    match configuration.training.objective:
        case GoObjectiveConfiguration() as objective:
            pass
        case _:
            raise ValueError('The Go trainer requires the Go policy/value objective.')
    model_seed = derive_seed(
        configuration.experiment.root_seed,
        ModelInitializationSeedCoordinates(
            purpose=SeedPurpose.MODEL_INITIALIZATION,
            model_stage=0,
        ),
    )
    module = create_go_training_module(
        game_configuration=configuration.game,
        model_configuration=architecture,
        objective_configuration=objective,
        payload_schema_version=configuration.replay.payload_schema_version,
        device=device,
        model_initialization_seed=model_seed,
    )
    return CreditTrainer(
        game_module=module,
        replay_storage=storage,
        checkpoint_repository=checkpoint_repository,
        training_configuration=configuration.training,
        credit_configuration=configuration.replay.credits,
        root_seed=configuration.experiment.root_seed,
        rank=TrainingRank(
            rank=rank,
            world_size=world_size,
            device=device,
            backend=DistributedBackend.GLOO if device.type == 'cpu' else DistributedBackend.NCCL,
        ),
        run_determinism_mode=configuration.experiment.manifest_policy.determinism_mode,
    )


def _train_one_available(
    trainer: CreditTrainer,
    storage: ReplayShardStorage,
    configuration: ResolvedRunConfiguration,
    repository: ExperimentRunRepository,
) -> bool:
    if (
        trainer.credit_state.completed_optimizer_steps >= configuration.training.maximum_optimizer_steps
        or repository.stop_requested()
    ):
        return False
    reconciled = trainer.credit_state.reconcile(
        storage.credit_journal.snapshot,
        Decimal(str(configuration.replay.credits.target_reuse)),
    )
    remaining_steps = configuration.training.maximum_optimizer_steps - reconciled.completed_optimizer_steps
    quantum_steps = min(configuration.replay.credits.optimizer_steps_per_quantum, remaining_steps)
    required = Decimal(quantum_steps * configuration.training.global_batch_size)
    if (
        reconciled.credited_unique_positions < configuration.replay.credits.minimum_positions_before_training
        or reconciled.available_position_credits < required
    ):
        return False
    trainer.train_quantum()
    return True


def _run_distributed_trainer(
    serialized_configuration: bytes,
    run_directory_text: str,
    run_id_text: str,
    configuration_sha256: str,
    rank: int,
    world_size: int,
    device_index: int,
    initialization_path_text: str,
    stop_signal: _ProcessSignal,
    start_signal: _ProcessSignal,
    replay_lock: _ProcessLock,
    error_queue: _ByteQueue,
    ready_queue: _ByteQueue,
) -> None:
    try:
        configuration = ResolvedRunConfiguration.model_validate_json(serialized_configuration)
        run_directory = Path(run_directory_text).resolve()
        device = torch.device(f'cuda:{device_index}')
        torch.cuda.set_device(device)
        distributed.init_process_group(
            backend=DistributedBackend.NCCL.value,
            init_method=Path(initialization_path_text).as_uri(),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=20),
        )
        ready_queue.put(str(rank).encode())
        start_signal.wait()
        checkpoint_repository = CheckpointRepository(
            (run_directory / 'checkpoints').resolve(),
            UUID(run_id_text),
            configuration_sha256,
        )
        while True:
            continue_training = torch.tensor(int(not stop_signal.is_set()), device=device, dtype=torch.int32)
            distributed.all_reduce(continue_training, op=distributed.ReduceOp.MIN)
            if int(continue_training.item()) == 0:
                break
            if rank == 0:
                replay_lock.__enter__()
            distributed.barrier()
            try:
                storage = _replay_storage(
                    ExperimentRunRepository(run_directory),
                    configuration,
                )
                credit_state = (
                    checkpoint_repository.load_distributed(rank).rank.state.replay_credits
                    if checkpoint_repository.has_current()
                    else ReplayCreditState.initial()
                )
                reconciled = credit_state.reconcile(
                    storage.credit_journal.snapshot,
                    Decimal(str(configuration.replay.credits.target_reuse)),
                )
                remaining_steps = configuration.training.maximum_optimizer_steps - reconciled.completed_optimizer_steps
                quantum_steps = min(configuration.replay.credits.optimizer_steps_per_quantum, remaining_steps)
                enough_credits = (
                    remaining_steps > 0
                    and reconciled.credited_unique_positions
                    >= configuration.replay.credits.minimum_positions_before_training
                    and reconciled.available_position_credits
                    >= Decimal(quantum_steps * configuration.training.global_batch_size)
                )
                readiness = torch.tensor(int(enough_credits), device=device, dtype=torch.int32)
                distributed.all_reduce(readiness, op=distributed.ReduceOp.MIN)
                if int(readiness.item()) == 1:
                    trainer = _create_trainer(
                        configuration,
                        storage,
                        checkpoint_repository,
                        device,
                        rank,
                        world_size,
                    )
                    trainer.train_quantum()
            finally:
                distributed.barrier()
                if rank == 0:
                    replay_lock.__exit__(None, None, None)
            if not enough_credits:
                time.sleep(0.05)
    except Exception as error:
        error_queue.put(f'rank {rank}: {type(error).__name__}: {error}'.encode())
        stop_signal.set()
        raise
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def _replay_storage(
    repository: ExperimentRunRepository,
    configuration: ResolvedRunConfiguration,
) -> ReplayShardStorage:
    return ReplayShardStorage(
        directory=(repository.directory / 'replay').resolve(),
        maximum_positions_per_shard=configuration.replay.maximum_positions_per_shard,
        capacity_positions=configuration.replay.capacity_positions,
        game_identifier=GameIdentifier.GO,
        payload_schema_version=configuration.replay.payload_schema_version,
        compression=configuration.replay.compression,
        credit_journal=ReplayCreditJournal(repository.directory / 'replay-credits.azc'),
    )


def _checkpoint_artifacts(
    repository: ExperimentRunRepository,
    checkpoints: CheckpointRepository,
) -> tuple[RunArtifact, ...]:
    if not checkpoints.has_current():
        return ()
    pointer = CheckpointPointer.model_validate_json(checkpoints.pointer_path.read_bytes())
    checkpoint_directory = checkpoints.pointer_path.parent / pointer.checkpoint_directory
    if pointer.checkpoint_directory.startswith('distributed-'):
        manifest_path = checkpoint_directory / 'distributed-manifest.json'
        manifest = DistributedCheckpointManifest.model_validate_json(manifest_path.read_bytes())
        expected_paths = (
            manifest_path,
            checkpoint_directory / manifest.model.filename,
            checkpoint_directory / manifest.optimizer.filename,
            checkpoint_directory / manifest.gradient_scaler.filename,
            *tuple(
                checkpoint_directory / f'rank-{rank.rank:05d}' / artifact.filename
                for rank in manifest.ranks
                for artifact in (rank.torch_random_state, rank.cuda_random_stream)
            ),
        )
    else:
        manifest_path = checkpoint_directory / 'manifest.json'
        manifest = CheckpointManifest.model_validate_json(manifest_path.read_bytes())
        expected_paths = (
            manifest_path,
            *tuple(
                checkpoint_directory / artifact.filename
                for artifact in (
                    manifest.model,
                    manifest.optimizer,
                    manifest.torch_random_state,
                    manifest.cuda_random_stream,
                    manifest.gradient_scaler,
                )
            ),
        )
    actual_paths = frozenset(path.resolve() for path in checkpoint_directory.rglob('*') if path.is_file())
    if actual_paths != frozenset(path.resolve() for path in expected_paths):
        raise ValueError('Final checkpoint contains missing or unregistered files.')
    return (
        repository.artifact(RunArtifactKind.CHECKPOINT_POINTER, checkpoints.pointer_path),
        *tuple(repository.artifact(RunArtifactKind.CHECKPOINT, path) for path in expected_paths),
    )

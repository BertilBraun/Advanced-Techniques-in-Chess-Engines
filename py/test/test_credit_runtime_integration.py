from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

sys.modules.setdefault('GPUtil', ModuleType('GPUtil'))

import src.cluster.CreditEvaluationScheduler as scheduler_module
from src.Encoding import C, H, W
from src.cluster.CommanderProcess import CommanderProcess
from src.cluster.CreditEvaluationScheduler import (
    CreditEvaluationScheduler,
    CreditEvaluationStatus,
)
from src.cluster.SelfPlayProcess import SelfPlayProcess
from src.experiment.cost_accounting import CostCurrency
from src.experiment.run_configuration import (
    ApprovalRecord,
    ResolvedHardware,
    RunManifest,
    configuration_sha256,
    load_run_configuration,
)
from src.self_play.SelfPlayDataset import ReplaySampleMetadata, SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.self_play.model_refresh import SearchScheduleState
from src.self_play.value_target import ReplayValueTarget, TerminationReason
from src.settings import TRAINING_ARGS
from src.train.CreditPublication import (
    CreditPublicationManifest,
    PublicationValidationScope,
    file_sha256,
    load_credit_publication_pointer,
    publication_manifest_path,
    write_credit_publication_manifest,
    create_credit_publication_manifest,
)
from src.train.CreditTrainingLedger import CreditTrainingLedger
from src.train.RollingReplayBuffer import (
    ReplayQuantumRequest,
    RollingReplayBuffer,
    commit_replay_shard,
    decode_rank_quantum,
)
from src.train.TrainingArgs import CreditTrainingParams, TrainingArgs
from src.util.communication import (
    Communication,
    LATEST_SELF_PLAY_MODEL_VERSION,
    self_play_model_refreshed_message,
)
from src.util.save_paths import CheckpointManifest


CONFIGURATION_PATH = Path(__file__).parents[1] / 'configs' / 'chess-continuation-4x4070-pilot.json'
GLOBAL_BATCH_SIZE = 4
WORLD_SIZE = 4
OPTIMIZER_STEPS_PER_QUANTUM = 50
REPLAY_RATIO = Decimal(4)
UNIQUE_SAMPLES_PER_QUANTUM = 50


EvaluationTarget = Callable[[int, TrainingArgs, int, int | None], None]


class _InterruptibleEvaluationProcess:
    created: list[_InterruptibleEvaluationProcess] = []

    def __init__(
        self,
        target: EvaluationTarget,
        args: tuple[int, TrainingArgs, int, int],
        name: str,
    ) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.pid: int | None = None
        self.exitcode: int | None = None
        self.alive = False
        self.terminated = False
        self.__class__.created.append(self)

    def start(self) -> None:
        self.pid = 40_000 + len(self.__class__.created)
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False
        self.exitcode = -15


@dataclass
class _RefreshProbe:
    dataset: list[int] = field(default_factory=lambda: [1, 2, 3])
    roots: list[int] = field(default_factory=lambda: [10, 11])
    search_schedule_state: SearchScheduleState | None = None
    refreshes: list[tuple[int, Path]] = field(default_factory=list)

    def search_schedule(self, schedule_version: int) -> SearchScheduleState:
        return SearchScheduleState(
            schedule_version=schedule_version,
            num_parallel_searches=2,
            num_full_searches=64,
            num_fast_searches=16,
            endgame_shortcut_strength=0,
        )

    def update_search_schedule(self, schedule: SearchScheduleState) -> None:
        self.search_schedule_state = schedule

    def refresh_model(self, model_version: int, model_path: Path) -> None:
        self.refreshes.append((model_version, model_path))


def _credit_parameters() -> CreditTrainingParams:
    return CreditTrainingParams(
        replay_ratio=REPLAY_RATIO,
        optimizer_steps_per_quantum=OPTIMIZER_STEPS_PER_QUANTUM,
        maximum_optimizer_steps=100,
        retained_checkpoint_interval_steps=50,
        evaluation_interval_optimizer_steps=50,
        evaluation_timeout_seconds=60,
        evaluation_maximum_attempts=2,
        evaluation_retry_backoff_seconds=0,
    )


def _training_arguments(run_path: Path) -> TrainingArgs:
    training = replace(
        TRAINING_ARGS.training,
        global_batch_size=GLOBAL_BATCH_SIZE,
        local_batch_size=1,
        credit_training=_credit_parameters(),
    )
    return replace(TRAINING_ARGS, save_path=str(run_path), training=training)


def _replay_dataset(sample_count: int) -> SelfPlayDataset:
    dataset = SelfPlayDataset()
    for sample_index in range(sample_count):
        state = np.zeros((C, H, W), dtype=np.int8)
        state[0, sample_index % H, sample_index % W] = 1
        dataset.add_sample(
            state=state,
            visit_counts=[(sample_index % 64, 3), ((sample_index + 1) % 64, 1)],
            value_target=ReplayValueTarget.from_scores(
                final_score=float((sample_index % 3) - 1),
                mcts_root_value=float((sample_index % 5) - 2) / 2,
                termination_reason=TerminationReason.NATURAL,
            ),
            sample_metadata=ReplaySampleMetadata(
                ply=sample_index,
                current_player_piece_count=16,
                opponent_piece_count=16,
            ),
        )
    dataset.stats = SelfPlayDatasetStats(num_samples=sample_count, num_games=1)
    return dataset


def _write_run_manifest(run_path: Path) -> RunManifest:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    source_revision = 'a' * 40
    manifest = RunManifest(
        configuration=configuration,
        approval=ApprovalRecord(
            approved_by='integration-test',
            approved_at_utc=datetime.now(timezone.utc),
            run_name=configuration.run_name,
            source_revision=source_revision,
            configuration_sha256=configuration_sha256(configuration),
            provider_name=configuration.hardware.provider_name,
            offer_id=configuration.hardware.offer_id,
            cost_currency=CostCurrency.EUR,
            hourly_price=1,
            maximum_cost=1,
            maximum_wall_time_minutes=60,
        ),
        resolved_hardware=ResolvedHardware(
            visible_gpu_names=(),
            visible_gpu_count=0,
            logical_cpu_count=1,
            total_ram_gib=1,
            free_disk_gib=1,
        ),
        source_revision=source_revision,
        source_worktree_clean=True,
        initial_model_sha256='b' * 64,
        evaluation_dataset_sha256=None,
        stockfish_binary_sha256=None,
        open_file_soft_limit=1_024,
        torch_version='test',
        cuda_version=None,
    )
    (run_path / 'run_manifest.json').write_text(
        manifest.model_dump_json(indent=2) + '\n',
        encoding='utf-8',
    )
    return manifest


def _write_checkpoint(
    run_path: Path,
    model_version: int,
    sample_signature: tuple[tuple[str, int], ...],
) -> Path:
    signature_bytes = repr(sample_signature).encode()
    artifact_paths = (
        run_path / f'model_{model_version}.pt',
        run_path / f'optimizer_{model_version}.pt',
        run_path / f'model_{model_version}.jit.pt',
    )
    for artifact_index, artifact_path in enumerate(artifact_paths):
        artifact_path.write_bytes(signature_bytes + bytes((artifact_index,)))
    checkpoint = CheckpointManifest(
        iteration=model_version,
        model_path=artifact_paths[0].name,
        model_sha256=file_sha256(artifact_paths[0]),
        optimizer_path=artifact_paths[1].name,
        optimizer_sha256=file_sha256(artifact_paths[1]),
        jit_model_path=artifact_paths[2].name,
        jit_model_sha256=file_sha256(artifact_paths[2]),
        replay_files=(),
    )
    checkpoint_path = run_path / f'checkpoint_{model_version}.json'
    checkpoint_path.write_text(
        checkpoint.model_dump_json(indent=2) + '\n',
        encoding='utf-8',
    )
    return checkpoint_path


def _sample_quantum(replay: RollingReplayBuffer) -> tuple[tuple[str, int], ...]:
    with replay.lease_quantum(
        global_step=0,
        global_sample_count=GLOBAL_BATCH_SIZE * OPTIMIZER_STEPS_PER_QUANTUM,
        world_size=WORLD_SIZE,
        global_batch_size=GLOBAL_BATCH_SIZE,
    ) as lease:
        assert len(lease.partitions) == WORLD_SIZE
        assert all(len(partition.references) == OPTIMIZER_STEPS_PER_QUANTUM for partition in lease.partitions)
        references = tuple(reference for partition in lease.partitions for reference in partition.references)
        for optimizer_step in range(OPTIMIZER_STEPS_PER_QUANTUM):
            batch = tuple(
                reference
                for reference in references
                if reference.global_batch_position // GLOBAL_BATCH_SIZE == optimizer_step
            )
            assert len(batch) == GLOBAL_BATCH_SIZE
            assert len({(reference.segment_id, reference.sample_index) for reference in batch}) == GLOBAL_BATCH_SIZE
        return tuple(
            (reference.segment_id, reference.sample_index)
            for reference in sorted(references, key=lambda item: item.global_batch_position)
        )


def _decode_four_rank_quantum(replay: RollingReplayBuffer) -> None:
    requests = tuple(
        ReplayQuantumRequest(
            global_step=0,
            optimizer_steps=OPTIMIZER_STEPS_PER_QUANTUM,
            global_batch_size=GLOBAL_BATCH_SIZE,
            world_size=WORLD_SIZE,
            rank=rank,
        )
        for rank in range(WORLD_SIZE)
    )
    quanta = tuple(decode_rank_quantum(replay, request) for request in requests)

    assert all(quantum.optimizer_steps == OPTIMIZER_STEPS_PER_QUANTUM for quantum in quanta)
    assert all(quantum.local_batch_size == 1 for quantum in quanta)
    assert sum(len(quantum.full_batch) for quantum in quanta) == (GLOBAL_BATCH_SIZE * OPTIMIZER_STEPS_PER_QUANTUM)
    assert all(len(tuple(quantum.optimizer_batches())) == OPTIMIZER_STEPS_PER_QUANTUM for quantum in quanta)
    assert all(quantum.decode_statistics.selected_rows == OPTIMIZER_STEPS_PER_QUANTUM for quantum in quanta)
    assert all(bool(np.allclose(quantum.full_batch.policy_targets.sum(dim=1).numpy(), 1.0)) for quantum in quanta)


def _refresh_self_play(
    run_path: Path,
    arguments: TrainingArgs,
    communication: Communication,
    pointer_json: str,
    publication: CreditPublicationManifest,
) -> _RefreshProbe:
    communication.publish_persistent_value(LATEST_SELF_PLAY_MODEL_VERSION, pointer_json)
    process = object.__new__(SelfPlayProcess)
    process.args = arguments
    process.node_id = 0
    process.communication = communication
    process.self_play = _RefreshProbe()
    process.loaded_credit_jit_sha256 = None
    process.loaded_credit_publication_pointer = None
    previous_dataset = process.self_play.dataset
    previous_roots = process.self_play.roots

    assert process._refresh_model_if_requested(0) == publication.model_version
    assert process.self_play.dataset is previous_dataset
    assert process.self_play.roots is previous_roots
    assert process.self_play.search_schedule_state is not None
    assert process.self_play.search_schedule_state.schedule_version == publication.model_version
    assert process.self_play.refreshes == [
        (
            publication.model_version,
            run_path / publication.jit_model.path,
        )
    ]
    return process.self_play


def test_credit_runtime_end_to_end_survives_restart_eviction_and_evaluation_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_path = tmp_path / 'run'
    replay_inbox = run_path / 'replay_inbox'
    replay_index_path = run_path / 'rolling-replay-index.json'
    run_path.mkdir()
    run_manifest = _write_run_manifest(run_path)
    arguments = _training_arguments(run_path)
    parameters = _credit_parameters()
    replay = RollingReplayBuffer(
        replay_inbox,
        replay_index_path,
        capacity=UNIQUE_SAMPLES_PER_QUANTUM,
        sampler_seed=73,
    )
    first_manifest = commit_replay_shard(
        _replay_dataset(UNIQUE_SAMPLES_PER_QUANTUM),
        replay_inbox,
        producing_worker=0,
        minimum_model_version=0,
        maximum_model_version=0,
        shard_id='first',
    )

    assert first_manifest.unique_sample_count == UNIQUE_SAMPLES_PER_QUANTUM
    assert first_manifest.content_sha256 == file_sha256(replay_inbox / first_manifest.hdf5_file_name)
    assert not tuple(replay_inbox.glob('*.tmp'))
    ingest = replay.discover_committed_shards()
    assert ingest.unique_samples == UNIQUE_SAMPLES_PER_QUANTUM
    assert ingest.presentation_credits == UNIQUE_SAMPLES_PER_QUANTUM * int(REPLAY_RATIO)
    assert replay.discover_committed_shards().presentation_credits == 0

    ledger = CreditTrainingLedger(
        run_path,
        parameters,
        GLOBAL_BATCH_SIZE,
    )
    progress = ledger.reconcile_credited_samples(replay.credited_unique_sample_count)
    required_credits = parameters.presentation_credits_per_quantum(GLOBAL_BATCH_SIZE)
    assert required_credits == 200
    assert parameters.unique_samples_per_quantum(GLOBAL_BATCH_SIZE) == UNIQUE_SAMPLES_PER_QUANTUM
    assert progress.can_train(required_credits)
    assert progress.available_position_credits == Decimal(200)

    sample_signature = _sample_quantum(replay)
    assert len(sample_signature) == required_credits
    restarted_replay = RollingReplayBuffer(
        replay_inbox,
        replay_index_path,
        capacity=UNIQUE_SAMPLES_PER_QUANTUM,
    )
    assert _sample_quantum(restarted_replay) == sample_signature
    _decode_four_rank_quantum(restarted_replay)

    checkpoint_path = _write_checkpoint(run_path, model_version=1, sample_signature=sample_signature)
    prepared = ledger.prepare_quantum(checkpoint_path)
    assert prepared.prepared_progress.completed_optimizer_steps == OPTIMIZER_STEPS_PER_QUANTUM
    assert prepared.prepared_progress.completed_training_quanta == 1
    assert prepared.prepared_progress.model_version == 1
    assert prepared.prepared_progress.consumed_position_credits == Decimal(required_credits)
    assert prepared.checkpoint_manifest_sha256 == file_sha256(checkpoint_path)

    restarted_ledger = CreditTrainingLedger(
        run_path,
        parameters,
        GLOBAL_BATCH_SIZE,
    )
    recovered_prepared = restarted_ledger.prepared_quantum
    assert recovered_prepared == prepared
    publication = create_credit_publication_manifest(
        run_path,
        recovered_prepared.prepared_progress,
        GLOBAL_BATCH_SIZE,
    )
    pointer = write_credit_publication_manifest(run_path, publication)
    loaded_pointer, loaded_publication = load_credit_publication_pointer(
        run_path,
        pointer.model_dump_json(),
        PublicationValidationScope.ALL_ARTIFACTS,
    )
    assert loaded_pointer == pointer
    assert loaded_publication == publication
    assert pointer.manifest_sha256 == file_sha256(publication_manifest_path(run_path, 1))
    assert publication.checkpoint_manifest_sha256 == file_sha256(checkpoint_path)
    assert publication.run_configuration_sha256 == configuration_sha256(run_manifest.configuration)

    communication = Communication(str(run_path / 'communication'))
    _refresh_self_play(
        run_path,
        arguments,
        communication,
        pointer.model_dump_json(),
        publication,
    )
    commander = object.__new__(CommanderProcess)
    commander.communication = communication
    commander._wait_for_model_acknowledgements(
        model_version=publication.model_version,
        jit_sha256=publication.jit_model.sha256,
        node_ids=(0,),
        timeout_seconds=1,
    )
    assert (
        communication.try_receive_value_from_id(
            self_play_model_refreshed_message(publication.model_version),
            0,
        )
        is None
    )

    committed = restarted_ledger.commit_prepared_quantum()
    assert committed.completed_optimizer_steps == OPTIMIZER_STEPS_PER_QUANTUM
    assert committed.completed_training_quanta == 1
    assert committed.model_version == 1
    assert committed.available_position_credits == 0

    _InterruptibleEvaluationProcess.created.clear()
    monkeypatch.setattr(scheduler_module, 'Process', _InterruptibleEvaluationProcess)
    scheduler = CreditEvaluationScheduler(run_id=1, args=arguments)
    scheduler.offer(publication)
    scheduler.poll()
    assert scheduler.state.active is not None
    assert scheduler.state.active.source.model_version == publication.model_version
    assert scheduler.state.active.source.publication_manifest_sha256 == pointer.manifest_sha256

    second_manifest = commit_replay_shard(
        _replay_dataset(8),
        replay_inbox,
        producing_worker=1,
        minimum_model_version=1,
        maximum_model_version=1,
        shard_id='second',
    )
    second_ingest = restarted_replay.discover_committed_shards()
    assert second_ingest.unique_samples == 8
    assert restarted_replay.unique_sample_count == 8
    assert restarted_replay.credited_unique_sample_count == 58
    assert not (replay_inbox / first_manifest.hdf5_file_name).exists()
    assert not (replay_inbox / first_manifest.manifest_file_name).exists()
    assert (replay_inbox / second_manifest.hdf5_file_name).is_file()

    recovered_after_commit = CreditTrainingLedger(
        run_path,
        parameters,
        GLOBAL_BATCH_SIZE,
    )
    recovered_progress = recovered_after_commit.reconcile_credited_samples(
        restarted_replay.credited_unique_sample_count
    )
    assert recovered_progress.credited_unique_samples == 58
    assert recovered_progress.earned_position_credits == Decimal(232)
    assert recovered_progress.consumed_position_credits == Decimal(200)
    assert recovered_progress.available_position_credits == Decimal(32)
    assert recovered_progress.completed_optimizer_steps == OPTIMIZER_STEPS_PER_QUANTUM

    failed_process = _InterruptibleEvaluationProcess.created[-1]
    failed_process.alive = False
    failed_process.exitcode = -9
    scheduler.poll()
    assert scheduler.state.pending is not None
    assert scheduler.state.results[-1].status is CreditEvaluationStatus.RETRY_PENDING

    scheduler.poll()
    assert scheduler.state.active is not None
    retry_process = _InterruptibleEvaluationProcess.created[-1]
    assert retry_process is not failed_process
    scheduler.close()
    assert retry_process.terminated
    assert scheduler.state.active is None
    assert scheduler.state.results[-1].status is CreditEvaluationStatus.INTERRUPTED
    assert scheduler.state.results[-1].source.publication_manifest_sha256 == pointer.manifest_sha256
    restarted_scheduler = CreditEvaluationScheduler(run_id=1, args=arguments)
    assert restarted_scheduler.state == scheduler.state

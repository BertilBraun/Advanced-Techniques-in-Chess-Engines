from __future__ import annotations

import multiprocessing
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import pytest
import torch
import torch.distributed as distributed

sys.modules.setdefault('GPUtil', ModuleType('GPUtil'))

from src.cluster.CreditTrainerProcess import MaintainCreditReplayCommand, _maintain_replay, credit_quantum_request
from src.cluster.CommanderProcess import CommanderProcess, credit_training_progress_axis
from src.cluster.TrainerProcess import available_tcp_port
from src.settings import TRAINING_ARGS
from src.train.RollingReplayBuffer import (
    ROLLING_REPLAY_INDEX_SCHEMA_VERSION,
    LogicalReplaySegment,
    PhysicalReplayPayload,
    ReplayLiveStatistics,
    ReplayPayloadKind,
    ReplayShardManifest,
    RollingReplayBuffer,
    RollingReplayIndexState,
    TerminationCounts,
    file_sha256,
    replay_live_statistics,
)
from src.self_play.value_target import REPLAY_SCHEMA_VERSION
from src.train.TrainingArgs import CreditTrainingParams, TrainingArgs
from src.train.CreditTrainingLedger import (
    CreditTrainingLedger,
    CreditTrainingProgress,
    PreparedTrainingQuantum,
)
from src.util.communication import Communication, self_play_model_refreshed_message
from src.util.save_paths import CheckpointManifest
from test_helpers.credit_trainer_protocol import run_gloo_decode_rank, run_gloo_protocol_rank
from tools.production_ddp_fixture import write_replay_fixture


WORLD_SIZE = 4


class _ReplayMaintenanceProbe:
    def __init__(self, credited_unique_samples: int) -> None:
        self.credited_unique_sample_count = credited_unique_samples
        self.credited_completed_search_count = credited_unique_samples * 100
        self.unique_sample_count = credited_unique_samples
        self.compaction_calls = 0
        self.capacity = 0

    def set_capacity(self, capacity: int) -> None:
        self.capacity = capacity

    def discover_committed_shards(self) -> None:
        self.credited_unique_sample_count = 12_800
        self.credited_completed_search_count = 1_280_000
        self.unique_sample_count = 12_800

    def compact_one_idle_container(self) -> NoReturn:
        self.compaction_calls += 1
        raise AssertionError('Compaction must not start after replay becomes credit-eligible.')

    def live_statistics(self, measured_at_seconds: float) -> ReplayLiveStatistics:
        assert measured_at_seconds > 0
        return ReplayLiveStatistics(
            oldest_source_model_version=2,
            newest_source_model_version=8,
            weighted_mean_source_model_version_midpoint=5.5,
            oldest_position_age_seconds=10,
            weighted_mean_position_age_seconds=4,
        )


def _credit_training_arguments() -> TrainingArgs:
    parameters = CreditTrainingParams(
        replay_ratio=Decimal(4),
        optimizer_steps_per_quantum=50,
        maximum_optimizer_steps=500_000,
        initial_replay_capacity_unique_positions=100_000,
        maximum_replay_capacity_unique_positions=2_500_000,
        replay_capacity_ramp_model_versions=1_000,
        retained_checkpoint_interval_steps=1_000,
    )
    training = replace(
        TRAINING_ARGS.training,
        global_batch_size=1_024,
        local_batch_size=256,
        credit_training=parameters,
    )
    cluster = replace(
        TRAINING_ARGS.cluster,
        trainer_device_type='cpu',
        trainer_process_group_backend='gloo',
        trainer_rank_zero_device_id=0,
        trainer_ddp_device_ids=(0, 1, 2, 3),
    )
    return replace(TRAINING_ARGS, training=training, cluster=cluster)


def _write_large_index(index_path: Path) -> None:
    manifest = ReplayShardManifest(
        schema_version=REPLAY_SCHEMA_VERSION,
        shard_id='large',
        game_count=1,
        unique_sample_count=60_000,
        producing_worker=0,
        minimum_model_version=0,
        maximum_model_version=0,
        termination_counts=TerminationCounts(
            natural=60_000,
            resignation=0,
            ply_cap=0,
            material_adjudication=0,
            diagnostic=0,
        ),
        content_sha256='0' * 64,
        creation_timestamp_seconds=1.0,
        hdf5_file_name='large.hdf5',
    )
    state = RollingReplayIndexState(
        schema_version=ROLLING_REPLAY_INDEX_SCHEMA_VERSION,
        sampler_seed=91,
        credited_unique_samples=60_000,
        live_segments=(
            LogicalReplaySegment(
                segment_id='large',
                source_manifest=manifest,
                physical_payload_id='large',
                physical_offset=0,
                unique_sample_count=60_000,
            ),
        ),
        physical_payloads=(
            PhysicalReplayPayload(
                payload_id='large',
                kind=ReplayPayloadKind.PRODUCER_SHARD,
                hdf5_file_name='large.hdf5',
                sidecar_file_name='large.manifest.json',
                unique_sample_count=60_000,
                content_sha256='0' * 64,
            ),
        ),
        retired_payload_ids=(),
        active_compaction=None,
    )
    index_path.write_text(state.model_dump_json(), encoding='utf-8')


def _write_decode_fixture(replay_inbox: Path, index_path: Path) -> None:
    replay_inbox.mkdir(parents=True)
    hdf5_path = replay_inbox / 'decode.hdf5'
    write_replay_fixture(hdf5_path, sample_count=16, seed=71)
    manifest = ReplayShardManifest(
        schema_version=REPLAY_SCHEMA_VERSION,
        shard_id='decode',
        game_count=2,
        unique_sample_count=16,
        producing_worker=0,
        minimum_model_version=0,
        maximum_model_version=0,
        termination_counts=TerminationCounts(
            natural=16,
            resignation=0,
            ply_cap=0,
            material_adjudication=0,
            diagnostic=0,
        ),
        content_sha256=file_sha256(hdf5_path),
        creation_timestamp_seconds=1.0,
        hdf5_file_name=hdf5_path.name,
    )
    (replay_inbox / manifest.manifest_file_name).write_text(
        manifest.model_dump_json(),
        encoding='utf-8',
    )
    writer = RollingReplayBuffer(
        replay_inbox=replay_inbox,
        index_path=index_path,
        sampler_seed=37,
    )
    writer.discover_committed_shards()


def test_four_rank_gloo_protocol_observes_identical_phase_transitions() -> None:
    context = multiprocessing.get_context('spawn')
    output_queue = context.Queue()
    initialization_method = f'tcp://127.0.0.1:{available_tcp_port()}'
    processes = tuple(
        context.Process(
            target=run_gloo_protocol_rank,
            args=(rank, initialization_method, output_queue),
        )
        for rank in range(WORLD_SIZE)
    )
    for process in processes:
        process.start()
    results = tuple(output_queue.get(timeout=30) for _ in range(WORLD_SIZE))
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    expected = (
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 0, 2),
        (1, 0, 0, 3),
        (2, 1_250, 26, 0),
        (2, 1_250, 26, 1),
        (2, 1_250, 26, 2),
        (2, 1_250, 26, 3),
    )
    assert {rank for rank, _ in results} == set(range(WORLD_SIZE))
    assert all(observed == expected for _, observed in results)


def test_four_gloo_ranks_execute_real_rank_local_deterministic_decode(
    tmp_path: Path,
) -> None:
    replay_inbox = tmp_path / 'inbox'
    index_path = tmp_path / 'index.json'
    _write_decode_fixture(replay_inbox, index_path)
    context = multiprocessing.get_context('spawn')
    output_queue = context.Queue()
    initialization_method = f'tcp://127.0.0.1:{available_tcp_port()}'
    processes = tuple(
        context.Process(
            target=run_gloo_decode_rank,
            args=(
                rank,
                initialization_method,
                str(replay_inbox),
                str(index_path),
                output_queue,
            ),
        )
        for rank in range(WORLD_SIZE)
    )
    for process in processes:
        process.start()
    results = tuple(output_queue.get(timeout=30) for _ in range(WORLD_SIZE))
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    ordered = tuple(result for result in sorted(results))
    assert all(optimizer_steps == 2 for _, _, optimizer_steps in ordered)
    decoded_plies = tuple(ply for _, plies, _ in ordered for ply in plies)
    assert len(decoded_plies) == 8
    assert len(set(decoded_plies)) == 8


def test_four_ranks_reconstruct_exact_disjoint_fifty_step_quantum(
    tmp_path: Path,
) -> None:
    arguments = _credit_training_arguments()
    index_path = tmp_path / 'index.json'
    _write_large_index(index_path)
    replay = RollingReplayBuffer(
        replay_inbox=tmp_path,
        index_path=index_path,
        read_only=True,
    )

    requests = tuple(credit_quantum_request(arguments, rank, global_step=1_250) for rank in range(WORLD_SIZE))
    assert all(request.optimizer_steps == 50 for request in requests)
    assert all(request.global_sample_count == 51_200 for request in requests)
    assert all(request.local_batch_size == 256 for request in requests)

    with replay.lease_quantum(
        global_step=1_250,
        global_sample_count=51_200,
        world_size=WORLD_SIZE,
        global_batch_size=1_024,
    ) as lease:
        partitions = lease.partitions
        selected = {
            (reference.segment_id, reference.sample_index)
            for partition in partitions
            for reference in partition.references
        }
        assert len(selected) == 51_200
        for rank, partition in enumerate(partitions):
            assert partition.rank == rank
            assert len(partition.references) == 50 * 256
            assert all(reference.global_batch_position % WORLD_SIZE == rank for reference in partition.references)
            for step in range(50):
                local_batch = partition.references[step * 256 : (step + 1) * 256]
                assert len(local_batch) == 256


def test_replay_maintenance_skips_compaction_when_ingest_reaches_credit_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _ReplayMaintenanceProbe(credited_unique_samples=12_799)
    monkeypatch.setattr(distributed, 'barrier', lambda: None)
    monkeypatch.setattr(distributed, 'broadcast', lambda tensor, src: tensor)

    response = _maintain_replay(
        MaintainCreditReplayCommand(
            phase_id=7,
            replay_capacity_unique_positions=100_000,
            compact_below_credited_unique_samples=12_800,
        ),
        probe,
        rank=0,
        device=torch.device('cpu'),
    )

    assert response.credited_unique_samples == 12_800
    assert probe.capacity == 100_000
    assert not response.compacted_container
    assert probe.compaction_calls == 0
    assert response.oldest_source_model_version == 2
    assert response.newest_source_model_version == 8
    assert response.weighted_mean_source_model_version_midpoint == 5.5
    assert response.oldest_position_age_seconds == 10
    assert response.weighted_mean_position_age_seconds == 4


def test_replay_live_statistics_weight_manifest_provenance_by_live_rows() -> None:
    termination_counts = TerminationCounts(
        natural=400,
        resignation=0,
        ply_cap=0,
        material_adjudication=0,
        diagnostic=0,
    )
    older_manifest = ReplayShardManifest(
        schema_version=REPLAY_SCHEMA_VERSION,
        shard_id='older',
        game_count=1,
        unique_sample_count=100,
        producing_worker=0,
        minimum_model_version=2,
        maximum_model_version=4,
        termination_counts=termination_counts,
        content_sha256='0' * 64,
        creation_timestamp_seconds=90,
        hdf5_file_name='older.hdf5',
    )
    newer_manifest = ReplayShardManifest(
        schema_version=REPLAY_SCHEMA_VERSION,
        shard_id='newer',
        game_count=1,
        unique_sample_count=300,
        producing_worker=1,
        minimum_model_version=8,
        maximum_model_version=10,
        termination_counts=termination_counts,
        content_sha256='1' * 64,
        creation_timestamp_seconds=98,
        hdf5_file_name='newer.hdf5',
    )
    live_segments = (
        LogicalReplaySegment(
            segment_id='older',
            source_manifest=older_manifest,
            physical_payload_id='older',
            physical_offset=0,
            unique_sample_count=100,
        ),
        LogicalReplaySegment(
            segment_id='newer',
            source_manifest=newer_manifest,
            physical_payload_id='newer',
            physical_offset=0,
            unique_sample_count=300,
        ),
    )

    statistics = replay_live_statistics(live_segments, measured_at_seconds=100)

    assert statistics.oldest_source_model_version == 2
    assert statistics.newest_source_model_version == 10
    assert statistics.weighted_mean_source_model_version_midpoint == 7.5
    assert statistics.oldest_position_age_seconds == 10
    assert statistics.weighted_mean_position_age_seconds == 4


def test_empty_replay_live_statistics_are_absent() -> None:
    statistics = replay_live_statistics((), measured_at_seconds=100)

    assert statistics == ReplayLiveStatistics(
        oldest_source_model_version=None,
        newest_source_model_version=None,
        weighted_mean_source_model_version_midpoint=None,
        oldest_position_age_seconds=None,
        weighted_mean_position_age_seconds=None,
    )


def test_credit_progress_axis_is_trained_position_presentations() -> None:
    progress = CreditTrainingProgress.initial().model_copy(
        update={
            'completed_optimizer_steps': 50,
            'completed_training_quanta': 1,
            'model_version': 1,
            'sampler_global_step': 50,
        }
    )

    assert credit_training_progress_axis(progress, 1_024) == 51_200


def test_model_acknowledgement_rejects_wrong_immutable_jit_hash(tmp_path: Path) -> None:
    commander = object.__new__(CommanderProcess)
    commander.communication = Communication(str(tmp_path / 'communication'))
    acknowledgement = self_play_model_refreshed_message(1)
    commander.communication.send_value_to_id(acknowledgement, 0, 'b' * 64)

    with pytest.raises(ValueError, match='expected'):
        commander._wait_for_model_acknowledgements(
            model_version=1,
            jit_sha256='a' * 64,
            node_ids=(0,),
            timeout_seconds=1,
        )


def test_transient_evaluation_checkpoint_keeps_only_jit_artifact(tmp_path: Path) -> None:
    arguments = _credit_training_arguments()
    parameters = arguments.training.credit_training
    assert parameters is not None
    arguments = replace(
        arguments,
        save_path=str(tmp_path),
        training=replace(
            arguments.training,
            credit_training=replace(
                parameters,
                evaluation_interval_optimizer_steps=500,
            ),
        ),
    )
    manifest = CheckpointManifest(
        iteration=10,
        model_path='model_10.pt',
        model_sha256='a' * 64,
        optimizer_path='optimizer_10.pt',
        optimizer_sha256='b' * 64,
        jit_model_path='model_10.jit.pt',
        jit_model_sha256='c' * 64,
        replay_files=(),
    )
    (tmp_path / 'checkpoint_10.json').write_text(manifest.model_dump_json(), encoding='utf-8')
    for artifact in (manifest.model_path, manifest.optimizer_path, manifest.jit_model_path):
        (tmp_path / artifact).write_text('artifact', encoding='utf-8')
    commander = object.__new__(CommanderProcess)
    commander.args = arguments

    commander._prune_nonretained_credit_checkpoint(10)
    commander._prune_nonretained_credit_checkpoint(10)

    assert not (tmp_path / manifest.model_path).exists()
    assert not (tmp_path / manifest.optimizer_path).exists()
    assert (tmp_path / manifest.jit_model_path).exists()


def _prepared_ledger(run_path: Path) -> CreditTrainingLedger:
    ledger = CreditTrainingLedger(
        run_path,
        CreditTrainingParams(
            replay_ratio=Decimal(4),
            optimizer_steps_per_quantum=1,
            maximum_optimizer_steps=10,
            initial_replay_capacity_unique_positions=1,
            maximum_replay_capacity_unique_positions=10,
            replay_capacity_ramp_model_versions=10,
            retained_checkpoint_interval_steps=1,
        ),
        global_batch_size=1,
    )
    ledger.reconcile_credited_samples(1)
    checkpoint_manifest = run_path / 'checkpoint_1.json'
    checkpoint_manifest.write_text('prepared checkpoint\n', encoding='utf-8')
    ledger.prepare_quantum(checkpoint_manifest)
    return ledger


def test_publication_failure_leaves_prepared_quantum_and_credits_uncommitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ledger = _prepared_ledger(tmp_path)
    commander = object.__new__(CommanderProcess)

    def fail_publication(prepared: PreparedTrainingQuantum) -> None:
        raise RuntimeError(f'publication failed for {prepared.prepared_progress.model_version}')

    monkeypatch.setattr(commander, '_publish_prepared_quantum', fail_publication)
    monkeypatch.setattr(
        commander,
        '_validate_credit_recovery_checkpoint',
        lambda model_version, run_path: None,
    )

    with pytest.raises(RuntimeError, match='publication failed'):
        prepared = ledger.prepared_quantum
        assert prepared is not None
        commander._finish_prepared_publication(ledger, prepared)

    assert ledger.progress.completed_optimizer_steps == 0
    assert ledger.progress.available_position_credits == Decimal(4)
    assert ledger.prepared_quantum is not None


def test_prepared_restart_publishes_and_commits_without_retraining(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ledger = _prepared_ledger(tmp_path)
    commander = object.__new__(CommanderProcess)
    commander.latest_completed_iteration = 0
    published_versions: list[int] = []
    pruned_versions: list[int] = []

    monkeypatch.setattr(
        commander,
        '_publish_prepared_quantum',
        lambda prepared: published_versions.append(prepared.prepared_progress.model_version),
    )
    monkeypatch.setattr(
        commander,
        '_validate_credit_recovery_checkpoint',
        lambda model_version, run_path: None,
    )
    monkeypatch.setattr(
        commander,
        '_prune_nonretained_credit_checkpoint',
        pruned_versions.append,
    )

    prepared = ledger.prepared_quantum
    assert prepared is not None
    commander._finish_prepared_publication(ledger, prepared)

    assert published_versions == [1]
    assert pruned_versions == [0]
    assert ledger.progress.completed_optimizer_steps == 1
    assert ledger.progress.completed_training_quanta == 1
    assert ledger.prepared_quantum is None
    assert commander.latest_completed_iteration == 1

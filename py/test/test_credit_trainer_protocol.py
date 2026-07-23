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
    ReplayPayloadKind,
    ReplayShardManifest,
    RollingReplayBuffer,
    RollingReplayIndexState,
    TerminationCounts,
    file_sha256,
)
from src.self_play.value_target import REPLAY_SCHEMA_VERSION
from src.train.TrainingArgs import CreditTrainingParams, TrainingArgs
from src.train.CreditTrainingLedger import (
    CreditTrainingLedger,
    CreditTrainingProgress,
    PreparedTrainingQuantum,
)
from test_helpers.credit_trainer_protocol import run_gloo_decode_rank, run_gloo_protocol_rank
from tools.production_ddp_fixture import write_replay_fixture


WORLD_SIZE = 4


class _ReplayMaintenanceProbe:
    def __init__(self, credited_unique_samples: int) -> None:
        self.credited_unique_sample_count = credited_unique_samples
        self.unique_sample_count = credited_unique_samples
        self.compaction_calls = 0

    def discover_committed_shards(self) -> None:
        self.credited_unique_sample_count = 12_800
        self.unique_sample_count = 12_800

    def compact_one_idle_container(self) -> NoReturn:
        self.compaction_calls += 1
        raise AssertionError('Compaction must not start after replay becomes credit-eligible.')


def _credit_training_arguments() -> TrainingArgs:
    parameters = CreditTrainingParams(
        replay_ratio=Decimal(4),
        optimizer_steps_per_quantum=50,
        maximum_optimizer_steps=500_000,
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
            compact_below_credited_unique_samples=12_800,
        ),
        probe,
        rank=0,
        device=torch.device('cpu'),
    )

    assert response.credited_unique_samples == 12_800
    assert not response.compacted_container
    assert probe.compaction_calls == 0


def test_credit_progress_axis_is_optimizer_steps_not_quantum_index() -> None:
    progress = CreditTrainingProgress.initial().model_copy(
        update={
            'completed_optimizer_steps': 50,
            'completed_training_quanta': 1,
            'model_version': 1,
            'sampler_global_step': 50,
        }
    )

    assert credit_training_progress_axis(progress) == 50


def _prepared_ledger(run_path: Path) -> CreditTrainingLedger:
    ledger = CreditTrainingLedger(
        run_path,
        CreditTrainingParams(
            replay_ratio=Decimal(4),
            optimizer_steps_per_quantum=1,
            maximum_optimizer_steps=10,
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

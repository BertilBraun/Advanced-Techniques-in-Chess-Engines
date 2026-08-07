from decimal import Decimal
from pathlib import Path

import pytest

from src.cluster.CommanderProcess import CommanderProcess, credit_training_progress_axis
from test_helpers.chess_configuration import CHESS_TRAINING
from src.train.CreditTrainingLedger import (
    CreditTrainingLedger,
    CreditTrainingProgress,
    PreparedTrainingQuantum,
)
from src.train.TrainingArgs import CreditTrainingParams, TrainingArgs
from src.util.communication import Communication, self_play_model_refreshed_message
from src.util.save_paths import CheckpointManifest


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
    trainer = CHESS_TRAINING.trainer.validated_copy(update={'global_batch_size': 1_024, 'local_batch_size': 256})
    trainer_topology = CHESS_TRAINING.topology.trainer.validated_copy(
        update={
            'device_type': 'cuda',
            'process_group_backend': 'nccl',
            'rank_zero_device_id': 0,
            'ddp_device_ids': [0, 1, 2, 3],
        }
    )
    topology = CHESS_TRAINING.topology.validated_copy(update={'trainer': trainer_topology.model_dump(mode='json')})
    lifecycle = CHESS_TRAINING.lifecycle.validated_copy(update={'credit': parameters.model_dump(mode='json')})
    return CHESS_TRAINING.validated_copy(
        update={
            'trainer': trainer.model_dump(mode='json'),
            'topology': topology.model_dump(mode='json'),
            'lifecycle': lifecycle.model_dump(mode='json'),
        }
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
    schedule = arguments.lifecycle.evaluation.validated_copy(update={'interval_optimizer_steps': 500})
    lifecycle = arguments.lifecycle.validated_copy(update={'evaluation': schedule.model_dump(mode='json')})
    arguments = arguments.validated_copy(
        update={'save_path': str(tmp_path), 'lifecycle': lifecycle.model_dump(mode='json')}
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
    monkeypatch.setattr(commander, '_validate_credit_recovery_checkpoint', lambda model_version, run_path: None)

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
    commander.latest_completed_model_version = 0
    published_versions: list[int] = []
    pruned_versions: list[int] = []

    monkeypatch.setattr(
        commander,
        '_publish_prepared_quantum',
        lambda prepared: published_versions.append(prepared.prepared_progress.model_version),
    )
    monkeypatch.setattr(commander, '_validate_credit_recovery_checkpoint', lambda model_version, run_path: None)
    monkeypatch.setattr(commander, '_prune_nonretained_credit_checkpoint', pruned_versions.append)

    prepared = ledger.prepared_quantum
    assert prepared is not None
    commander._finish_prepared_publication(ledger, prepared)

    assert published_versions == [1]
    assert pruned_versions == [0]
    assert ledger.progress.completed_optimizer_steps == 1
    assert ledger.progress.completed_training_quanta == 1
    assert ledger.prepared_quantum is None
    assert commander.latest_completed_model_version == 1

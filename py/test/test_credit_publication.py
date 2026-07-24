from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src.experiment.cost_accounting import CostCurrency
from src.experiment.run_configuration import (
    ApprovalRecord,
    ResolvedHardware,
    RunManifest,
    configuration_sha256,
    load_run_configuration,
)
from src.train.CreditPublication import (
    PublicationValidationScope,
    create_credit_publication_manifest,
    file_sha256,
    load_credit_publication_pointer,
    write_credit_publication_manifest,
)
from src.train.CreditTrainingLedger import CreditTrainingProgress
from src.util.save_paths import CheckpointManifest


CONFIGURATION_PATH = Path(__file__).parents[1] / 'configs' / 'chess-continuation-4x4070-pilot.json'


def _write_run_manifest(run_path: Path, source_revision: str = 'a' * 40) -> RunManifest:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    manifest = RunManifest(
        configuration=configuration,
        approval=ApprovalRecord(
            approved_by='test',
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


def _write_checkpoint(run_path: Path, model_version: int) -> None:
    artifact_paths = (
        run_path / f'model_{model_version}.pt',
        run_path / f'optimizer_{model_version}.pt',
        run_path / f'model_{model_version}.jit.pt',
    )
    for index, path in enumerate(artifact_paths):
        path.write_bytes(f'artifact-{index}'.encode())
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
    (run_path / f'checkpoint_{model_version}.json').write_text(
        checkpoint.model_dump_json(indent=2) + '\n',
        encoding='utf-8',
    )


def _progress() -> CreditTrainingProgress:
    return CreditTrainingProgress(
        credited_unique_samples=12_800,
        earned_position_credits=Decimal(51_200),
        consumed_position_credits=Decimal(51_200),
        available_position_credits=Decimal(0),
        completed_optimizer_steps=50,
        completed_training_quanta=1,
        model_version=1,
        sampler_global_step=50,
    )


def test_immutable_publication_binds_artifacts_configuration_and_source(tmp_path: Path) -> None:
    run_manifest = _write_run_manifest(tmp_path)
    _write_checkpoint(tmp_path, 1)

    publication = create_credit_publication_manifest(tmp_path, _progress(), 1_024)
    pointer = write_credit_publication_manifest(tmp_path, publication)
    loaded_pointer, loaded = load_credit_publication_pointer(tmp_path, pointer.model_dump_json())

    assert loaded_pointer == pointer
    assert loaded == publication
    assert loaded.source_revision == run_manifest.source_revision
    assert loaded.run_configuration_sha256 == configuration_sha256(run_manifest.configuration)
    assert loaded.trained_position_presentations == 51_200
    assert loaded.available_position_credits == 0


def test_self_play_scope_validates_exact_jit_but_not_pruned_recovery_artifacts(
    tmp_path: Path,
) -> None:
    _write_run_manifest(tmp_path)
    _write_checkpoint(tmp_path, 1)
    publication = create_credit_publication_manifest(tmp_path, _progress(), 1_024)
    pointer = write_credit_publication_manifest(tmp_path, publication)
    (tmp_path / publication.model.path).unlink()
    (tmp_path / publication.optimizer.path).unlink()

    _, loaded = load_credit_publication_pointer(
        tmp_path,
        pointer.model_dump_json(),
        PublicationValidationScope.JIT_ONLY,
    )
    assert loaded.jit_model.sha256 == publication.jit_model.sha256

    (tmp_path / publication.jit_model.path).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='artifact hash does not match'):
        load_credit_publication_pointer(
            tmp_path,
            pointer.model_dump_json(),
            PublicationValidationScope.JIT_ONLY,
        )


def test_publication_rejects_run_manifest_provenance_change(tmp_path: Path) -> None:
    _write_run_manifest(tmp_path)
    _write_checkpoint(tmp_path, 1)
    publication = create_credit_publication_manifest(tmp_path, _progress(), 1_024)
    pointer = write_credit_publication_manifest(tmp_path, publication)
    _write_run_manifest(tmp_path, source_revision='c' * 40)

    with pytest.raises(ValueError, match='source revision'):
        load_credit_publication_pointer(tmp_path, pointer.model_dump_json())

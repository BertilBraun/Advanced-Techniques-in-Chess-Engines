from __future__ import annotations

import hashlib
import os
import uuid
from decimal import Decimal
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.experiment.run_configuration import RunManifest, configuration_sha256
from src.train.CreditTrainingLedger import CreditTrainingProgress
from src.util.save_paths import CheckpointManifest, load_checkpoint_manifest


CREDIT_PUBLICATION_SCHEMA_VERSION = 1


class PublicationValidationScope(str, Enum):
    ALL_ARTIFACTS = 'all_artifacts'
    JIT_ONLY = 'jit_only'


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary_path.open('x', encoding='utf-8') as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, path)


class PublishedArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class CreditPublicationManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schema_version: int = CREDIT_PUBLICATION_SCHEMA_VERSION
    model_version: int = Field(ge=0)
    completed_optimizer_steps: int = Field(ge=0)
    trained_position_presentations: int = Field(ge=0)
    global_batch_size: int = Field(gt=0)
    credited_unique_positions: int = Field(ge=0)
    earned_position_credits: Decimal = Field(ge=0)
    consumed_position_credits: Decimal = Field(ge=0)
    available_position_credits: Decimal = Field(ge=0)
    model: PublishedArtifact
    optimizer: PublishedArtifact
    jit_model: PublishedArtifact
    checkpoint_manifest_path: str = Field(min_length=1)
    checkpoint_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_revision: str = Field(pattern=r'^[0-9a-f]{40}$')
    run_configuration_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def validate_counters(self) -> CreditPublicationManifest:
        if self.schema_version != CREDIT_PUBLICATION_SCHEMA_VERSION:
            raise ValueError(f'Unsupported credit-publication schema {self.schema_version}.')
        if self.trained_position_presentations != self.completed_optimizer_steps * self.global_batch_size:
            raise ValueError('Trained position presentations must equal optimizer steps times global batch size.')
        if Decimal(self.trained_position_presentations) != self.consumed_position_credits:
            raise ValueError('Trained position presentations must equal consumed presentation credits.')
        if self.consumed_position_credits > self.earned_position_credits:
            raise ValueError('Consumed position credits cannot exceed earned position credits.')
        if self.available_position_credits != self.earned_position_credits - self.consumed_position_credits:
            raise ValueError('Available position credits must equal earned minus consumed credits.')
        return self


class CreditPublicationPointer(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    model_version: int = Field(ge=0)
    manifest_path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


def publication_manifest_path(run_path: Path, model_version: int) -> Path:
    if model_version < 0:
        raise ValueError('Model version must be nonnegative.')
    return run_path / 'credit-publications' / f'model-version-{model_version:010d}.json'


def create_credit_publication_manifest(
    run_path: Path,
    progress: CreditTrainingProgress,
    global_batch_size: int,
) -> CreditPublicationManifest:
    if global_batch_size <= 0:
        raise ValueError('Global batch size must be positive.')
    checkpoint = load_checkpoint_manifest(progress.model_version, run_path)
    run_manifest_path = run_path / 'run_manifest.json'
    if not run_manifest_path.is_file():
        raise ValueError(f'Run manifest does not exist: {run_manifest_path}')
    run_manifest = RunManifest.model_validate_json(run_manifest_path.read_text(encoding='utf-8'))
    checkpoint_path = run_path / f'checkpoint_{progress.model_version}.json'
    expected_presentations = progress.completed_optimizer_steps * global_batch_size
    if Decimal(expected_presentations) != progress.consumed_position_credits:
        raise ValueError('Credit progress does not match optimizer-step presentations.')
    return CreditPublicationManifest(
        model_version=progress.model_version,
        completed_optimizer_steps=progress.completed_optimizer_steps,
        trained_position_presentations=expected_presentations,
        global_batch_size=global_batch_size,
        credited_unique_positions=progress.credited_unique_samples,
        earned_position_credits=progress.earned_position_credits,
        consumed_position_credits=progress.consumed_position_credits,
        available_position_credits=progress.available_position_credits,
        model=_artifact(checkpoint.model_path, checkpoint.model_sha256),
        optimizer=_artifact(checkpoint.optimizer_path, checkpoint.optimizer_sha256),
        jit_model=_artifact(checkpoint.jit_model_path, checkpoint.jit_model_sha256),
        checkpoint_manifest_path=checkpoint_path.relative_to(run_path).as_posix(),
        checkpoint_manifest_sha256=file_sha256(checkpoint_path),
        source_revision=run_manifest.source_revision,
        run_configuration_sha256=configuration_sha256(run_manifest.configuration),
    )


def write_credit_publication_manifest(
    run_path: Path,
    manifest: CreditPublicationManifest,
) -> CreditPublicationPointer:
    path = publication_manifest_path(run_path, manifest.model_version)
    serialized = manifest.model_dump_json(indent=2) + '\n'
    if path.exists():
        existing = CreditPublicationManifest.model_validate_json(path.read_text(encoding='utf-8'))
        if existing != manifest:
            raise ValueError(f'Immutable credit publication already exists with different content: {path}')
    else:
        _atomic_write(path, serialized)
    validate_credit_publication_manifest(run_path, manifest)
    return CreditPublicationPointer(
        model_version=manifest.model_version,
        manifest_path=path.relative_to(run_path).as_posix(),
        manifest_sha256=file_sha256(path),
    )


def load_credit_publication_pointer(
    run_path: Path,
    serialized_pointer: str,
    validation_scope: PublicationValidationScope = PublicationValidationScope.ALL_ARTIFACTS,
) -> tuple[CreditPublicationPointer, CreditPublicationManifest]:
    pointer = CreditPublicationPointer.model_validate_json(serialized_pointer)
    manifest_path = run_path / pointer.manifest_path
    if not manifest_path.is_file() or file_sha256(manifest_path) != pointer.manifest_sha256:
        raise ValueError('Credit publication pointer references an invalid immutable manifest.')
    manifest = CreditPublicationManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    if manifest.model_version != pointer.model_version:
        raise ValueError('Credit publication pointer and immutable manifest versions differ.')
    validate_credit_publication_manifest(run_path, manifest, validation_scope)
    return pointer, manifest


def load_credit_publication_manifest(
    run_path: Path,
    model_version: int,
    verify_artifacts: bool = True,
) -> CreditPublicationManifest:
    path = publication_manifest_path(run_path, model_version)
    if not path.is_file():
        raise ValueError(f'Credit publication manifest does not exist: {path}')
    manifest = CreditPublicationManifest.model_validate_json(path.read_text(encoding='utf-8'))
    if manifest.model_version != model_version:
        raise ValueError('Credit publication filename and model version differ.')
    if verify_artifacts:
        validate_credit_publication_manifest(run_path, manifest)
    return manifest


def validate_credit_publication_manifest(
    run_path: Path,
    manifest: CreditPublicationManifest,
    validation_scope: PublicationValidationScope = PublicationValidationScope.ALL_ARTIFACTS,
) -> None:
    checkpoint_path = run_path / manifest.checkpoint_manifest_path
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != manifest.checkpoint_manifest_sha256:
        raise ValueError('Credit publication references an invalid checkpoint manifest.')
    checkpoint = CheckpointManifest.model_validate_json(checkpoint_path.read_text(encoding='utf-8'))
    if checkpoint.iteration != manifest.model_version:
        raise ValueError('Checkpoint and publication model versions differ.')
    run_manifest_path = run_path / 'run_manifest.json'
    run_manifest = RunManifest.model_validate_json(run_manifest_path.read_text(encoding='utf-8'))
    if run_manifest.source_revision != manifest.source_revision:
        raise ValueError('Credit publication source revision differs from the run manifest.')
    if configuration_sha256(run_manifest.configuration) != manifest.run_configuration_sha256:
        raise ValueError('Credit publication configuration hash differs from the run manifest.')
    expected_artifacts = (
        (manifest.model, checkpoint.model_path, checkpoint.model_sha256),
        (manifest.optimizer, checkpoint.optimizer_path, checkpoint.optimizer_sha256),
        (manifest.jit_model, checkpoint.jit_model_path, checkpoint.jit_model_sha256),
    )
    for artifact, checkpoint_path_value, checkpoint_sha256 in expected_artifacts:
        if artifact.path != checkpoint_path_value or artifact.sha256 != checkpoint_sha256:
            raise ValueError('Credit publication artifact identity differs from its checkpoint manifest.')
        if validation_scope is PublicationValidationScope.JIT_ONLY and artifact != manifest.jit_model:
            continue
        artifact_path = run_path / artifact.path
        if not artifact_path.is_file() or file_sha256(artifact_path) != artifact.sha256:
            raise ValueError(f'Credit publication artifact hash does not match: {artifact_path}')


def _artifact(path: str, sha256: str) -> PublishedArtifact:
    return PublishedArtifact(path=path, sha256=sha256)

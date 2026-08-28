from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import TypeVar

from pydantic import Field, model_validator
from src.search_budget.sampling import LabelPositionIdentity
from src.util.atomic_file import write_bytes_atomically, write_text_atomically
from src.util.frozen_model import FrozenModel


class LabelShardPhase(str, Enum):
    PREDICTION = 'prediction'
    DEEP_SEARCH = 'deep_search'


class LabelShardManifest(FrozenModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    phase: LabelShardPhase
    source_generation: int = Field(ge=0)
    shard_index: int = Field(ge=0)
    attempt: int = Field(gt=0)
    device_id: int = Field(ge=0)
    position_identities: tuple[LabelPositionIdentity, ...] = Field(min_length=1, max_length=512)
    position_count: int = Field(gt=0, le=512)
    duration_seconds: float = Field(ge=0.0)
    artifact_path: Path
    artifact_sha256: str = Field(min_length=64, max_length=64)
    artifact_size_bytes: int = Field(ge=0)
    checkpoint_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode='after')
    def validate_manifest(self) -> LabelShardManifest:
        if self.position_count != len(self.position_identities):
            raise ValueError('Shard position count must match its identity manifest.')
        if len(set(self.position_identities)) != len(self.position_identities):
            raise ValueError('Shard position identities must be unique.')
        if any(identity.source_generation != self.source_generation for identity in self.position_identities):
            raise ValueError('Every shard position must belong to the manifest source generation.')
        return self


PersistedModelT = TypeVar('PersistedModelT', bound=FrozenModel)


def write_immutable_artifact(path: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    if path.exists():
        existing_content = path.read_bytes()
        if existing_content != content:
            raise ValueError(f'Immutable artifact already exists with different content: {path}')
        return digest
    write_bytes_atomically(path, content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Artifact checksum verification failed after publication: {path}')
    return digest


def write_persisted_model(path: Path, model: FrozenModel) -> None:
    write_text_atomically(path, model.model_dump_json(indent=2) + '\n')


def load_persisted_model(path: Path, model_type: type[PersistedModelT]) -> PersistedModelT:
    return model_type.model_validate_json(path.read_text(encoding='utf-8'))


def validate_complete_shard_coverage(
    expected_positions: tuple[LabelPositionIdentity, ...],
    manifests: tuple[LabelShardManifest, ...],
    phase: LabelShardPhase,
) -> tuple[LabelShardManifest, ...]:
    if not expected_positions:
        raise ValueError('Complete shard validation requires a nonempty generation sample.')
    if len(set(expected_positions)) != len(expected_positions):
        raise ValueError('Expected generation position identities must be unique.')
    ordered_manifests = tuple(sorted(manifests, key=lambda manifest: manifest.shard_index))
    if tuple(manifest.shard_index for manifest in ordered_manifests) != tuple(range(len(ordered_manifests))):
        raise ValueError('Shard indices must be unique and contiguous from zero.')
    if any(manifest.phase != phase for manifest in ordered_manifests):
        raise ValueError('Shard phase does not match the requested generation phase.')
    if not ordered_manifests:
        raise ValueError('A partial generation with no completed shards cannot be finalized.')
    source_generation = expected_positions[0].source_generation
    if any(manifest.source_generation != source_generation for manifest in ordered_manifests):
        raise ValueError('Shard manifests cannot span source generations.')
    covered_positions = tuple(identity for manifest in ordered_manifests for identity in manifest.position_identities)
    if covered_positions != expected_positions:
        raise ValueError('Shard manifests do not provide exact deterministic generation coverage.')
    for manifest in ordered_manifests:
        if not manifest.artifact_path.is_file():
            raise ValueError(f'Shard artifact is missing: {manifest.artifact_path}')
        artifact = manifest.artifact_path.read_bytes()
        if len(artifact) != manifest.artifact_size_bytes:
            raise ValueError(f'Shard artifact size does not match its manifest: {manifest.artifact_path}')
        if hashlib.sha256(artifact).hexdigest() != manifest.artifact_sha256:
            raise ValueError(f'Shard artifact checksum does not match its manifest: {manifest.artifact_path}')
    return ordered_manifests

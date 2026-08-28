from __future__ import annotations

from pathlib import Path

import pytest
from src.search_budget.artifacts import (
    LabelShardManifest,
    LabelShardPhase,
    load_persisted_model,
    validate_complete_shard_coverage,
    write_immutable_artifact,
    write_persisted_model,
)
from src.search_budget.sampling import LabelPositionIdentity


def identity(index: int) -> LabelPositionIdentity:
    return LabelPositionIdentity(source_generation=4, game_identity=f'game-{index // 2}', ply=index)


def manifest(
    tmp_path: Path,
    shard_index: int,
    positions: tuple[LabelPositionIdentity, ...],
) -> LabelShardManifest:
    path = tmp_path / f'shard-{shard_index}.bin'
    content = f'artifact-{shard_index}'.encode()
    digest = write_immutable_artifact(path, content)
    return LabelShardManifest(
        phase=LabelShardPhase.DEEP_SEARCH,
        source_generation=4,
        shard_index=shard_index,
        attempt=1,
        device_id=shard_index,
        position_identities=positions,
        position_count=len(positions),
        duration_seconds=1.5,
        artifact_path=path,
        artifact_sha256=digest,
        artifact_size_bytes=len(content),
        checkpoint_sha256='b' * 64,
    )


def test_immutable_artifact_is_atomic_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / 'artifact.bin'
    first_digest = write_immutable_artifact(path, b'payload')
    assert write_immutable_artifact(path, b'payload') == first_digest
    with pytest.raises(ValueError, match='different content'):
        write_immutable_artifact(path, b'conflict')


def test_complete_coverage_validates_order_counts_and_checksums(tmp_path: Path) -> None:
    expected = tuple(identity(index) for index in range(5))
    manifests = (manifest(tmp_path, 0, expected[:3]), manifest(tmp_path, 1, expected[3:]))
    assert (
        validate_complete_shard_coverage(expected, tuple(reversed(manifests)), LabelShardPhase.DEEP_SEARCH) == manifests
    )

    manifests[1].artifact_path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='size|checksum'):
        validate_complete_shard_coverage(expected, manifests, LabelShardPhase.DEEP_SEARCH)


def test_partial_generation_is_rejected(tmp_path: Path) -> None:
    expected = tuple(identity(index) for index in range(5))
    partial = (manifest(tmp_path, 0, expected[:3]),)
    with pytest.raises(ValueError, match='coverage'):
        validate_complete_shard_coverage(expected, partial, LabelShardPhase.DEEP_SEARCH)


def test_typed_manifest_round_trip_uses_atomic_persistence(tmp_path: Path) -> None:
    positions = (identity(0), identity(1))
    original = manifest(tmp_path, 0, positions)
    path = tmp_path / 'manifest.json'
    write_persisted_model(path, original)
    assert load_persisted_model(path, LabelShardManifest) == original

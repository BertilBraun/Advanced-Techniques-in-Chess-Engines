import hashlib
from pathlib import Path

import pytest

from src.util.save_paths import (
    CheckpointManifest,
    ReplayFileReference,
    inference_model_path,
    load_checkpoint_manifest,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checkpoint_fixture(root: Path) -> None:
    model_path = root / 'model_2.pt'
    optimizer_path = root / 'optimizer_2.pt'
    jit_model_path = root / 'model_2.jit.pt'
    replay_path = root / 'memory_1' / 'sample.hdf5'
    replay_path.parent.mkdir(parents=True)

    model_path.write_bytes(b'model')
    optimizer_path.write_bytes(b'optimizer')
    jit_model_path.write_bytes(b'jit')
    replay_path.write_bytes(b'replay')

    manifest = CheckpointManifest(
        iteration=2,
        model_path=model_path.name,
        model_sha256=sha256(model_path),
        optimizer_path=optimizer_path.name,
        optimizer_sha256=sha256(optimizer_path),
        jit_model_path=jit_model_path.name,
        jit_model_sha256=sha256(jit_model_path),
        replay_files=(
            ReplayFileReference(
                path=replay_path.relative_to(root).as_posix(),
                size_bytes=replay_path.stat().st_size,
            ),
        ),
    )
    (root / 'checkpoint_2.json').write_text(
        manifest.model_dump_json(),
        encoding='utf-8',
    )


def test_checkpoint_manifest_validates_artifacts_and_replay(tmp_path: Path) -> None:
    write_checkpoint_fixture(tmp_path)

    manifest = load_checkpoint_manifest(2, tmp_path)

    assert manifest.iteration == 2
    assert manifest.replay_files[0].size_bytes == 6


def test_checkpoint_manifest_rejects_tampered_artifact(tmp_path: Path) -> None:
    write_checkpoint_fixture(tmp_path)
    (tmp_path / 'model_2.pt').write_bytes(b'tampered')

    with pytest.raises(ValueError, match='hash does not match'):
        load_checkpoint_manifest(2, tmp_path)


@pytest.mark.parametrize(
    ('configured_path', 'expected_path'),
    (
        ('model_2.pt', 'model_2.jit.pt'),
        ('model_2.jit.pt', 'model_2.jit.pt'),
    ),
)
def test_inference_model_path_identifies_executable_artifact(
    configured_path: str,
    expected_path: str,
) -> None:
    assert inference_model_path(configured_path) == Path(expected_path)

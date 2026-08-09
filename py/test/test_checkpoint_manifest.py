import hashlib
from pathlib import Path

import pytest

from src.training.checkpoint import CheckpointManifest
from src.training.checkpoint.contracts import load_checkpoint_manifest
from src.training.checkpoint.paths import inference_model_path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checkpoint_fixture(root: Path) -> None:
    model_path = root / 'model_2.pt'
    optimizer_path = root / 'optimizer_2.pt'
    jit_model_path = root / 'model_2.jit.pt'
    model_path.write_bytes(b'model')
    optimizer_path.write_bytes(b'optimizer')
    jit_model_path.write_bytes(b'jit')

    manifest = CheckpointManifest(
        generation=2,
        model_path=model_path.name,
        model_sha256=sha256(model_path),
        optimizer_path=optimizer_path.name,
        optimizer_sha256=sha256(optimizer_path),
        inference_model_path=jit_model_path.name,
        inference_model_sha256=sha256(jit_model_path),
    )
    (root / 'checkpoint_2.json').write_text(
        manifest.model_dump_json(),
        encoding='utf-8',
    )


def test_checkpoint_manifest_validates_artifacts(tmp_path: Path) -> None:
    write_checkpoint_fixture(tmp_path)

    manifest = load_checkpoint_manifest(2, tmp_path)

    assert manifest.generation == 2


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

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from src.games.representation import NetworkDimensions
from src.training.checkpoint import CheckpointManifest
from src.training.checkpoint.contracts import QatCheckpointRecord, load_checkpoint_manifest
from src.training.checkpoint.paths import inference_model_path
from src.training.checkpoint.persistence import import_checkpoint
from src.training.network import GoPointPassPolicyHeadConfiguration, NetworkDefinition, NetworkParams
from src.training.quantization.configuration import QatCheckpointPhase

NETWORK_DEFINITION = NetworkDefinition(
    architecture=NetworkParams(num_layers=1, hidden_size=8, policy_head=GoPointPassPolicyHeadConfiguration()),
    dimensions=NetworkDimensions(channels=3, rows=3, columns=3, actions=10),
    auxiliary_heads=(),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checkpoint_fixture(root: Path, content_prefix: bytes = b'') -> None:
    model_path = root / 'model_2.pt'
    optimizer_path = root / 'optimizer_2.pt'
    jit_model_path = root / 'model_2.jit.pt'
    model_path.write_bytes(content_prefix + b'model')
    optimizer_path.write_bytes(content_prefix + b'optimizer')
    jit_model_path.write_bytes(content_prefix + b'jit')

    manifest = CheckpointManifest(
        generation=2,
        network=NETWORK_DEFINITION,
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


def test_import_checkpoint_preserves_generation_optimizer_and_hashes(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    write_checkpoint_fixture(source)
    destination = tmp_path / 'destination'

    imported = import_checkpoint(source / 'checkpoint_2.json', 2, destination)

    assert imported.generation == 2
    assert imported.model_path.read_bytes() == b'model'
    assert imported.optimizer_path.read_bytes() == b'optimizer'
    assert imported.inference_model_path.read_bytes() == b'jit'
    assert load_checkpoint_manifest(2, destination) == load_checkpoint_manifest(2, source)


def test_import_qat_checkpoint_preserves_onnx_inference_artifact_type(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    model_path = source / 'model_2.pt'
    optimizer_path = source / 'optimizer_2.pt'
    inference_path = source / 'model_2.int8.onnx'
    qat_state_path = source / 'qat_state_2.pt'
    model_path.write_bytes(b'model')
    optimizer_path.write_bytes(b'optimizer')
    inference_path.write_bytes(b'onnx')
    qat_state_path.write_bytes(b'qat')
    manifest = CheckpointManifest(
        generation=2,
        network=NETWORK_DEFINITION,
        model_path=model_path.name,
        model_sha256=sha256(model_path),
        optimizer_path=optimizer_path.name,
        optimizer_sha256=sha256(optimizer_path),
        inference_model_path=inference_path.name,
        inference_model_sha256=sha256(inference_path),
        qat=QatCheckpointRecord(
            phase=QatCheckpointPhase.DEPLOYMENT,
            completed_optimizer_steps=2,
            state_path=qat_state_path.name,
            state_sha256=sha256(qat_state_path),
        ),
    )
    manifest_path = source / 'checkpoint_2.json'
    manifest_path.write_text(manifest.model_dump_json(), encoding='utf-8')

    imported = import_checkpoint(manifest_path, 2, tmp_path / 'destination')

    assert imported.inference_model_path.name == 'model_2.int8.onnx'
    assert imported.inference_model_path.read_bytes() == b'onnx'


def test_import_checkpoint_rejects_wrong_generation(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    write_checkpoint_fixture(source)

    with pytest.raises(ValueError, match='does not match 3'):
        import_checkpoint(source / 'checkpoint_2.json', 3, tmp_path / 'destination')


def test_import_checkpoint_rejects_different_existing_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    write_checkpoint_fixture(source)
    destination = tmp_path / 'destination'
    destination.mkdir()
    write_checkpoint_fixture(destination, b'different-')

    with pytest.raises(ValueError, match='does not match the configured source checkpoint'):
        import_checkpoint(source / 'checkpoint_2.json', 2, destination)


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

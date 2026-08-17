from __future__ import annotations

import hashlib
from os import PathLike
from pathlib import Path

from src.training.checkpoint.paths import checkpoint_manifest_path
from src.training.network import NetworkDefinition
from src.util.frozen_model import FrozenModel


class CheckpointManifest(FrozenModel):
    generation: int
    network: NetworkDefinition
    model_path: str
    model_sha256: str
    optimizer_path: str
    optimizer_sha256: str
    inference_model_path: str
    inference_model_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint_artifact(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f'Checkpoint artifact does not exist: {path}')
    if _sha256(path) != expected_sha256:
        raise ValueError(f'Checkpoint artifact hash does not match: {path}')


def read_checkpoint_manifest(
    generation: int,
    save_folder: str | PathLike[str],
) -> CheckpointManifest:
    manifest_path = checkpoint_manifest_path(generation, save_folder)
    if not manifest_path.is_file():
        raise ValueError(f'Checkpoint manifest does not exist: {manifest_path}')
    manifest = CheckpointManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    if manifest.generation != generation:
        raise ValueError(f'Checkpoint manifest generation {manifest.generation} does not match {generation}.')
    return manifest


def load_checkpoint_manifest(
    generation: int,
    save_folder: str | PathLike[str],
) -> CheckpointManifest:
    manifest = read_checkpoint_manifest(generation, save_folder)
    root = Path(save_folder)
    artifacts = (
        (root / manifest.model_path, manifest.model_sha256),
        (root / manifest.optimizer_path, manifest.optimizer_sha256),
        (root / manifest.inference_model_path, manifest.inference_model_sha256),
    )
    for artifact_path, expected_sha256 in artifacts:
        _validate_checkpoint_artifact(artifact_path, expected_sha256)
    return manifest


def load_inference_checkpoint_manifest(
    generation: int,
    save_folder: str | PathLike[str],
) -> CheckpointManifest:
    manifest = read_checkpoint_manifest(generation, save_folder)
    _validate_checkpoint_artifact(
        Path(save_folder) / manifest.inference_model_path,
        manifest.inference_model_sha256,
    )
    return manifest


def load_checkpoint_manifest_path(path: Path, expected_generation: int) -> CheckpointManifest:
    if not path.is_file():
        raise ValueError(f'Checkpoint manifest does not exist: {path}')
    manifest = CheckpointManifest.model_validate_json(path.read_text(encoding='utf-8'))
    if manifest.generation != expected_generation:
        raise ValueError(f'Checkpoint manifest generation {manifest.generation} does not match {expected_generation}.')
    artifacts = (
        (path.parent / manifest.model_path, manifest.model_sha256),
        (path.parent / manifest.optimizer_path, manifest.optimizer_sha256),
        (path.parent / manifest.inference_model_path, manifest.inference_model_sha256),
    )
    for artifact_path, expected_sha256 in artifacts:
        _validate_checkpoint_artifact(artifact_path, expected_sha256)
    return manifest


class CheckpointReference(FrozenModel):
    generation: int
    manifest_path: Path
    model_path: Path
    optimizer_path: Path
    inference_model_path: Path
    inference_model_sha256: str

    def validate_inference_model(self) -> None:
        if not self.inference_model_path.is_file():
            raise ValueError(f'Inference model does not exist: {self.inference_model_path}')
        digest = hashlib.sha256()
        with self.inference_model_path.open('rb') as model_file:
            while chunk := model_file.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != self.inference_model_sha256:
            raise ValueError(f'Inference model hash does not match: {self.inference_model_path}')

    @classmethod
    def load(cls, run_path: Path, generation: int) -> CheckpointReference:
        manifest = load_checkpoint_manifest(generation, run_path)
        return cls.from_manifest(run_path, manifest)

    @classmethod
    def load_for_inference(cls, run_path: Path, generation: int) -> CheckpointReference:
        manifest = load_inference_checkpoint_manifest(generation, run_path)
        return cls.from_manifest(run_path, manifest)

    @classmethod
    def from_manifest(cls, run_path: Path, manifest: CheckpointManifest) -> CheckpointReference:
        return cls(
            generation=manifest.generation,
            manifest_path=checkpoint_manifest_path(manifest.generation, run_path),
            model_path=run_path / manifest.model_path,
            optimizer_path=run_path / manifest.optimizer_path,
            inference_model_path=run_path / manifest.inference_model_path,
            inference_model_sha256=manifest.inference_model_sha256,
        )

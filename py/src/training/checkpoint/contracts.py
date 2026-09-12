from __future__ import annotations

from os import PathLike
from pathlib import Path

from pydantic import Field
from src.training.checkpoint.paths import checkpoint_manifest_path
from src.training.network import NetworkDefinition
from src.training.quantization.configuration import QatCheckpointPhase, QatStateIdentity
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256


class BootstrapPolicyPriorRecord(FrozenModel):
    initial_top1_mass: float
    initial_top3_mass: float
    calibrated_top1_mass: float
    calibrated_top3_mass: float
    target_top3_mass: float
    applied_scale: float


class QatCheckpointRecord(FrozenModel):
    phase: QatCheckpointPhase
    completed_optimizer_steps: int = Field(ge=0)
    state_path: str
    state_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class CheckpointManifest(FrozenModel):
    generation: int
    network: NetworkDefinition
    model_path: str
    model_sha256: str
    optimizer_path: str
    optimizer_sha256: str
    inference_model_path: str
    inference_model_sha256: str
    qat: QatCheckpointRecord | None = None
    policy_prior_calibration: BootstrapPolicyPriorRecord | None = None


def _validate_checkpoint_artifact(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f'Checkpoint artifact does not exist: {path}')
    if file_sha256(path) != expected_sha256:
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
    _validate_qat_artifact(root, manifest)
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
    _validate_qat_artifact(path.parent, manifest)
    return manifest


def _validate_qat_artifact(root: Path, manifest: CheckpointManifest) -> None:
    if manifest.qat is None:
        return
    _validate_checkpoint_artifact(root / manifest.qat.state_path, manifest.qat.state_sha256)


class CheckpointReference(FrozenModel):
    generation: int
    manifest_path: Path
    model_path: Path
    optimizer_path: Path
    inference_model_path: Path
    inference_model_sha256: str
    qat_state: QatStateIdentity | None = None

    def validate_inference_model(self) -> None:
        if not self.inference_model_path.is_file():
            raise ValueError(f'Inference model does not exist: {self.inference_model_path}')
        if file_sha256(self.inference_model_path) != self.inference_model_sha256:
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
        qat_state = None
        if manifest.qat is not None:
            qat_state = QatStateIdentity(
                phase=manifest.qat.phase,
                completed_optimizer_steps=manifest.qat.completed_optimizer_steps,
                path=run_path / manifest.qat.state_path,
                sha256=manifest.qat.state_sha256,
            )
        return cls(
            generation=manifest.generation,
            manifest_path=checkpoint_manifest_path(manifest.generation, run_path),
            model_path=run_path / manifest.model_path,
            optimizer_path=run_path / manifest.optimizer_path,
            inference_model_path=run_path / manifest.inference_model_path,
            inference_model_sha256=manifest.inference_model_sha256,
            qat_state=qat_state,
        )

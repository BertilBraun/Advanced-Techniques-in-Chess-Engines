from __future__ import annotations

import hashlib
from pathlib import Path

from src.util.frozen_model import FrozenModel
from src.util.save_paths import (
    CheckpointManifest,
    checkpoint_manifest_path,
    load_checkpoint_manifest,
    load_inference_checkpoint_manifest,
)


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
            generation=manifest.iteration,
            manifest_path=checkpoint_manifest_path(manifest.iteration, run_path),
            model_path=run_path / manifest.model_path,
            optimizer_path=run_path / manifest.optimizer_path,
            inference_model_path=run_path / manifest.jit_model_path,
            inference_model_sha256=manifest.jit_model_sha256,
        )

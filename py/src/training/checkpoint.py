from __future__ import annotations

from pathlib import Path

from src.util.frozen_model import FrozenModel
from src.util.save_paths import CheckpointManifest, checkpoint_manifest_path, load_checkpoint_manifest


class CheckpointReference(FrozenModel):
    generation: int
    manifest_path: Path
    model_path: Path
    optimizer_path: Path
    inference_model_path: Path
    inference_model_sha256: str

    @classmethod
    def load(cls, run_path: Path, generation: int) -> CheckpointReference:
        manifest = load_checkpoint_manifest(generation, run_path)
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

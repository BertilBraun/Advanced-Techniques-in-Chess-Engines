from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal
from uuid import UUID

from pydantic import model_validator

from src.az.config.base import FrozenModel, Sha256


class CalibrationArtifactReference(FrozenModel):
    artifact_root: Literal['reference_artifacts']
    artifact_id: UUID
    path: PurePosixPath
    sha256: Sha256

    @model_validator(mode='after')
    def validate_path(self) -> CalibrationArtifactReference:
        if self.path.is_absolute() or '..' in self.path.parts:
            raise ValueError('Calibration artifact path must stay below its declared artifact root.')
        return self


class CheckpointArtifactReference(FrozenModel):
    artifact_root: Literal['reference_artifacts']
    manifest_path: PurePosixPath
    manifest_sha256: Sha256
    model_path: PurePosixPath
    model_artifact_sha256: Sha256

    @model_validator(mode='after')
    def validate_path(self) -> CheckpointArtifactReference:
        for path in (self.manifest_path, self.model_path):
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Checkpoint artifact path must stay below its declared artifact root.')
        return self

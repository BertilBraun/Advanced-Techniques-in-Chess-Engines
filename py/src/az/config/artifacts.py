from __future__ import annotations

from pathlib import PurePosixPath
from uuid import UUID

from src.az.config.base import FrozenModel, Sha256


class CalibrationArtifactReference(FrozenModel):
    artifact_id: UUID
    path: PurePosixPath
    sha256: Sha256

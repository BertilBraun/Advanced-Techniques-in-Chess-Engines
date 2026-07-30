from __future__ import annotations

import hashlib
import os
from pathlib import Path

from src.az.evaluation.models import CandidateCheckpointIdentity
from src.az.training.checkpoints import LoadedModelCheckpoint


class EvaluationModelArtifactRepository:
    """Retains immutable model bytes so historical evaluations remain resumable."""

    def __init__(self, directory: Path) -> None:
        if not directory.is_absolute():
            raise ValueError('Evaluation model artifact directory must be absolute.')
        self._directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def claim(self, checkpoint: LoadedModelCheckpoint) -> CandidateCheckpointIdentity:
        identity = CandidateCheckpointIdentity(
            checkpoint_id=checkpoint.manifest.checkpoint_id,
            model_artifact_sha256=checkpoint.manifest.model.sha256,
            model_version=checkpoint.manifest.model_version,
        )
        if hashlib.sha256(checkpoint.model_artifact).hexdigest() != identity.model_artifact_sha256:
            raise ValueError('Claimed evaluation model bytes do not match the checkpoint manifest.')
        path = self._path(identity)
        if path.exists():
            if path.read_bytes() != checkpoint.model_artifact:
                raise ValueError('Evaluation model identity already resolves to different bytes.')
            return identity
        partial = path.with_suffix('.partial')
        if partial.exists():
            partial.unlink()
        with partial.open('xb') as stream:
            stream.write(checkpoint.model_artifact)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        return identity

    def load(self, identity: CandidateCheckpointIdentity) -> bytes:
        path = self._path(identity)
        if not path.is_file():
            raise ValueError('Claimed evaluation model artifact is unavailable.')
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != identity.model_artifact_sha256:
            raise ValueError('Claimed evaluation model artifact checksum mismatch.')
        return contents

    def _path(self, identity: CandidateCheckpointIdentity) -> Path:
        return self._directory / (
            f'model-{identity.model_version:010d}-{identity.checkpoint_id.hex}-{identity.model_artifact_sha256}.pt'
        )

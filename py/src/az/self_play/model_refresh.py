from __future__ import annotations

from src.az.training.checkpoints import CheckpointRepository, LoadedModelCheckpoint


def load_newer_model_checkpoint(
    repository: CheckpointRepository,
    current_model_version: int,
) -> LoadedModelCheckpoint | None:
    available_version = repository.current_model_version()
    if available_version is None or available_version <= current_model_version:
        return None
    checkpoint = repository.load_current_model()
    if checkpoint.manifest.model_version <= current_model_version:
        return None
    return checkpoint

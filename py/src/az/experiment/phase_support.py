from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from pydantic import Field

from src.az.config.base import FrozenModel, Sha256
from src.az.experiment.lifecycle import (
    ExperimentPhase,
    ExperimentRunRepository,
    ExperimentRunState,
    ExperimentStatus,
    RunArtifact,
    RunArtifactKind,
)
from src.az.evaluation.models import CandidateCheckpointIdentity


class ScheduledCheckpointClaim(FrozenModel):
    run_id: UUID
    resolved_configuration_sha256: Sha256
    requested_elapsed_seconds: int = Field(gt=0)
    published_elapsed_seconds: float = Field(ge=0)
    candidate: CandidateCheckpointIdentity


def begin_phase(
    repository: ExperimentRunRepository,
    phase: ExperimentPhase,
) -> ExperimentRunState:
    state = repository.load()
    if state.next_phase is not phase:
        raise ValueError(f'Run expects {state.next_phase.value}, not {phase.value}.')
    if repository.stop_requested():
        raise ValueError('Run has an authenticated stop request; use resume to clear it.')
    running = state.model_copy(
        update={
            'status': ExperimentStatus.RUNNING,
            'stop_requested': False,
            'updated_at': datetime.now(timezone.utc),
        }
    )
    repository.save(state, running)
    repository.acquire_lease(running)
    return running


def complete_phase(
    repository: ExperimentRunRepository,
    state: ExperimentRunState,
    phase: ExperimentPhase,
    artifacts: tuple[RunArtifact, ...],
    self_play_elapsed_seconds: float | None = None,
    checkpoint_published_elapsed_seconds: float | None = None,
) -> ExperimentRunState:
    completed = (*state.completed_phases, phase)
    next_phase = (
        ExperimentPhase.COMPLETE
        if phase is ExperimentPhase.REPORTING
        else ExperimentPhase.EVALUATION
        if phase is ExperimentPhase.TRAINING_RUN
        else ExperimentPhase.REPORTING
    )
    updated = state.model_copy(
        update={
            'status': ExperimentStatus.COMPLETE if next_phase is ExperimentPhase.COMPLETE else ExperimentStatus.READY,
            'next_phase': next_phase,
            'completed_phases': completed,
            'artifacts': merge_artifacts(state.artifacts, artifacts),
            'self_play_elapsed_seconds': (
                state.self_play_elapsed_seconds if self_play_elapsed_seconds is None else self_play_elapsed_seconds
            ),
            'checkpoint_published_elapsed_seconds': (
                state.checkpoint_published_elapsed_seconds
                if checkpoint_published_elapsed_seconds is None
                else checkpoint_published_elapsed_seconds
            ),
            'updated_at': datetime.now(timezone.utc),
        }
    )
    repository.save(state, updated)
    repository.release_lease(updated)
    return updated


def interrupt_phase(
    repository: ExperimentRunRepository,
    state: ExperimentRunState,
    elapsed: float | None,
    artifacts: tuple[RunArtifact, ...],
) -> ExperimentRunState:
    updated = state.model_copy(
        update={
            'status': ExperimentStatus.STOPPED,
            'stop_requested': True,
            'artifacts': merge_artifacts(state.artifacts, artifacts),
            'self_play_elapsed_seconds': elapsed,
            'updated_at': datetime.now(timezone.utc),
        }
    )
    repository.save(state, updated)
    repository.release_lease(updated)
    return updated


def registered_artifact_paths(
    repository: ExperimentRunRepository,
    state: ExperimentRunState,
    kind: RunArtifactKind,
) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (repository.directory / artifact.relative_path).resolve()
            for artifact in state.artifacts
            if artifact.kind is kind
        )
    )


def merge_artifacts(
    existing: tuple[RunArtifact, ...],
    additions: tuple[RunArtifact, ...],
) -> tuple[RunArtifact, ...]:
    by_path = {artifact.relative_path: artifact for artifact in existing}
    for artifact in additions:
        by_path[artifact.relative_path] = artifact
    return tuple(sorted(by_path.values(), key=lambda artifact: artifact.relative_path))

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from src.az.calibration.calibrate import calibrate_from_committed_trace_artifacts
from src.az.calibration.models import (
    MaximumMeanDisagreementRule,
    VisitMarginCandidate,
    load_trace_collection_artifact,
    publish_calibration_artifact,
)
from src.az.config.artifacts import CalibrationArtifactReference
from src.az.config.base import FrozenModel
from src.az.experiment.commit_journal import ReplayCommitJournal
from src.az.experiment.lifecycle import (
    ExperimentRunRepository,
    RunArtifact,
    RunArtifactKind,
    require_exact_artifact_files,
)


class AdaptiveCalibrationRequest(FrozenModel):
    schema_version: Literal[1] = 1
    artifact_id: UUID
    candidates: tuple[VisitMarginCandidate, ...] = Field(min_length=1)
    acceptance_rule: MaximumMeanDisagreementRule

    @model_validator(mode='after')
    def validate_candidate_grid(self) -> AdaptiveCalibrationRequest:
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError('Adaptive calibration candidate grid must be unique.')
        return self


def calibrate_run(
    repository: ExperimentRunRepository,
    request: AdaptiveCalibrationRequest,
) -> CalibrationArtifactReference:
    state = repository.load()
    commit_artifact = _single_artifact(
        state.artifacts,
        RunArtifactKind.REPLAY_COMMIT_JOURNAL,
    )
    commit_path = _artifact_path(repository, commit_artifact)
    commit_journal = ReplayCommitJournal(commit_path)
    _verify_unchanged(commit_path, commit_artifact)

    trace_artifacts = tuple(artifact for artifact in state.artifacts if artifact.kind is RunArtifactKind.SEARCH_TRACE)
    if not trace_artifacts:
        raise ValueError('Calibration requires registered search trace artifacts.')
    require_exact_artifact_files(
        repository.directory / 'search-traces',
        'trace-*.json',
        tuple(_artifact_path(repository, artifact) for artifact in trace_artifacts),
    )
    loaded_traces = tuple(
        load_trace_collection_artifact(_artifact_path(repository, artifact)) for artifact in trace_artifacts
    )
    calibration = calibrate_from_committed_trace_artifacts(
        artifact_id=request.artifact_id,
        loaded_artifacts=loaded_traces,
        committed_replay_sample_ids=commit_journal.sample_ids,
        candidates=request.candidates,
        acceptance_rule=request.acceptance_rule,
    )
    if (
        calibration.payload.source_run_id != state.run_id
        or calibration.payload.source_configuration_sha256 != state.resolved_configuration_sha256
    ):
        raise ValueError('Calibration traces do not belong to the authenticated run.')

    path = publish_calibration_artifact(
        (repository.directory / 'calibrations').resolve(),
        calibration,
    )
    repository.register_artifact(RunArtifactKind.CALIBRATION, path)
    return CalibrationArtifactReference(
        artifact_root='reference_artifacts',
        artifact_id=request.artifact_id,
        path=PurePosixPath(path.relative_to(repository.directory).as_posix()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def load_calibration_request(path: Path) -> AdaptiveCalibrationRequest:
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Calibration request must identify an absolute JSON file.')
    return AdaptiveCalibrationRequest.model_validate_json(path.read_bytes())


def _single_artifact(
    artifacts: tuple[RunArtifact, ...],
    kind: RunArtifactKind,
) -> RunArtifact:
    matching = tuple(artifact for artifact in artifacts if artifact.kind is kind)
    if len(matching) != 1:
        raise ValueError(f'Calibration requires exactly one registered {kind.value}.')
    return matching[0]


def _artifact_path(
    repository: ExperimentRunRepository,
    artifact: RunArtifact,
) -> Path:
    path = (repository.directory / artifact.relative_path).resolve()
    if repository.directory.resolve() not in path.parents:
        raise ValueError('Registered calibration input is outside the run directory.')
    return path


def _verify_unchanged(path: Path, artifact: RunArtifact) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact.sha256:
        raise ValueError('Replay commit journal changed during authentication.')

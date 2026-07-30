from __future__ import annotations

import hashlib
import os
import socket
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from src.az.config.base import FrozenModel, Sha256
from src.az.config.artifacts import CalibrationArtifactReference
from src.az.config.manifest import (
    RunManifest,
    inspect_source_revision,
    inspect_source_state,
)
from src.az.config.serialization import load_resolved_configuration, model_sha256
from src.az.config.search import VisitMarginAdaptiveRule
from src.az.games.go.configuration import CheckpointGoOpponent, RandomGoOpponent


class ExperimentPhase(str, Enum):
    TRAINING_RUN = 'training_run'
    EVALUATION = 'evaluation'
    REPORTING = 'reporting'
    COMPLETE = 'complete'


class ExperimentStatus(str, Enum):
    READY = 'ready'
    RUNNING = 'running'
    STOPPED = 'stopped'
    FAILED = 'failed'
    COMPLETE = 'complete'


class RunArtifactKind(str, Enum):
    RESOLVED_CONFIGURATION = 'resolved_configuration'
    RUN_MANIFEST = 'run_manifest'
    REPLAY_SHARD = 'replay_shard'
    REPLAY_COMMIT_JOURNAL = 'replay_commit_journal'
    RUNTIME_TELEMETRY = 'runtime_telemetry'
    CHECKPOINT = 'checkpoint'
    CHECKPOINT_POINTER = 'checkpoint_pointer'
    CHECKPOINT_CLAIM = 'checkpoint_claim'
    EVALUATION_MODEL = 'evaluation_model'
    EVALUATION_RESULT = 'evaluation_result'
    SEARCH_TRACE = 'search_trace'
    CALIBRATION = 'calibration'
    RESEARCH_REPORT = 'research_report'
    REFERENCE_ARTIFACT = 'reference_artifact'


class RunArtifact(FrozenModel):
    kind: RunArtifactKind
    relative_path: str = Field(min_length=1)
    sha256: Sha256


class ExperimentRunState(FrozenModel):
    schema_version: Literal[1] = 1
    run_id: UUID
    resolved_configuration_sha256: Sha256
    source_revision: str = Field(min_length=1)
    source_repository_root: Path
    status: ExperimentStatus
    next_phase: ExperimentPhase
    completed_phases: tuple[ExperimentPhase, ...]
    artifacts: tuple[RunArtifact, ...]
    self_play_elapsed_seconds: float | None = Field(default=None, ge=0)
    checkpoint_published_elapsed_seconds: float | None = Field(default=None, ge=0)
    stop_requested: bool = False
    failure: str | None = None
    updated_at: datetime

    @model_validator(mode='after')
    def validate_progress(self) -> ExperimentRunState:
        phase_order = (
            ExperimentPhase.TRAINING_RUN,
            ExperimentPhase.EVALUATION,
            ExperimentPhase.REPORTING,
        )
        expected_completed = phase_order[: len(self.completed_phases)]
        if self.completed_phases != expected_completed:
            raise ValueError('Completed experiment phases must be an ordered prefix.')
        expected_next = (
            ExperimentPhase.COMPLETE
            if len(self.completed_phases) == len(phase_order)
            else phase_order[len(self.completed_phases)]
        )
        if self.next_phase is not expected_next:
            raise ValueError('Next experiment phase does not follow completed phases.')
        if self.status is ExperimentStatus.COMPLETE and self.next_phase is not ExperimentPhase.COMPLETE:
            raise ValueError('Only a fully completed run can have complete status.')
        if self.next_phase is ExperimentPhase.COMPLETE and self.status is not ExperimentStatus.COMPLETE:
            raise ValueError('A fully completed run must have complete status.')
        if (self.status is ExperimentStatus.FAILED) != (self.failure is not None):
            raise ValueError('Only failed runs carry failure evidence.')
        if not self.source_repository_root.is_absolute():
            raise ValueError('Source repository root must be absolute.')
        paths = tuple(artifact.relative_path for artifact in self.artifacts)
        if len(set(paths)) != len(paths):
            raise ValueError('Run artifact paths must be unique.')
        return self


class StopRequest(FrozenModel):
    schema_version: Literal[1] = 1
    run_id: UUID
    resolved_configuration_sha256: Sha256
    requested_at: datetime


class RunLease(FrozenModel):
    run_id: UUID
    resolved_configuration_sha256: Sha256
    process_id: int = Field(gt=0)
    host_name: str = Field(min_length=1)
    acquired_at: datetime


class ExperimentRunRepository:
    CONFIGURATION_FILENAME = 'resolved-configuration.json'
    STATE_FILENAME = 'run-state.json'
    STOP_FILENAME = 'stop-request.json'
    MANIFEST_FILENAME = 'run-manifest.json'
    LEASE_FILENAME = 'run-lease.json'

    def __init__(self, run_directory: Path) -> None:
        if not run_directory.is_absolute():
            raise ValueError('Experiment run directory must be absolute.')
        self._directory = run_directory

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def configuration_path(self) -> Path:
        return self._directory / self.CONFIGURATION_FILENAME

    @property
    def state_path(self) -> Path:
        return self._directory / self.STATE_FILENAME

    @property
    def stop_path(self) -> Path:
        return self._directory / self.STOP_FILENAME

    @property
    def lease_path(self) -> Path:
        return self._directory / self.LEASE_FILENAME

    def freeze(
        self,
        configuration_path: Path,
        run_id: UUID,
        manifest: RunManifest,
        source_repository_root: Path,
        reference_artifact_root: Path | None = None,
    ) -> ExperimentRunState:
        if self._directory.exists():
            raise ValueError('Experiment run directory already exists; refusing to overwrite it.')
        if source_repository_root.resolve() in self._directory.resolve().parents:
            raise ValueError('Experiment run directory must be outside the authenticated source repository.')
        configuration = load_resolved_configuration(configuration_path)
        configuration_sha256 = model_sha256(configuration)
        if manifest.configuration != configuration or manifest.configuration_sha256 != configuration_sha256:
            raise ValueError('Run manifest does not authenticate the resolved configuration.')
        self._directory.mkdir(parents=True)
        configuration_contents = configuration.model_dump_json(indent=2).encode() + b'\n'
        self._atomic_create(self.configuration_path, configuration_contents)
        manifest_contents = manifest.model_dump_json(indent=2).encode() + b'\n'
        self._atomic_create(self._directory / self.MANIFEST_FILENAME, manifest_contents)
        match configuration.search.stopping:
            case VisitMarginAdaptiveRule(calibration=calibration):
                pass
            case _:
                calibration = None
        reference_artifacts = self._copy_reference_artifacts(
            configuration.evaluation.suite.opponent,
            calibration,
            reference_artifact_root,
        )
        state = ExperimentRunState(
            run_id=run_id,
            resolved_configuration_sha256=configuration_sha256,
            source_revision=manifest.source.revision,
            source_repository_root=source_repository_root.resolve(),
            status=ExperimentStatus.READY,
            next_phase=ExperimentPhase.TRAINING_RUN,
            completed_phases=(),
            artifacts=(
                RunArtifact(
                    kind=RunArtifactKind.RESOLVED_CONFIGURATION,
                    relative_path=self.CONFIGURATION_FILENAME,
                    sha256=hashlib.sha256(configuration_contents).hexdigest(),
                ),
                RunArtifact(
                    kind=RunArtifactKind.RUN_MANIFEST,
                    relative_path=self.MANIFEST_FILENAME,
                    sha256=hashlib.sha256(manifest_contents).hexdigest(),
                ),
                *reference_artifacts,
            ),
            updated_at=datetime.now(timezone.utc),
        )
        self._atomic_create(self.state_path, state.model_dump_json(indent=2).encode() + b'\n')
        return state

    def _copy_reference_artifacts(
        self,
        opponent: RandomGoOpponent | CheckpointGoOpponent,
        calibration: CalibrationArtifactReference | None,
        source_root: Path | None,
    ) -> tuple[RunArtifact, ...]:
        references: list[tuple[PurePosixPath, Sha256]] = []
        match opponent:
            case RandomGoOpponent():
                pass
            case CheckpointGoOpponent(checkpoint=reference):
                references.extend(
                    (
                        (reference.manifest_path, reference.manifest_sha256),
                        (reference.model_path, reference.model_artifact_sha256),
                    )
                )
        if calibration is not None:
            references.append((calibration.path, calibration.sha256))
        if not references:
            if source_root is not None:
                raise ValueError('Configuration has no artifacts to copy from a reference root.')
            return ()
        if source_root is None or not source_root.is_absolute():
            raise ValueError('Configured reference artifacts require an absolute reference artifact root.')
        copied: list[RunArtifact] = []
        for relative_path, expected_sha256 in references:
            source = source_root.joinpath(*relative_path.parts).resolve()
            if source_root.resolve() not in source.parents or not source.is_file():
                raise ValueError('Reference artifact is outside its declared source root.')
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
                raise ValueError('Reference artifact checksum mismatch during freeze.')
            destination = self._directory / 'reference-artifacts'
            destination = destination.joinpath(*relative_path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.append(self.artifact(RunArtifactKind.REFERENCE_ARTIFACT, destination))
        return tuple(copied)

    def load(self) -> ExperimentRunState:
        if not self.state_path.is_file() or not self.configuration_path.is_file():
            raise ValueError('Experiment run is not frozen or is incomplete.')
        state = ExperimentRunState.model_validate_json(self.state_path.read_bytes())
        configuration = load_resolved_configuration(self.configuration_path)
        manifest_path = self._directory / self.MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ValueError('Experiment run manifest is missing.')
        manifest = RunManifest.model_validate_json(manifest_path.read_bytes())
        if model_sha256(configuration) != state.resolved_configuration_sha256:
            raise ValueError('Frozen configuration does not match the run-state identity.')
        if (
            manifest.configuration != configuration
            or manifest.configuration_sha256 != state.resolved_configuration_sha256
            or manifest.source.revision != state.source_revision
        ):
            raise ValueError('Run manifest does not match the frozen run identity.')
        if configuration.hardware.profile_name == 'local-cpu-smoke':
            if inspect_source_revision(state.source_repository_root) != manifest.source.revision:
                raise ValueError('Current source revision does not match the smoke run manifest.')
        elif inspect_source_state(state.source_repository_root) != manifest.source:
            raise ValueError('Current source state does not match the frozen run manifest.')
        configuration_artifact = next(
            (artifact for artifact in state.artifacts if artifact.kind is RunArtifactKind.RESOLVED_CONFIGURATION),
            None,
        )
        if (
            configuration_artifact is None
            or configuration_artifact.sha256 != hashlib.sha256(self.configuration_path.read_bytes()).hexdigest()
        ):
            raise ValueError('Frozen configuration artifact checksum mismatch.')
        self._authenticate_artifacts(state)
        return state

    def save(self, previous: ExperimentRunState, updated: ExperimentRunState) -> None:
        current = self.load()
        if current != previous:
            raise ValueError('Run state changed concurrently; refusing to overwrite it.')
        if (
            updated.run_id != previous.run_id
            or updated.resolved_configuration_sha256 != previous.resolved_configuration_sha256
            or updated.source_revision != previous.source_revision
            or updated.source_repository_root != previous.source_repository_root
        ):
            raise ValueError('Run identity is immutable.')
        self._atomic_replace(self.state_path, updated.model_dump_json(indent=2).encode() + b'\n')

    def request_stop(self) -> StopRequest:
        state = self.load()
        if state.status is ExperimentStatus.COMPLETE:
            raise ValueError('A completed run cannot be stopped.')
        request = StopRequest(
            run_id=state.run_id,
            resolved_configuration_sha256=state.resolved_configuration_sha256,
            requested_at=datetime.now(timezone.utc),
        )
        contents = request.model_dump_json(indent=2).encode() + b'\n'
        if self.stop_path.exists():
            existing = StopRequest.model_validate_json(self.stop_path.read_bytes())
            self._validate_stop_request(existing, state)
            return existing
        self._atomic_create(self.stop_path, contents)
        return request

    def stop_requested(self) -> bool:
        if not self.stop_path.exists():
            return False
        if not self.state_path.is_file():
            raise ValueError('Experiment run state is missing.')
        state = ExperimentRunState.model_validate_json(self.state_path.read_bytes())
        request = StopRequest.model_validate_json(self.stop_path.read_bytes())
        self._validate_stop_request(request, state)
        return True

    def clear_stop_request(self) -> None:
        if not self.stop_path.exists():
            return
        state = self.load()
        request = StopRequest.model_validate_json(self.stop_path.read_bytes())
        self._validate_stop_request(request, state)
        self.stop_path.unlink()

    def resume(self, recover_crash: bool = False) -> ExperimentRunState:
        state = self.load()
        if state.status is ExperimentStatus.STOPPED:
            if not self.stop_path.exists() or not state.stop_requested:
                raise ValueError('Interrupted run is missing its authenticated stop request.')
            request = StopRequest.model_validate_json(self.stop_path.read_bytes())
            self._validate_stop_request(request, state)
            self.stop_path.unlink()
        elif state.status in (ExperimentStatus.RUNNING, ExperimentStatus.FAILED):
            if not recover_crash:
                raise ValueError('Crash recovery requires the explicit recover-crash option.')
            self._recover_lease(state)
        else:
            raise ValueError('Only interrupted or failed runs can be resumed.')
        resumed = state.model_copy(
            update={
                'status': ExperimentStatus.READY,
                'stop_requested': False,
                'failure': None,
                'updated_at': datetime.now(timezone.utc),
            }
        )
        self.save(state, resumed)
        return resumed

    def acquire_lease(self, state: ExperimentRunState) -> None:
        if self.lease_path.exists():
            raise ValueError('Run already has an execution lease.')
        lease = RunLease(
            run_id=state.run_id,
            resolved_configuration_sha256=state.resolved_configuration_sha256,
            process_id=os.getpid(),
            host_name=socket.gethostname(),
            acquired_at=datetime.now(timezone.utc),
        )
        self._atomic_create(self.lease_path, lease.model_dump_json(indent=2).encode() + b'\n')

    def release_lease(self, state: ExperimentRunState) -> None:
        if not self.lease_path.exists():
            return
        lease = RunLease.model_validate_json(self.lease_path.read_bytes())
        self._validate_lease(lease, state)
        if lease.process_id != os.getpid() or lease.host_name != socket.gethostname():
            raise ValueError('Only the lease owner can release an active run lease.')
        self.lease_path.unlink()

    def record_failure(self, message: str) -> ExperimentRunState:
        state = self.load()
        if state.status is not ExperimentStatus.RUNNING:
            return state
        failed = state.model_copy(
            update={
                'status': ExperimentStatus.FAILED,
                'failure': message,
                'updated_at': datetime.now(timezone.utc),
            }
        )
        self.save(state, failed)
        self.release_lease(failed)
        return failed

    def _recover_lease(self, state: ExperimentRunState) -> None:
        if not self.lease_path.exists():
            return
        lease = RunLease.model_validate_json(self.lease_path.read_bytes())
        self._validate_lease(lease, state)
        if lease.host_name == socket.gethostname() and _process_is_running(lease.process_id):
            raise ValueError('Run lease owner is still active.')
        self.lease_path.unlink()

    def artifact(self, kind: RunArtifactKind, path: Path) -> RunArtifact:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._directory.resolve())
        except ValueError as error:
            raise ValueError('Run artifacts must be inside the run directory.') from error
        if not resolved.is_file():
            raise ValueError('Run artifact does not exist.')
        return RunArtifact(
            kind=kind,
            relative_path=relative.as_posix(),
            sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        )

    def register_artifact(self, kind: RunArtifactKind, path: Path) -> ExperimentRunState:
        state = self.load()
        if state.status is ExperimentStatus.RUNNING:
            raise ValueError('Cannot register a side artifact while the run is active.')
        artifact = self.artifact(kind, path)
        existing = {registered.relative_path: registered for registered in state.artifacts}
        registered = existing.get(artifact.relative_path)
        if registered is not None:
            if registered != artifact:
                raise ValueError('Registered artifact path has different authenticated contents.')
            return state
        updated = state.model_copy(
            update={
                'artifacts': tuple(
                    sorted(
                        (*state.artifacts, artifact),
                        key=lambda item: item.relative_path,
                    )
                ),
                'updated_at': datetime.now(timezone.utc),
            }
        )
        self.save(state, updated)
        return updated

    def complete_training_at_stop(
        self,
        running: ExperimentRunState,
        artifacts: tuple[RunArtifact, ...],
        elapsed_seconds: float,
        checkpoint_elapsed_seconds: float,
    ) -> ExperimentRunState:
        if (
            running.status is not ExperimentStatus.RUNNING
            or running.next_phase is not ExperimentPhase.TRAINING_RUN
            or not self.stop_requested()
        ):
            raise ValueError('Stopped training completion requires an active training run and stop request.')
        by_path = {artifact.relative_path: artifact for artifact in running.artifacts}
        for artifact in artifacts:
            existing = by_path.get(artifact.relative_path)
            if existing is not None and existing != artifact:
                raise ValueError('Training completion artifact conflicts with registered evidence.')
            by_path[artifact.relative_path] = artifact
        stopped = running.model_copy(
            update={
                'status': ExperimentStatus.STOPPED,
                'next_phase': ExperimentPhase.EVALUATION,
                'completed_phases': (ExperimentPhase.TRAINING_RUN,),
                'artifacts': tuple(sorted(by_path.values(), key=lambda artifact: artifact.relative_path)),
                'self_play_elapsed_seconds': elapsed_seconds,
                'checkpoint_published_elapsed_seconds': checkpoint_elapsed_seconds,
                'stop_requested': True,
                'updated_at': datetime.now(timezone.utc),
            }
        )
        self.save(running, stopped)
        self.release_lease(stopped)
        return stopped

    def _authenticate_artifacts(self, state: ExperimentRunState) -> None:
        for artifact in state.artifacts:
            path = (self._directory / artifact.relative_path).resolve()
            if self._directory.resolve() not in path.parents or not path.is_file():
                raise ValueError('Run artifact is missing or outside the run directory.')
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact.sha256:
                raise ValueError(f'Run artifact checksum mismatch: {artifact.relative_path}.')

    @staticmethod
    def _validate_stop_request(request: StopRequest, state: ExperimentRunState) -> None:
        if (
            request.run_id != state.run_id
            or request.resolved_configuration_sha256 != state.resolved_configuration_sha256
        ):
            raise ValueError('Stop request does not belong to the active run identity.')

    @staticmethod
    def _validate_lease(lease: RunLease, state: ExperimentRunState) -> None:
        if lease.run_id != state.run_id or lease.resolved_configuration_sha256 != state.resolved_configuration_sha256:
            raise ValueError('Run lease does not belong to the active run identity.')

    @staticmethod
    def _atomic_create(path: Path, contents: bytes) -> None:
        with path.open('xb') as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _atomic_replace(path: Path, contents: bytes) -> None:
        temporary = path.with_suffix(path.suffix + '.partial')
        if temporary.exists():
            temporary.unlink()
        with temporary.open('xb') as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def _process_is_running(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except OSError:
        return False
    return True


def require_exact_artifact_files(
    directory: Path,
    pattern: str,
    expected: tuple[Path, ...],
) -> None:
    actual = frozenset(path.resolve() for path in directory.glob(pattern))
    if actual != frozenset(path.resolve() for path in expected):
        raise ValueError(f'{directory.name} contains missing or unregistered files.')

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from src.az.config.base import FrozenModel, Sha256
from src.az.config.artifacts import CalibrationArtifactReference
from src.az.config.seeds import (
    SEED_DERIVATION_VERSION,
    SearchTraceSampleSeedCoordinates,
    SeedDerivationVersion,
    SeedPurpose,
    derive_seed,
)


class SearchTraceSnapshot(FrozenModel):
    simulations: int = Field(gt=0)
    root_policy: tuple[float, ...] = Field(min_length=2)
    root_visits: tuple[int, ...] = Field(min_length=2)
    root_value: float = Field(ge=-1, le=1)

    @model_validator(mode='after')
    def validate_snapshot(self) -> SearchTraceSnapshot:
        if len(self.root_policy) != len(self.root_visits):
            raise ValueError('Trace policy and visit vectors must have equal length.')
        if any(probability < 0 for probability in self.root_policy):
            raise ValueError('Trace policy probabilities cannot be negative.')
        if any(visits < 0 for visits in self.root_visits):
            raise ValueError('Trace visit counts cannot be negative.')
        if sum(self.root_visits) != self.simulations:
            raise ValueError('Trace root visits must sum to the simulation count.')
        if abs(sum(self.root_policy) - 1.0) > 1e-9:
            raise ValueError('Trace root policy must sum to one.')
        return self


class SearchTraceObservation(FrozenModel):
    source_position_id: UUID
    full: SearchTraceSnapshot
    prefixes: tuple[SearchTraceSnapshot, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_prefixes(self) -> SearchTraceObservation:
        checkpoints = tuple(prefix.simulations for prefix in self.prefixes)
        if tuple(sorted(set(checkpoints))) != checkpoints:
            raise ValueError('Trace checkpoints must increase strictly.')
        if checkpoints[-1] >= self.full.simulations:
            raise ValueError('Trace prefixes must precede the full search.')
        if any(len(prefix.root_policy) != len(self.full.root_policy) for prefix in self.prefixes):
            raise ValueError('Trace snapshots must share one action space.')
        return self


class SearchTraceSampleLineage(FrozenModel):
    derivation_version: SeedDerivationVersion
    root_seed: int = Field(ge=0, le=2**63 - 1)
    process_index: int = Field(ge=0)
    worker_index: int = Field(ge=0)
    game_index: int = Field(ge=0)
    ply: int = Field(ge=0)
    trace_sample_seed: int = Field(ge=0, le=2**63 - 1)

    @model_validator(mode='after')
    def validate_derivation(self) -> SearchTraceSampleLineage:
        expected = derive_seed(
            self.root_seed,
            SearchTraceSampleSeedCoordinates(
                purpose=SeedPurpose.SEARCH_TRACE_SAMPLE,
                process_index=self.process_index,
                worker_index=self.worker_index,
                game_index=self.game_index,
                ply=self.ply,
            ),
        )
        if self.derivation_version != SEED_DERIVATION_VERSION or self.trace_sample_seed != expected:
            raise ValueError('Search trace sample seed does not match its deterministic coordinates.')
        return self


class InitialTraceModelIdentity(FrozenModel):
    kind: Literal['initial_model']
    model_initialization_seed: int = Field(ge=0, le=2**63 - 1)
    model_configuration_sha256: Sha256


class CheckpointTraceModelIdentity(FrozenModel):
    kind: Literal['checkpoint']
    checkpoint_id: UUID
    model_artifact_sha256: Sha256


TraceModelIdentity = Annotated[
    InitialTraceModelIdentity | CheckpointTraceModelIdentity,
    Field(discriminator='kind'),
]


class CalibrationSourceTrace(FrozenModel):
    source_position_id: UUID
    trace_payload_sha256: Sha256
    trace_file_sha256: Sha256


class SearchTraceCollectionPayload(FrozenModel):
    schema_version: Literal[1] = 1
    artifact_id: UUID
    source_run_id: UUID
    source_configuration_sha256: Sha256
    source_model: TraceModelIdentity
    game_id: UUID
    replay_sample_id: UUID
    lifecycle: Literal['completed_game_awaiting_replay_commit']
    game_configuration_sha256: Sha256
    native_state_hash: int = Field(ge=0)
    encoding_planes: int = Field(gt=0)
    encoding_board_size: int = Field(ge=3)
    canonical_encoding: tuple[int, ...] = Field(min_length=1)
    legal_actions: tuple[int, ...] = Field(min_length=1)
    observation: SearchTraceObservation
    seed_lineage: SearchTraceSampleLineage

    @model_validator(mode='after')
    def validate_position_evidence(self) -> SearchTraceCollectionPayload:
        if self.observation.source_position_id != self.replay_sample_id:
            raise ValueError('Trace source position must equal its replay sample identity.')
        expected_values = self.encoding_planes * self.encoding_board_size**2
        if len(self.canonical_encoding) != expected_values:
            raise ValueError('Trace canonical encoding length does not match its shape.')
        if any(value not in (0, 1) for value in self.canonical_encoding):
            raise ValueError('Trace canonical encoding must be binary.')
        action_count = self.encoding_board_size**2 + 1
        if len(self.observation.full.root_policy) != action_count:
            raise ValueError('Trace action space must equal board size squared plus pass.')
        if tuple(sorted(set(self.legal_actions))) != tuple(sorted(self.legal_actions)):
            raise ValueError('Trace legal actions must be unique.')
        if any(action < 0 or action >= action_count for action in self.legal_actions):
            raise ValueError('Trace legal actions must be inside the full action space.')
        return self


class SearchTraceCollectionArtifact(FrozenModel):
    payload: SearchTraceCollectionPayload
    payload_sha256: Sha256

    @model_validator(mode='after')
    def validate_digest(self) -> SearchTraceCollectionArtifact:
        if self.payload_sha256 != trace_collection_payload_sha256(self.payload):
            raise ValueError('Search trace collection payload SHA-256 does not match.')
        return self


@dataclass(frozen=True)
class LoadedSearchTraceArtifact:
    path: Path
    file_sha256: Sha256
    artifact: SearchTraceCollectionArtifact


def load_trace_collection_artifact(path: Path) -> LoadedSearchTraceArtifact:
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Search trace artifact path must identify an absolute file.')
    contents = path.read_bytes()
    artifact = SearchTraceCollectionArtifact.model_validate_json(contents)
    return LoadedSearchTraceArtifact(
        path=path.resolve(),
        file_sha256=hashlib.sha256(contents).hexdigest(),
        artifact=artifact,
    )


class VisitMarginCandidate(FrozenModel):
    minimum_simulations: int = Field(gt=0)
    check_interval_simulations: int = Field(gt=0)
    required_top_visit_fraction: float = Field(gt=0.5, le=1)
    required_top_two_margin: float = Field(ge=0, le=1)


class MaximumMeanDisagreementRule(FrozenModel):
    kind: Literal['maximum_mean_disagreement']
    maximum_policy_total_variation: float = Field(ge=0, le=1)
    maximum_value_absolute_error: float = Field(ge=0, le=2)


class CalibratedCandidate(FrozenModel):
    candidate: VisitMarginCandidate
    mean_policy_total_variation: float = Field(ge=0, le=1)
    mean_value_absolute_error: float = Field(ge=0, le=2)
    early_stop_fraction: float = Field(ge=0, le=1)
    mean_simulations: float = Field(gt=0)
    policy_total_variation_median: float = Field(ge=0, le=1)
    policy_total_variation_p95: float = Field(ge=0, le=1)
    policy_total_variation_maximum: float = Field(ge=0, le=1)
    value_absolute_error_median: float = Field(ge=0, le=2)
    value_absolute_error_p95: float = Field(ge=0, le=2)
    value_absolute_error_maximum: float = Field(ge=0, le=2)
    stop_simulations_median: float = Field(gt=0)
    stop_simulations_p95: float = Field(gt=0)
    stop_simulations_maximum: int = Field(gt=0)


class SimulationCountFrequency(FrozenModel):
    simulations: int = Field(gt=0)
    observation_count: int = Field(gt=0)


class SearchCalibrationProfile(FrozenModel):
    full_simulation_cap: int = Field(gt=1)
    trace_checkpoints: tuple[int, ...] = Field(min_length=1)
    selected: CalibratedCandidate
    candidate_results: tuple[CalibratedCandidate, ...] = Field(min_length=1)
    selected_simulation_distribution: tuple[SimulationCountFrequency, ...] = Field(min_length=1)
    observation_count: int = Field(gt=0)
    sources: tuple[CalibrationSourceTrace, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_profile(self) -> SearchCalibrationProfile:
        if tuple(sorted(set(self.trace_checkpoints))) != self.trace_checkpoints:
            raise ValueError('Calibration trace checkpoints must increase strictly.')
        if self.trace_checkpoints[-1] >= self.full_simulation_cap:
            raise ValueError('Calibration checkpoints must precede the full cap.')
        if self.selected not in self.candidate_results:
            raise ValueError('Selected calibration result must be declared.')
        if sum(item.observation_count for item in self.selected_simulation_distribution) != self.observation_count:
            raise ValueError('Calibration simulation distribution must account for every observation.')
        if len(self.sources) != self.observation_count:
            raise ValueError('Calibration profile must identify every source observation.')
        source_positions = tuple(source.source_position_id for source in self.sources)
        if tuple(sorted(set(source_positions), key=lambda identity: identity.hex)) != source_positions:
            raise ValueError('Calibration profile sources must be unique and ordered by position identity.')
        return self


class SearchCalibrationPayload(FrozenModel):
    schema_version: Literal[1] = 1
    artifact_id: UUID
    source_run_id: UUID
    source_configuration_sha256: Sha256
    source_model: TraceModelIdentity
    game_configuration_sha256: Sha256
    acceptance_rule: MaximumMeanDisagreementRule
    candidates: tuple[VisitMarginCandidate, ...] = Field(min_length=1)
    profiles: tuple[SearchCalibrationProfile, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_payload(self) -> SearchCalibrationPayload:
        caps = tuple(profile.full_simulation_cap for profile in self.profiles)
        if tuple(sorted(set(caps))) != caps:
            raise ValueError('Calibration profiles must have unique increasing full caps.')
        if any(profile.selected.candidate not in self.candidates for profile in self.profiles):
            raise ValueError('Selected calibration candidates must be declared.')
        all_source_positions = tuple(
            source.source_position_id for profile in self.profiles for source in profile.sources
        )
        if len(set(all_source_positions)) != len(all_source_positions):
            raise ValueError('Calibration source positions cannot occur in multiple cap profiles.')
        for profile in self.profiles:
            checkpoints = set(profile.trace_checkpoints)
            for candidate in self.candidates:
                if candidate.minimum_simulations >= profile.full_simulation_cap:
                    raise ValueError('Every calibration candidate minimum must be below every profile cap.')
                required = set(
                    range(
                        candidate.minimum_simulations,
                        profile.full_simulation_cap,
                        candidate.check_interval_simulations,
                    )
                )
                if not required.issubset(checkpoints):
                    raise ValueError('Trace checkpoints must cover every candidate decision point.')
        return self


class SearchCalibrationArtifact(FrozenModel):
    payload: SearchCalibrationPayload
    payload_sha256: Sha256

    @model_validator(mode='after')
    def validate_digest(self) -> SearchCalibrationArtifact:
        if self.payload_sha256 != calibration_payload_sha256(self.payload):
            raise ValueError('Calibration payload SHA-256 does not match its payload.')
        return self


def load_calibration_artifact(reference: CalibrationArtifactReference, root: Path) -> SearchCalibrationArtifact:
    path = (root / Path(reference.path)).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Calibration artifact path does not identify a file below the artifact root.')
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != reference.sha256:
        raise ValueError('Calibration artifact file SHA-256 does not match its reference.')
    artifact = SearchCalibrationArtifact.model_validate_json(contents)
    if artifact.payload.artifact_id != reference.artifact_id:
        raise ValueError('Calibration artifact identity does not match its reference.')
    return artifact


def calibration_payload_sha256(payload: SearchCalibrationPayload) -> str:
    canonical = json.dumps(
        payload.model_dump(mode='json'),
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def trace_collection_payload_sha256(payload: SearchTraceCollectionPayload) -> str:
    canonical = json.dumps(
        payload.model_dump(mode='json'),
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def publish_trace_collection_artifact(
    directory: Path,
    payload: SearchTraceCollectionPayload,
) -> Path:
    if not directory.is_absolute():
        raise ValueError('Search trace artifact directory must be absolute.')
    directory.mkdir(parents=True, exist_ok=True)
    artifact = SearchTraceCollectionArtifact(
        payload=payload,
        payload_sha256=trace_collection_payload_sha256(payload),
    )
    path = directory / f'trace-{payload.artifact_id.hex}.json'
    contents = artifact.model_dump_json(indent=2).encode() + b'\n'
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError('Search trace artifact identity already has different contents.')
        return path
    partial = path.with_suffix('.partial')
    if partial.exists():
        partial.unlink()
    with partial.open('xb') as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return path


def publish_calibration_artifact(
    directory: Path,
    artifact: SearchCalibrationArtifact,
) -> Path:
    if not directory.is_absolute():
        raise ValueError('Calibration artifact directory must be absolute.')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'calibration-{artifact.payload.artifact_id.hex}.json'
    contents = artifact.model_dump_json(indent=2).encode() + b'\n'
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError('Calibration artifact identity already has different contents.')
        return path
    partial = path.with_suffix('.partial')
    if partial.exists():
        partial.unlink()
    with partial.open('xb') as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return path

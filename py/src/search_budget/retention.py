from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator
from src.search_budget.artifacts import write_persisted_model
from src.search_budget.configuration import LabelArtifactRetention
from src.util.frozen_model import FrozenModel


class LabelArtifactKind(str, Enum):
    GENERATION_SOURCE = 'generation_source'
    PREDICTION_SHARD = 'prediction_shard'
    DEEP_SEARCH_SHARD = 'deep_search_shard'


class LabelArtifactCleanupTarget(FrozenModel):
    relative_path: Path
    kind: LabelArtifactKind
    size_bytes: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_relative_path(self) -> LabelArtifactCleanupTarget:
        if self.relative_path.is_absolute() or '..' in self.relative_path.parts:
            raise ValueError('Cleanup targets must be safe paths relative to one generation directory.')
        return self


class LabelArtifactCleanupEvidence(FrozenModel):
    final_report_path: Path
    manager_state_path: Path
    calibration_state_path: Path
    replay_writeback_state_path: Path

    @model_validator(mode='after')
    def validate_relative_paths(self) -> LabelArtifactCleanupEvidence:
        paths = (
            self.final_report_path,
            self.manager_state_path,
            self.calibration_state_path,
            self.replay_writeback_state_path,
        )
        if any(path.is_absolute() or '..' in path.parts for path in paths):
            raise ValueError('Cleanup evidence must use safe paths relative to the run directory.')
        return self


class PreparedLabelArtifactCleanupReceipt(FrozenModel):
    kind: Literal['prepared'] = 'prepared'
    schema_version: int = Field(default=1, ge=1, le=1)
    source_generation: int = Field(ge=0)
    retention: Literal[LabelArtifactRetention.REMOVE_BULKY_AFTER_FINALIZATION]
    targets: tuple[LabelArtifactCleanupTarget, ...]
    preserved_manifest_count: int = Field(ge=0)
    evidence: LabelArtifactCleanupEvidence


class CompletedLabelArtifactCleanupReceipt(FrozenModel):
    kind: Literal['completed'] = 'completed'
    schema_version: int = Field(default=1, ge=1, le=1)
    source_generation: int = Field(ge=0)
    retention: Literal[LabelArtifactRetention.REMOVE_BULKY_AFTER_FINALIZATION]
    targets: tuple[LabelArtifactCleanupTarget, ...]
    preserved_manifest_count: int = Field(ge=0)
    evidence: LabelArtifactCleanupEvidence

    @property
    def removed_file_count(self) -> int:
        return len(self.targets)

    @property
    def removed_size_bytes(self) -> int:
        return sum(target.size_bytes for target in self.targets)


class FailedLabelArtifactCleanupReceipt(FrozenModel):
    kind: Literal['failed'] = 'failed'
    schema_version: int = Field(default=1, ge=1, le=1)
    source_generation: int = Field(ge=0)
    retention: Literal[LabelArtifactRetention.REMOVE_BULKY_AFTER_FINALIZATION]
    targets: tuple[LabelArtifactCleanupTarget, ...]
    preserved_manifest_count: int = Field(ge=0)
    evidence: LabelArtifactCleanupEvidence
    failure: str = Field(min_length=1)


LabelArtifactCleanupReceipt = Annotated[
    PreparedLabelArtifactCleanupReceipt | CompletedLabelArtifactCleanupReceipt | FailedLabelArtifactCleanupReceipt,
    Field(discriminator='kind'),
]

_RECEIPT_ADAPTER = TypeAdapter(LabelArtifactCleanupReceipt)
RECEIPT_FILE_NAME = 'artifact-cleanup-receipt.json'


def cleanup_completed_generation_artifacts(
    job_path: Path,
    source_generation: int,
    evidence: LabelArtifactCleanupEvidence,
) -> CompletedLabelArtifactCleanupReceipt | FailedLabelArtifactCleanupReceipt:
    receipt_path = job_path / RECEIPT_FILE_NAME
    existing = load_label_artifact_cleanup_receipt(receipt_path) if receipt_path.exists() else None
    if isinstance(existing, CompletedLabelArtifactCleanupReceipt):
        _validate_existing_receipt(existing, source_generation, evidence)
        return existing
    if existing is not None:
        _validate_existing_receipt(existing, source_generation, evidence)
    if existing is None:
        targets = _discover_cleanup_targets(job_path)
        preserved_manifest_count = _manifest_count(job_path)
    else:
        targets = existing.targets
        preserved_manifest_count = existing.preserved_manifest_count
    prepared = PreparedLabelArtifactCleanupReceipt(
        source_generation=source_generation,
        retention=LabelArtifactRetention.REMOVE_BULKY_AFTER_FINALIZATION,
        targets=targets,
        preserved_manifest_count=preserved_manifest_count,
        evidence=evidence,
    )
    try:
        write_persisted_model(receipt_path, prepared)
    except (OSError, ValueError) as error:
        return _failed_receipt(prepared, error)
    try:
        for target in prepared.targets:
            target_path = _resolve_target(job_path, target.relative_path)
            target_path.unlink(missing_ok=True)
    except (OSError, ValueError) as error:
        failed = _failed_receipt(prepared, error)
        _write_receipt_without_raising(receipt_path, failed)
        return failed
    completed = CompletedLabelArtifactCleanupReceipt(
        source_generation=prepared.source_generation,
        retention=prepared.retention,
        targets=prepared.targets,
        preserved_manifest_count=prepared.preserved_manifest_count,
        evidence=prepared.evidence,
    )
    try:
        write_persisted_model(receipt_path, completed)
    except (OSError, ValueError) as error:
        return _failed_receipt(prepared, error)
    return completed


def load_label_artifact_cleanup_receipt(path: Path) -> LabelArtifactCleanupReceipt:
    return _RECEIPT_ADAPTER.validate_json(path.read_text(encoding='utf-8'))


def _validate_existing_receipt(
    receipt: LabelArtifactCleanupReceipt,
    source_generation: int,
    evidence: LabelArtifactCleanupEvidence,
) -> None:
    if receipt.source_generation != source_generation or receipt.evidence != evidence:
        raise ValueError('Cleanup receipt does not match the completed generation evidence.')


def _discover_cleanup_targets(job_path: Path) -> tuple[LabelArtifactCleanupTarget, ...]:
    targets: list[LabelArtifactCleanupTarget] = []
    source_path = job_path / 'source.json'
    if source_path.is_file():
        targets.append(_target(job_path, source_path, LabelArtifactKind.GENERATION_SOURCE))
    for phase_name, kind in (
        ('prediction', LabelArtifactKind.PREDICTION_SHARD),
        ('deep-search', LabelArtifactKind.DEEP_SEARCH_SHARD),
    ):
        phase_path = job_path / phase_name
        if not phase_path.exists():
            continue
        targets.extend(
            _target(job_path, artifact_path, kind)
            for artifact_path in sorted(phase_path.rglob('artifact-attempt-*.json'))
            if artifact_path.is_file()
        )
    return tuple(targets)


def _target(job_path: Path, path: Path, kind: LabelArtifactKind) -> LabelArtifactCleanupTarget:
    resolved = path.resolve()
    relative_path = resolved.relative_to(job_path.resolve())
    return LabelArtifactCleanupTarget(relative_path=relative_path, kind=kind, size_bytes=resolved.stat().st_size)


def _resolve_target(job_path: Path, relative_path: Path) -> Path:
    resolved_job_path = job_path.resolve()
    resolved_target = (job_path / relative_path).resolve()
    if not resolved_target.is_relative_to(resolved_job_path) or resolved_target == resolved_job_path:
        raise ValueError('Cleanup target escaped its generation directory.')
    return resolved_target


def _manifest_count(job_path: Path) -> int:
    return sum(
        1
        for phase_name in ('prediction', 'deep-search')
        for path in (job_path / phase_name).rglob('attempt-*.json')
        if path.is_file()
    )


def _failed_receipt(
    prepared: PreparedLabelArtifactCleanupReceipt,
    error: OSError | ValueError,
) -> FailedLabelArtifactCleanupReceipt:
    return FailedLabelArtifactCleanupReceipt(
        source_generation=prepared.source_generation,
        retention=prepared.retention,
        targets=prepared.targets,
        preserved_manifest_count=prepared.preserved_manifest_count,
        evidence=prepared.evidence,
        failure=f'{type(error).__name__}: {error}',
    )


def _write_receipt_without_raising(path: Path, receipt: FailedLabelArtifactCleanupReceipt) -> None:
    try:
        write_persisted_model(path, receipt)
    except (OSError, ValueError):
        return

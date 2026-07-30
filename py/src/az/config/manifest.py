from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveInt, model_validator

from src.az.config.base import DeterminismMode, FrozenModel, GitRevision, Sha256
from src.az.config.dependency_lock import (
    DependencyRecord,
    normalized_package_name,
    parse_pinned_dependency_lock,
)
from src.az.config.root import ResolvedRunConfiguration
from src.az.config.seeds import SEED_DERIVATION_VERSION, SeedDerivationVersion
from src.az.config.serialization import model_sha256


class SourceState(FrozenModel):
    revision: GitRevision
    clean: bool
    dirty_patch_sha256: Sha256 | None

    @model_validator(mode='after')
    def validate_clean_state(self) -> SourceState:
        if self.clean and self.dirty_patch_sha256 is not None:
            raise ValueError('A clean source state cannot have a dirty patch hash.')
        if not self.clean and self.dirty_patch_sha256 is None:
            raise ValueError('A dirty source state requires a dirty patch hash.')
        return self


class BuildDeclaration(FrozenModel):
    build_id: str = Field(min_length=1)
    build_type: str = Field(min_length=1)
    compiler: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    platform: str = Field(min_length=1)


class DependencyDeclaration(FrozenModel):
    lock_file: Path
    lock_file_sha256: Sha256
    packages: tuple[DependencyRecord, ...]

    @model_validator(mode='after')
    def validate_package_names(self) -> DependencyDeclaration:
        names = tuple(normalized_package_name(package.name) for package in self.packages)
        if len(set(names)) != len(names):
            raise ValueError('Dependency package names must be unique.')
        return self


class HardwareDeclaration(FrozenModel):
    gpu_model: str = Field(min_length=1)
    gpu_count: int = Field(ge=0)
    logical_cpu_count: PositiveInt
    ram_gib: float = Field(gt=0)
    free_disk_gib: float = Field(gt=0)

    @model_validator(mode='after')
    def validate_gpu_identity(self) -> HardwareDeclaration:
        if (self.gpu_count == 0) != (self.gpu_model == 'none'):
            raise ValueError("Zero GPUs require model 'none', and model 'none' requires zero GPUs.")
        return self


class RunManifest(FrozenModel):
    manifest_version: Literal[1] = 1
    created_at_utc: datetime
    configuration: ResolvedRunConfiguration
    configuration_sha256: Sha256
    source: SourceState
    build: BuildDeclaration
    dependencies: DependencyDeclaration
    hardware: HardwareDeclaration
    determinism_mode: DeterminismMode
    seed_derivation_version: SeedDerivationVersion

    @model_validator(mode='after')
    def validate_integrity(self) -> RunManifest:
        if self.created_at_utc.utcoffset() != timedelta(0):
            raise ValueError('Manifest creation time must use UTC.')
        if self.configuration_sha256 != model_sha256(self.configuration):
            raise ValueError('Manifest configuration SHA-256 does not match its configuration.')
        if self.determinism_mode is not self.configuration.experiment.manifest_policy.determinism_mode:
            raise ValueError('Manifest determinism mode does not match the configuration policy.')
        if self.seed_derivation_version != SEED_DERIVATION_VERSION:
            raise ValueError('Manifest seed derivation version is not supported.')
        record_versions = self.configuration.experiment.manifest_policy.record_dependency_versions
        if record_versions and not self.dependencies.packages:
            raise ValueError('Dependency package records are required by the manifest policy.')
        if not record_versions and self.dependencies.packages:
            raise ValueError('Dependency package records must be empty when version recording is disabled.')
        _validate_runtime_declarations(self.configuration, self.source, self.hardware)
        return self


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_source_state(repository_root: Path) -> SourceState:
    revision = _git(repository_root, 'rev-parse', 'HEAD').strip()
    status = _git(repository_root, 'status', '--porcelain=v1', '--untracked-files=all')
    if not status:
        return SourceState(revision=revision, clean=True, dirty_patch_sha256=None)

    digest = hashlib.sha256()
    digest.update(_git_bytes(repository_root, 'diff', '--binary', 'HEAD', '--'))
    untracked = _git(repository_root, 'ls-files', '--others', '--exclude-standard').splitlines()
    for relative_path in sorted(untracked):
        digest.update(relative_path.encode('utf-8'))
        digest.update(b'\0')
        digest.update((repository_root / relative_path).read_bytes())
        digest.update(b'\0')
    return SourceState(
        revision=revision,
        clean=False,
        dirty_patch_sha256=digest.hexdigest(),
    )


def inspect_source_revision(repository_root: Path) -> GitRevision:
    return _git(repository_root, 'rev-parse', 'HEAD').strip()


def default_build_declaration(build_id: str, build_type: str, compiler: str) -> BuildDeclaration:
    return BuildDeclaration(
        build_id=build_id,
        build_type=build_type,
        compiler=compiler,
        python_version=platform.python_version(),
        platform=platform.platform(),
    )


def build_manifest(
    configuration: ResolvedRunConfiguration,
    repository_root: Path,
    build: BuildDeclaration,
    dependencies: DependencyDeclaration,
    hardware: HardwareDeclaration,
    created_at_utc: datetime | None = None,
) -> RunManifest:
    timestamp = created_at_utc or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError('Manifest creation time must include a timezone.')
    source = inspect_source_state(repository_root)
    _validate_dependencies(configuration, dependencies)
    return RunManifest(
        created_at_utc=timestamp.astimezone(timezone.utc),
        configuration=configuration,
        configuration_sha256=model_sha256(configuration),
        source=source,
        build=build,
        dependencies=dependencies,
        hardware=hardware,
        determinism_mode=configuration.experiment.manifest_policy.determinism_mode,
        seed_derivation_version=SEED_DERIVATION_VERSION,
    )


def _validate_runtime_declarations(
    configuration: ResolvedRunConfiguration,
    source: SourceState,
    hardware: HardwareDeclaration,
) -> None:
    if configuration.experiment.manifest_policy.require_clean_source and not source.clean:
        raise ValueError('The manifest policy requires a clean source worktree.')
    expected = configuration.hardware
    if expected.profile_name != 'local-cpu-smoke':
        if hardware.gpu_model != expected.expected_gpu_model:
            raise ValueError(f'Expected GPU model {expected.expected_gpu_model!r}, found {hardware.gpu_model!r}.')
        if hardware.gpu_count != expected.expected_gpu_count:
            raise ValueError(f'Expected {expected.expected_gpu_count} GPUs, found {hardware.gpu_count}.')
    if hardware.logical_cpu_count < expected.minimum_logical_cpu_count:
        raise ValueError('Actual logical CPU count is below the configured minimum.')
    if hardware.ram_gib < expected.minimum_ram_gib:
        raise ValueError('Actual RAM is below the configured minimum.')
    if hardware.free_disk_gib < expected.minimum_free_disk_gib:
        raise ValueError('Actual free disk is below the configured minimum.')


def _validate_dependencies(
    configuration: ResolvedRunConfiguration,
    declaration: DependencyDeclaration,
) -> None:
    if not declaration.lock_file.is_file():
        raise ValueError(f'Dependency lock file does not exist: {declaration.lock_file}')
    if file_sha256(declaration.lock_file) != declaration.lock_file_sha256:
        raise ValueError('Dependency lock file SHA-256 does not match its declaration.')
    record_versions = configuration.experiment.manifest_policy.record_dependency_versions
    if record_versions and not declaration.packages:
        raise ValueError('Dependency package records are required by the manifest policy.')
    if not record_versions and declaration.packages:
        raise ValueError('Dependency package records must be empty when version recording is disabled.')
    if record_versions and declaration.packages != parse_pinned_dependency_lock(declaration.lock_file):
        raise ValueError('Dependency package records do not exactly match the pinned lock file.')


def _git(repository_root: Path, *arguments: str) -> str:
    return _git_bytes(repository_root, *arguments).decode('utf-8').strip()


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ('git', *arguments),
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def current_python_build(build_id: str = 'development') -> BuildDeclaration:
    return default_build_declaration(
        build_id=build_id,
        build_type='python',
        compiler=platform.python_compiler() or sys.implementation.name,
    )

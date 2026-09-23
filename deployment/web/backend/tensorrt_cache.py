from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256

_CACHE_SCHEMA_VERSION = 1


class TensorRtRuntimeIdentity(FrozenModel):
    tensorrt_version: str = Field(min_length=1)
    cuda_runtime_version: str = Field(min_length=1)
    nvidia_driver_version: str = Field(min_length=1)
    gpu_name: str = Field(min_length=1)
    gpu_compute_capability: str = Field(pattern=r'^\d+\.\d+$')


class TensorRtEngineCacheIdentity(FrozenModel):
    schema_version: int = _CACHE_SCHEMA_VERSION
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    batch_size: int = Field(gt=0)
    channels: int = Field(gt=0)
    rows: int = Field(gt=0)
    columns: int = Field(gt=0)
    builder_optimization_level: int = Field(ge=0, le=5)
    runtime: TensorRtRuntimeIdentity


class TensorRtEngineCacheMetadata(FrozenModel):
    identity: TensorRtEngineCacheIdentity
    engine_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


@dataclass(frozen=True)
class CachedTensorRtEngine:
    path: Path
    built: bool


TensorRtEngineBuilder = Callable[[Path, Path, int, int, int, int, int], None]


def prepare_cached_tensorrt_engine(
    source_path: Path,
    cache_root: Path,
    runtime: TensorRtRuntimeIdentity,
    build_engine: TensorRtEngineBuilder,
    batch_size: int,
    channels: int,
    rows: int,
    columns: int,
    builder_optimization_level: int,
) -> CachedTensorRtEngine:
    identity = create_tensorrt_cache_identity(
        source_sha256=file_sha256(source_path),
        runtime=runtime,
        batch_size=batch_size,
        channels=channels,
        rows=rows,
        columns=columns,
        builder_optimization_level=builder_optimization_level,
    )
    cached_engine = find_cached_tensorrt_engine(cache_root, identity)
    if cached_engine is not None:
        return cached_engine
    return build_cached_tensorrt_engine(source_path, cache_root, identity, build_engine)


def create_tensorrt_cache_identity(
    source_sha256: str,
    runtime: TensorRtRuntimeIdentity,
    batch_size: int,
    channels: int,
    rows: int,
    columns: int,
    builder_optimization_level: int,
) -> TensorRtEngineCacheIdentity:
    return TensorRtEngineCacheIdentity(
        source_sha256=source_sha256,
        batch_size=batch_size,
        channels=channels,
        rows=rows,
        columns=columns,
        builder_optimization_level=builder_optimization_level,
        runtime=runtime,
    )


def find_cached_tensorrt_engine(
    cache_root: Path,
    identity: TensorRtEngineCacheIdentity,
) -> CachedTensorRtEngine | None:
    cache_key = hashlib.sha256(identity.model_dump_json().encode('utf-8')).hexdigest()
    cache_directory = cache_root / cache_key
    engine_path = cache_directory / 'model.engine'
    metadata_path = cache_directory / 'metadata.json'

    if _valid_cached_engine(engine_path, metadata_path, identity):
        return CachedTensorRtEngine(path=engine_path, built=False)
    return None


def build_cached_tensorrt_engine(
    source_path: Path,
    cache_root: Path,
    identity: TensorRtEngineCacheIdentity,
    build_engine: TensorRtEngineBuilder,
) -> CachedTensorRtEngine:
    cache_key = hashlib.sha256(identity.model_dump_json().encode('utf-8')).hexdigest()
    cache_directory = cache_root / cache_key
    engine_path = cache_directory / 'model.engine'
    metadata_path = cache_directory / 'metadata.json'

    cache_directory.mkdir(parents=True, exist_ok=True)
    build_engine(
        source_path,
        engine_path,
        identity.batch_size,
        identity.channels,
        identity.rows,
        identity.columns,
        identity.builder_optimization_level,
    )
    metadata = TensorRtEngineCacheMetadata(identity=identity, engine_sha256=file_sha256(engine_path))
    write_text_atomically(metadata_path, metadata.model_dump_json(indent=2) + '\n')
    return CachedTensorRtEngine(path=engine_path, built=True)


def _valid_cached_engine(
    engine_path: Path,
    metadata_path: Path,
    identity: TensorRtEngineCacheIdentity,
) -> bool:
    if not engine_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = TensorRtEngineCacheMetadata.model_validate_json(metadata_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return metadata.identity == identity and metadata.engine_sha256 == file_sha256(engine_path)

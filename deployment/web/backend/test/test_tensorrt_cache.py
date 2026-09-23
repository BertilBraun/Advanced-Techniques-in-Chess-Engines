from __future__ import annotations

from pathlib import Path

from src.util.hashing import file_sha256

from deployment.web.backend.tensorrt_cache import (
    TensorRtRuntimeIdentity,
    create_tensorrt_cache_identity,
    find_cached_tensorrt_engine,
    prepare_cached_tensorrt_engine,
)


def _runtime() -> TensorRtRuntimeIdentity:
    return TensorRtRuntimeIdentity(
        tensorrt_version='10.14.1',
        cuda_runtime_version='12.6',
        nvidia_driver_version='580.65.06',
        gpu_name='NVIDIA A10',
        gpu_compute_capability='8.6',
    )


def test_reuses_valid_cached_engine(tmp_path: Path) -> None:
    source_path = tmp_path / 'model.onnx'
    source_path.write_bytes(b'model-one')
    builds: list[Path] = []

    def build_engine(
        source: Path,
        output: Path,
        batch_size: int,
        channels: int,
        rows: int,
        columns: int,
        optimization_level: int,
    ) -> None:
        assert source == source_path
        assert (batch_size, channels, rows, columns, optimization_level) == (64, 52, 8, 8, 3)
        builds.append(output)
        output.write_bytes(b'engine-one')

    first = prepare_cached_tensorrt_engine(
        source_path,
        tmp_path / 'cache',
        _runtime(),
        build_engine,
        64,
        52,
        8,
        8,
        3,
    )
    second = prepare_cached_tensorrt_engine(
        source_path,
        tmp_path / 'cache',
        _runtime(),
        build_engine,
        64,
        52,
        8,
        8,
        3,
    )

    assert first.built is True
    assert second.built is False
    assert second.path == first.path
    assert builds == [first.path]


def test_finds_cached_engine_from_pinned_source_digest(tmp_path: Path) -> None:
    source_path = tmp_path / 'model.onnx'
    source_path.write_bytes(b'model-one')

    def build_engine(
        source: Path,
        output: Path,
        batch_size: int,
        channels: int,
        rows: int,
        columns: int,
        optimization_level: int,
    ) -> None:
        del source, batch_size, channels, rows, columns, optimization_level
        output.write_bytes(b'engine-one')

    built = prepare_cached_tensorrt_engine(
        source_path,
        tmp_path / 'cache',
        _runtime(),
        build_engine,
        64,
        52,
        8,
        8,
        3,
    )
    identity = create_tensorrt_cache_identity(
        source_sha256=file_sha256(source_path),
        runtime=_runtime(),
        batch_size=64,
        channels=52,
        rows=8,
        columns=8,
        builder_optimization_level=3,
    )
    source_path.unlink()

    cached = find_cached_tensorrt_engine(tmp_path / 'cache', identity)

    assert cached is not None
    assert cached.built is False
    assert cached.path == built.path


def test_rebuilds_tampered_cached_engine(tmp_path: Path) -> None:
    source_path = tmp_path / 'model.onnx'
    source_path.write_bytes(b'model-one')
    build_count = 0

    def build_engine(
        source: Path,
        output: Path,
        batch_size: int,
        channels: int,
        rows: int,
        columns: int,
        optimization_level: int,
    ) -> None:
        nonlocal build_count
        del source, batch_size, channels, rows, columns, optimization_level
        build_count += 1
        output.write_bytes(f'engine-{build_count}'.encode())

    first = prepare_cached_tensorrt_engine(
        source_path,
        tmp_path / 'cache',
        _runtime(),
        build_engine,
        64,
        52,
        8,
        8,
        3,
    )
    first.path.write_bytes(b'tampered')
    rebuilt = prepare_cached_tensorrt_engine(
        source_path,
        tmp_path / 'cache',
        _runtime(),
        build_engine,
        64,
        52,
        8,
        8,
        3,
    )

    assert rebuilt.built is True
    assert build_count == 2


def test_runtime_or_source_change_uses_a_distinct_cache_entry(tmp_path: Path) -> None:
    source_path = tmp_path / 'model.onnx'
    source_path.write_bytes(b'model-one')

    def build_engine(
        source: Path,
        output: Path,
        batch_size: int,
        channels: int,
        rows: int,
        columns: int,
        optimization_level: int,
    ) -> None:
        del source, batch_size, channels, rows, columns, optimization_level
        output.write_bytes(b'engine')

    first = prepare_cached_tensorrt_engine(
        source_path,
        tmp_path / 'cache',
        _runtime(),
        build_engine,
        64,
        52,
        8,
        8,
        3,
    )
    source_path.write_bytes(b'model-two')
    second = prepare_cached_tensorrt_engine(
        source_path,
        tmp_path / 'cache',
        _runtime().model_copy(update={'tensorrt_version': '10.15.1'}),
        build_engine,
        64,
        52,
        8,
        8,
        3,
    )

    assert first.path != second.path

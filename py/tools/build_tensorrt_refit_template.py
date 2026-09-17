from __future__ import annotations

import argparse
import shutil
from enum import StrEnum
from pathlib import Path

import onnx
import tensorrt as trt
from src.util.atomic_file import write_bytes_atomically
from tools.publish_tensorrt_engine import export_onnx

WORKSPACE_BYTES = 4 * 1024**3


class RefitMode(StrEnum):
    ALL = 'all'
    INDIVIDUAL = 'individual'


def _mark_onnx_weights_refittable(network: trt.INetworkDefinition, onnx_path: Path) -> int:
    model = onnx.load(onnx_path, load_external_data=False)
    marked = sum(
        network.mark_weights_refittable(initializer.name)
        for initializer in sorted(model.graph.initializer, key=lambda initializer: initializer.name)
    )
    if marked == 0:
        raise ValueError('TensorRT did not recognize any ONNX weights as individually refittable.')
    return marked


def build_template(
    model_path: Path,
    output_path: Path,
    batch_size: int,
    channels: int,
    rows: int,
    columns: int,
    optimization_level: int,
    timing_cache_path: Path | None,
    refit_mode: RefitMode,
) -> None:
    onnx_path = output_path.with_suffix('.temporary.onnx')
    onnx_path.unlink(missing_ok=True)
    try:
        if model_path.suffix == '.onnx':
            shutil.copyfile(model_path, onnx_path)
        else:
            export_onnx(model_path, onnx_path, (batch_size, channels, rows, columns))
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            errors = tuple(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise ValueError(f'TensorRT ONNX parsing failed: {errors}')
        configuration = builder.create_builder_config()
        configuration.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
        configuration.builder_optimization_level = optimization_level
        configuration.set_flag(trt.BuilderFlag.FP16)
        match refit_mode:
            case RefitMode.ALL:
                configuration.set_flag(trt.BuilderFlag.REFIT)
            case RefitMode.INDIVIDUAL:
                _mark_onnx_weights_refittable(network, onnx_path)
                configuration.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)
        timing_cache = (
            b'' if timing_cache_path is None or not timing_cache_path.exists() else timing_cache_path.read_bytes()
        )
        cache = configuration.create_timing_cache(timing_cache)
        if not configuration.set_timing_cache(cache, False):
            raise ValueError('TensorRT rejected the serialized timing cache.')
        serialized = builder.build_serialized_network(network, configuration)
        if serialized is None:
            raise ValueError('TensorRT template build failed')
        write_bytes_atomically(output_path, bytes(serialized))
        if timing_cache_path is not None:
            write_bytes_atomically(timing_cache_path, bytes(configuration.get_timing_cache().serialize()))
    finally:
        onnx_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a refittable FP16 TensorRT template.')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--channels', type=int, default=52)
    parser.add_argument('--rows', type=int, default=8)
    parser.add_argument('--columns', type=int, default=8)
    parser.add_argument('--optimization-level', type=int, default=5, choices=range(0, 6))
    parser.add_argument('--timing-cache', type=Path)
    parser.add_argument('--refit-mode', type=RefitMode, choices=tuple(RefitMode), default=RefitMode.ALL)
    arguments = parser.parse_args()
    build_template(
        arguments.model,
        arguments.output,
        arguments.batch_size,
        arguments.channels,
        arguments.rows,
        arguments.columns,
        arguments.optimization_level,
        arguments.timing_cache,
        arguments.refit_mode,
    )


if __name__ == '__main__':
    main()

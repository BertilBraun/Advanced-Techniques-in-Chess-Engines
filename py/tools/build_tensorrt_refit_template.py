from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import tensorrt as trt
from src.util.atomic_file import write_bytes_atomically
from tools.publish_tensorrt_engine import export_onnx

WORKSPACE_BYTES = 4 * 1024**3


def build_template(
    model_path: Path,
    output_path: Path,
    batch_size: int,
    channels: int,
    rows: int,
    columns: int,
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
        configuration.builder_optimization_level = 5
        configuration.set_flag(trt.BuilderFlag.FP16)
        configuration.set_flag(trt.BuilderFlag.REFIT)
        serialized = builder.build_serialized_network(network, configuration)
        if serialized is None:
            raise ValueError('TensorRT template build failed')
        write_bytes_atomically(output_path, bytes(serialized))
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
    arguments = parser.parse_args()
    build_template(
        arguments.model,
        arguments.output,
        arguments.batch_size,
        arguments.channels,
        arguments.rows,
        arguments.columns,
    )


if __name__ == '__main__':
    main()

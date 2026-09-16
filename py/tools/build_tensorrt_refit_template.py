from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx
import tensorrt as trt
from src.self_play.tensorrt_refit import (
    TensorRtRefitTemplateMetadata,
    canonicalize_onnx_refit_names,
    onnx_refit_contract,
    template_metadata_path,
)
from src.util.atomic_file import write_bytes_atomically, write_text_atomically
from src.util.hashing import file_sha256
from tools.publish_tensorrt_engine import export_onnx

WORKSPACE_BYTES = 4 * 1024**3


def _mark_onnx_weights_refittable(network: trt.INetworkDefinition, onnx_path: Path) -> int:
    model = onnx.load(onnx_path, load_external_data=False)
    candidate_names = {initializer.name for initializer in model.graph.initializer}
    candidate_names.update(
        output_name for node in model.graph.node if node.op_type == 'Constant' for output_name in node.output
    )
    marked = sum(network.mark_weights_refittable(name) for name in sorted(candidate_names))
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
) -> None:
    onnx_path = output_path.with_suffix('.temporary.onnx')
    onnx_path.unlink(missing_ok=True)
    try:
        if model_path.suffix == '.onnx':
            shutil.copyfile(model_path, onnx_path)
        else:
            export_onnx(model_path, onnx_path, (batch_size, channels, rows, columns))
        exported = onnx.load(onnx_path)
        canonicalize_onnx_refit_names(exported)
        onnx.save(exported, onnx_path)
        refit_contract = onnx_refit_contract(exported)
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            errors = tuple(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise ValueError(f'TensorRT ONNX parsing failed: {errors}')
        _mark_onnx_weights_refittable(network, onnx_path)
        configuration = builder.create_builder_config()
        configuration.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
        configuration.builder_optimization_level = optimization_level
        configuration.set_flag(trt.BuilderFlag.FP16)
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
        metadata = TensorRtRefitTemplateMetadata(
            engine_sha256=file_sha256(output_path),
            onnx_refit_contract=refit_contract,
        )
        write_text_atomically(template_metadata_path(output_path), metadata.model_dump_json(indent=2) + '\n')
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
    )


if __name__ == '__main__':
    main()

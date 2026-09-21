from __future__ import annotations

import argparse
import shutil
from enum import StrEnum
from pathlib import Path

import onnx
import tensorrt as trt
from onnx import numpy_helper
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


def _separate_equal_quantization_scales(onnx_path: Path) -> int:
    """Make per-tensor Q/DQ scales pairwise distinct, returning how many were changed.

    TensorRT rewrites Q/DQ pairs whose scales compare equal, and documents that it withholds those
    rewrites when building a refittable engine because a refit could separate the scales. At
    optimization level 5 in 10.14.1 it applies them anyway: a template built from a checkpoint whose
    activations sit at the ReLU6 cap, where twenty of twenty-eight scales are exactly 6/127, then
    refit with a checkpoint whose scales differ, returns wrong and run-to-run non-deterministic
    logits while every refit call reports success. That is what served V90 and cost about 400 Elo.
    A template's own scale values are overwritten by every publish, so spreading them is free.
    """
    model = onnx.load(onnx_path, load_external_data=False)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    constants = {
        node.output[0]: attribute.t
        for node in model.graph.node
        if node.op_type == 'Constant' and node.output
        for attribute in node.attribute
        if attribute.name == 'value'
    }
    scale_names = sorted(
        {
            node.input[1]
            for node in model.graph.node
            if node.op_type in ('QuantizeLinear', 'DequantizeLinear') and len(node.input) >= 2
        }
    )
    scalar_names = []
    for name in scale_names:
        tensor = initializers.get(name) or constants.get(name)
        if tensor is not None and numpy_helper.to_array(tensor).size == 1:
            scalar_names.append(name)
    values = [
        float(numpy_helper.to_array(initializers.get(name) or constants[name]).reshape(())) for name in scalar_names
    ]
    if len(set(values)) == len(values):
        return 0
    for index, name in enumerate(scalar_names):
        tensor = initializers.get(name) or constants[name]
        array = numpy_helper.to_array(tensor)
        factor = 0.7 + 0.3 * index / max(len(scalar_names) - 1, 1)
        tensor.CopyFrom(numpy_helper.from_array((array * factor).astype(array.dtype), tensor.name))
    onnx.save(model, onnx_path)
    return len(scalar_names)


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
        separated = _separate_equal_quantization_scales(onnx_path)
        if separated:
            print(f'separated {separated} equal quantization scales before building the refit template')
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
    # Levels 4 and 5 apply the Myelin scale-equality fusions that make a refit template unsafe;
    # level 3 emitted none in testing, matched every direct build on fidelity and built faster.
    parser.add_argument('--optimization-level', type=int, default=3, choices=range(0, 6))
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

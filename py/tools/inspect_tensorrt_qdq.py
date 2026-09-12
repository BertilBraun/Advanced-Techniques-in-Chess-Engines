from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import tensorrt as trt
from onnx import numpy_helper
from pydantic import Field
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256


class QuantizeLinearDescription(FrozenModel):
    name: str = Field(min_length=1)
    input: str = Field(min_length=1)
    axis: int
    scale_shape: tuple[int, ...]
    minimum_scale: float = Field(gt=0.0)
    maximum_scale: float = Field(gt=0.0)
    zero_point_dtype: str = Field(min_length=1)
    zero_point_minimum: int
    zero_point_maximum: int


class Count(FrozenModel):
    name: str = Field(min_length=1)
    count: int = Field(ge=0)


class TensorRtTensorDescription(FrozenModel):
    name: str = Field(alias='Name', min_length=1)
    location: str = Field(alias='Location', min_length=1)
    dimensions: tuple[int, ...] = Field(alias='Dimensions')
    format_and_data_type: str = Field(alias='Format/Datatype', min_length=1)


class TensorRtWeightsDescription(FrozenModel):
    data_type: str = Field(alias='Type', min_length=1)
    count: int = Field(alias='Count', ge=0)


class TensorRtLayerDescription(FrozenModel):
    name: str = Field(alias='Name', min_length=1)
    layer_type: str = Field(alias='LayerType', min_length=1)
    inputs: tuple[TensorRtTensorDescription, ...] = Field(alias='Inputs')
    outputs: tuple[TensorRtTensorDescription, ...] = Field(alias='Outputs')
    parameter_type: str = Field(alias='ParameterType', min_length=1)
    origin: str | None = Field(default=None, alias='Origin')
    tactic_name: str | None = Field(default=None, alias='TacticName')
    tactic_value: str | None = Field(default=None, alias='TacticValue')
    metadata: str | None = Field(default=None, alias='Metadata')
    stream_id: int | None = Field(default=None, alias='StreamId')
    activation: str | None = Field(default=None, alias='Activation')
    bias: TensorRtWeightsDescription | None = Field(default=None, alias='Bias')
    weights: TensorRtWeightsDescription | None = Field(default=None, alias='Weights')
    bias_as_activation_input_index: int | None = Field(default=None, alias='BiasAsActInputIdx')
    convolution_as_activation_input_index: int | None = Field(default=None, alias='ConvXAsActInputIdx')
    residual_as_activation_input_index: int | None = Field(default=None, alias='ResAsActInputIdx')
    dilation: tuple[int, ...] | None = Field(default=None, alias='Dilation')
    event_id: int | None = Field(default=None, alias='EventId')
    groups: int | None = Field(default=None, alias='Groups')
    has_bias: int | None = Field(default=None, alias='HasBias')
    has_dynamic_bias: int | None = Field(default=None, alias='HasDynamicBias')
    has_dynamic_filter: int | None = Field(default=None, alias='HasDynamicFilter')
    has_relu: int | None = Field(default=None, alias='HasReLU')
    has_residual: int | None = Field(default=None, alias='HasResidual')
    has_sparse_weights: int | None = Field(default=None, alias='HasSparseWeights')
    kernel: tuple[int, ...] | None = Field(default=None, alias='Kernel')
    output_maps: int | None = Field(default=None, alias='OutMaps')
    padding_mode: str | None = Field(default=None, alias='PaddingMode')
    post_padding: tuple[int, ...] | None = Field(default=None, alias='PostPadding')
    pre_padding: tuple[int, ...] | None = Field(default=None, alias='PrePadding')
    stride: tuple[int, ...] | None = Field(default=None, alias='Stride')


class TensorRtQdqInspection(FrozenModel):
    schema_version: Literal[2] = 2
    onnx_path: str = Field(min_length=1)
    onnx_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    engine_path: str = Field(min_length=1)
    engine_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    diagnostic_engine_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    quantize_linear_nodes: tuple[QuantizeLinearDescription, ...]
    onnx_operator_counts: tuple[Count, ...]
    tensorrt_layer_type_counts: tuple[Count, ...]
    tensorrt_int8_output_layer_count: int = Field(ge=0)
    tensorrt_layers: tuple[TensorRtLayerDescription, ...]


def _initializer_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {initializer.name: numpy_helper.to_array(initializer) for initializer in model.graph.initializer}


def _attribute_int(node: onnx.NodeProto, name: str, default: int) -> int:
    return next((attribute.i for attribute in node.attribute if attribute.name == name), default)


def _quantize_linear_nodes(model: onnx.ModelProto) -> tuple[QuantizeLinearDescription, ...]:
    initializers = _initializer_arrays(model)
    rows: list[QuantizeLinearDescription] = []
    for node in model.graph.node:
        if node.op_type != 'QuantizeLinear':
            continue
        scale = initializers[node.input[1]]
        zero_point = initializers[node.input[2]]
        rows.append(
            QuantizeLinearDescription(
                name=node.name,
                input=node.input[0],
                axis=_attribute_int(node, 'axis', 1),
                scale_shape=scale.shape,
                minimum_scale=float(scale.min()),
                maximum_scale=float(scale.max()),
                zero_point_dtype=str(zero_point.dtype),
                zero_point_minimum=int(zero_point.min()),
                zero_point_maximum=int(zero_point.max()),
            )
        )
    return tuple(rows)


def _build_detailed_engine(onnx_path: Path) -> tuple[trt.ICudaEngine, str]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = tuple(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise ValueError(f'TensorRT ONNX conversion failed for {onnx_path}: {" | ".join(errors)}')
    configuration = builder.create_builder_config()
    configuration.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
    configuration.builder_optimization_level = 3
    configuration.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    configuration.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, configuration)
    if serialized is None:
        raise ValueError(f'TensorRT failed to build the detailed diagnostic engine for {onnx_path}.')
    serialized_bytes = bytes(serialized)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_bytes)
    if engine is None:
        raise ValueError(f'TensorRT could not deserialize the detailed diagnostic engine for {onnx_path}.')
    return engine, hashlib.sha256(serialized_bytes).hexdigest()


def inspect(onnx_path: Path, engine_path: Path) -> TensorRtQdqInspection:
    model = onnx.load(onnx_path)
    engine, diagnostic_engine_sha256 = _build_detailed_engine(onnx_path)
    inspector = engine.create_engine_inspector()
    engine_information = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = tuple(TensorRtLayerDescription.model_validate(layer) for layer in engine_information['Layers'])
    layer_types = Counter(layer.layer_type for layer in layers)
    int8_outputs = sum(any('Int8' in output.format_and_data_type for output in layer.outputs) for layer in layers)
    return TensorRtQdqInspection(
        onnx_path=str(onnx_path),
        onnx_sha256=file_sha256(onnx_path),
        engine_path=str(engine_path),
        engine_sha256=file_sha256(engine_path),
        diagnostic_engine_sha256=diagnostic_engine_sha256,
        quantize_linear_nodes=_quantize_linear_nodes(model),
        onnx_operator_counts=tuple(
            Count(name=name, count=count)
            for name, count in sorted(Counter(node.op_type for node in model.graph.node).items())
        ),
        tensorrt_layer_type_counts=tuple(Count(name=name, count=count) for name, count in sorted(layer_types.items())),
        tensorrt_int8_output_layer_count=int8_outputs,
        tensorrt_layers=layers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect exact ONNX Q/DQ parameters and TensorRT engine tactics.')
    parser.add_argument('--onnx', required=True, type=Path)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    report = inspect(arguments.onnx, arguments.engine)
    write_text_atomically(arguments.output, report.model_dump_json(indent=2) + '\n')
    print(report.model_dump_json(indent=2))


if __name__ == '__main__':
    main()

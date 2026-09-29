"""Compare TensorRT float16 against FP8 and weight-only INT8 for one TorchScript inference model.

Only activation-by-weight products are quantized, so attention's fused kernel stays intact; the heads stay in
float16. Throughput is forward-only at batch 320 through CUDA graphs, and fidelity is measured against the float16
engine on held-out evaluation positions. FP8 needs an Ada or newer GPU. Run on an idle GPU.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import onnx
import tensorrt as trt
import torch
from modelopt.onnx.quantization import quantize
from pydantic import Field
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from src.util.provenance import SourceRevision, read_source_revision
from tools.benchmark_tensorrt_inference import (
    BATCH_SIZE,
    TENSORRT_BUILDER_OPTIMIZATION_LEVEL,
    TENSORRT_WORKSPACE_BYTES,
    ArtifactIdentity,
    _driver_version,
    _export_onnx,
    _input_overlap_count,
    _measure_runner,
    _TensorRtCudaGraphRunner,
)
from tools.measure_inference_precision_agreement import load_positions
from tools.tensorrt_benchmark_metrics import (
    FidelityLimits,
    FidelityMetrics,
    ModelOutputs,
    TimingDistribution,
    fidelity_failures,
    measure_fidelity,
)
from tools.tensorrt_weight_quantization import (
    WEIGHT_PRODUCT_OPERATOR_TYPES,
    GraphNode,
    LowPrecision,
    require_precision_support,
    weight_product_node_patterns,
)
from torch import Tensor

DEFAULT_CALIBRATION_POSITION_COUNT = 4 * BATCH_SIZE


@dataclass(frozen=True)
class ComparisonArguments:
    inference_model_path: Path
    dataset_path: Path
    precisions: tuple[LowPrecision, ...]
    fidelity_position_count: int
    calibration_position_count: int
    artifact_directory: Path
    output_path: Path
    gpu_id: int
    warmup_iterations: int
    repetitions: int
    iterations_per_repetition: int
    fidelity_limits: FidelityLimits
    acknowledge_gpu_load: bool


class HardwareIdentity(FrozenModel):
    device_name: str = Field(min_length=1)
    compute_capability: tuple[int, int]
    driver_version: str = Field(min_length=1)
    tensorrt_version: str = Field(min_length=1)
    modelopt_version: str = Field(min_length=1)


class Float16Measurement(FrozenModel):
    engine: ArtifactIdentity
    timing: TimingDistribution


class LowPrecisionMeasurement(FrozenModel):
    precision: LowPrecision
    quantized_onnx_model: ArtifactIdentity
    quantized_node_count: int = Field(gt=0)
    quantize_linear_node_count: int = Field(gt=0)
    engine: ArtifactIdentity
    timing: TimingDistribution
    speedup_over_float16: float = Field(gt=0.0)
    fidelity_against_float16: FidelityMetrics
    fidelity_failures: tuple[str, ...]


class LowPrecisionComparisonReport(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: SourceRevision
    inference_model: ArtifactIdentity
    dataset: ArtifactIdentity
    fidelity_positions: int = Field(gt=0, le=BATCH_SIZE)
    calibration_positions: int = Field(gt=0)
    calibration_fidelity_overlap_positions: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    hardware: HardwareIdentity
    fidelity_limits: FidelityLimits
    float16: Float16Measurement
    candidates: tuple[LowPrecisionMeasurement, ...] = Field(min_length=1)


def _graph_nodes(model: onnx.ModelProto) -> tuple[GraphNode, ...]:
    return tuple(
        GraphNode(name=node.name, operator_type=node.op_type, inputs=tuple(node.input), outputs=tuple(node.output))
        for node in model.graph.node
    )


def _quantize_weight_products(
    source_path: Path,
    output_path: Path,
    precision: LowPrecision,
    calibration_states: Tensor,
    device: torch.device,
) -> tuple[ArtifactIdentity, int, int]:
    source = onnx.load(source_path)
    patterns = weight_product_node_patterns(
        _graph_nodes(source), (initializer.name for initializer in source.graph.initializer)
    )
    if not patterns:
        raise ValueError(f'{source_path} has no activation-by-weight products to quantize.')
    output_path.unlink(missing_ok=True)
    quantize(
        onnx_path=str(source_path),
        quantize_mode=precision.value,
        calibration_data=calibration_states.to(torch.float32).numpy(),
        calibration_method='max',
        calibration_eps=[f'cuda:{device.index}', 'cpu'],
        op_types_to_quantize=sorted(WEIGHT_PRODUCT_OPERATOR_TYPES),
        nodes_to_quantize=list(patterns),
        high_precision_dtype='fp16',
        output_path=str(output_path),
    )
    quantized = onnx.load(output_path)
    onnx.checker.check_model(quantized, full_check=True)
    quantize_linear_count = sum(node.op_type == 'QuantizeLinear' for node in quantized.graph.node)
    if quantize_linear_count == 0:
        raise ValueError(f'ModelOpt produced no {precision.value} Q/DQ nodes for {source_path}.')
    return (
        ArtifactIdentity(path=str(output_path), sha256=file_sha256(output_path)),
        len(patterns),
        quantize_linear_count,
    )


def _build_engine(onnx_path: Path, output_path: Path, precision: LowPrecision | None) -> ArtifactIdentity:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = ' | '.join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise ValueError(f'TensorRT could not parse {onnx_path}: {errors}')
    configuration = builder.create_builder_config()
    configuration.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, TENSORRT_WORKSPACE_BYTES)
    configuration.builder_optimization_level = TENSORRT_BUILDER_OPTIMIZATION_LEVEL
    configuration.set_flag(trt.BuilderFlag.FP16)
    match precision:
        case LowPrecision.FP8:
            configuration.set_flag(trt.BuilderFlag.FP8)
        case LowPrecision.INT8:
            configuration.set_flag(trt.BuilderFlag.INT8)
        case None:
            pass
    serialized = builder.build_serialized_network(network, configuration)
    if serialized is None:
        raise ValueError(f'TensorRT failed to build an engine from {onnx_path}.')
    output_path.write_bytes(bytes(serialized))
    return ArtifactIdentity(path=str(output_path), sha256=file_sha256(output_path))


def _fidelity_outputs(runner: _TensorRtCudaGraphRunner, positions: int) -> ModelOutputs:
    padded = runner.outputs()
    return ModelOutputs(
        policy_logits=padded.policy_logits[:positions], wdl_probabilities=padded.wdl_probabilities[:positions]
    )


def run_comparison(arguments: ComparisonArguments) -> LowPrecisionComparisonReport:
    if not arguments.acknowledge_gpu_load:
        raise ValueError('Engine building and timing require --acknowledge-gpu-load.')
    if not 1 <= arguments.fidelity_position_count <= BATCH_SIZE:
        raise ValueError(f'Fidelity positions must lie in [1, {BATCH_SIZE}].')
    if arguments.calibration_position_count <= 0 or arguments.calibration_position_count % BATCH_SIZE:
        raise ValueError(f'Calibration needs a positive whole number of {BATCH_SIZE}-position batches.')
    device = torch.device('cuda', arguments.gpu_id)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    compute_capability = (properties.major, properties.minor)
    for precision in arguments.precisions:
        require_precision_support(precision, compute_capability)

    states, legal_action_mask = load_positions(
        arguments.dataset_path, arguments.fidelity_position_count + arguments.calibration_position_count
    )
    states = states.to(torch.int8)
    fidelity_states = states[: arguments.fidelity_position_count]
    fidelity_mask = legal_action_mask[: arguments.fidelity_position_count]
    calibration_states = states[arguments.fidelity_position_count :]
    padding = torch.arange(BATCH_SIZE - fidelity_states.shape[0]) % fidelity_states.shape[0]
    runner_states = torch.cat((fidelity_states, fidelity_states[padding]), dim=0)

    directory = arguments.artifact_directory
    directory.mkdir(parents=True, exist_ok=True)
    stem = arguments.inference_model_path.stem
    float16_onnx = directory / f'{stem}-b{BATCH_SIZE}-fp16.onnx'
    float32_onnx = directory / f'{stem}-b{BATCH_SIZE}-fp32.onnx'
    _export_onnx(arguments.inference_model_path, float16_onnx, torch.float16)
    _export_onnx(arguments.inference_model_path, float32_onnx, torch.float32)

    def timed(runner: _TensorRtCudaGraphRunner) -> TimingDistribution:
        return _measure_runner(
            runner, arguments.warmup_iterations, arguments.repetitions, arguments.iterations_per_repetition, device
        )

    float16_engine_path = directory / f'{stem}-b{BATCH_SIZE}-fp16.engine'
    float16_engine = _build_engine(float16_onnx, float16_engine_path, None)
    float16_runner = _TensorRtCudaGraphRunner(float16_engine_path, runner_states, device, arguments.warmup_iterations)
    float16_outputs = _fidelity_outputs(float16_runner, arguments.fidelity_position_count)
    float16 = Float16Measurement(engine=float16_engine, timing=timed(float16_runner))
    del float16_runner

    candidates: list[LowPrecisionMeasurement] = []
    for precision in arguments.precisions:
        quantized_onnx = directory / f'{stem}-b{BATCH_SIZE}-{precision.value}-weights.onnx'
        quantized_artifact, quantized_node_count, quantize_linear_count = _quantize_weight_products(
            float32_onnx, quantized_onnx, precision, calibration_states, device
        )
        engine_path = directory / f'{stem}-b{BATCH_SIZE}-{precision.value}-weights.engine'
        engine = _build_engine(quantized_onnx, engine_path, precision)
        runner = _TensorRtCudaGraphRunner(engine_path, runner_states, device, arguments.warmup_iterations)
        fidelity = measure_fidelity(
            float16_outputs, _fidelity_outputs(runner, arguments.fidelity_position_count), fidelity_mask
        )
        timing = timed(runner)
        del runner
        candidates.append(
            LowPrecisionMeasurement(
                precision=precision,
                quantized_onnx_model=quantized_artifact,
                quantized_node_count=quantized_node_count,
                quantize_linear_node_count=quantize_linear_count,
                engine=engine,
                timing=timing,
                speedup_over_float16=timing.median_positions_per_second / float16.timing.median_positions_per_second,
                fidelity_against_float16=fidelity,
                fidelity_failures=fidelity_failures(fidelity, arguments.fidelity_limits),
            )
        )

    return LowPrecisionComparisonReport(
        source_revision=read_source_revision(),
        inference_model=ArtifactIdentity(
            path=str(arguments.inference_model_path), sha256=file_sha256(arguments.inference_model_path)
        ),
        dataset=ArtifactIdentity(path=str(arguments.dataset_path), sha256=file_sha256(arguments.dataset_path)),
        fidelity_positions=arguments.fidelity_position_count,
        calibration_positions=arguments.calibration_position_count,
        calibration_fidelity_overlap_positions=_input_overlap_count(calibration_states, fidelity_states),
        batch_size=BATCH_SIZE,
        hardware=HardwareIdentity(
            device_name=properties.name,
            compute_capability=compute_capability,
            driver_version=_driver_version(),
            tensorrt_version=trt.__version__,
            modelopt_version=version('nvidia-modelopt'),
        ),
        fidelity_limits=arguments.fidelity_limits,
        float16=float16,
        candidates=tuple(candidates),
    )


def parse_arguments() -> ComparisonArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference-model', type=Path, required=True, help='TorchScript inference model.')
    parser.add_argument('--dataset', type=Path, required=True, help='Evaluation dataset with its manifest.')
    parser.add_argument(
        '--precision', type=LowPrecision, choices=tuple(LowPrecision), action='append', dest='precisions'
    )
    parser.add_argument('--fidelity-position-count', type=int, default=BATCH_SIZE)
    parser.add_argument('--calibration-position-count', type=int, default=DEFAULT_CALIBRATION_POSITION_COUNT)
    parser.add_argument('--artifact-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--warmup-iterations', type=int, default=50)
    parser.add_argument('--repetitions', type=int, default=15)
    parser.add_argument('--iterations-per-repetition', type=int, default=100)
    parser.add_argument('--minimum-policy-top1-agreement', type=float, default=0.98)
    parser.add_argument('--maximum-mean-policy-kl-divergence', type=float, default=0.005)
    parser.add_argument('--maximum-wdl-mean-absolute-error', type=float, default=0.01)
    parser.add_argument('--maximum-expected-value-mean-absolute-error', type=float, default=0.015)
    parser.add_argument('--acknowledge-gpu-load', action='store_true')
    parsed = parser.parse_args()
    return ComparisonArguments(
        inference_model_path=parsed.inference_model.resolve(),
        dataset_path=parsed.dataset.resolve(),
        precisions=tuple(dict.fromkeys(parsed.precisions or (LowPrecision.FP8,))),
        fidelity_position_count=parsed.fidelity_position_count,
        calibration_position_count=parsed.calibration_position_count,
        artifact_directory=parsed.artifact_directory.resolve(),
        output_path=parsed.output.resolve(),
        gpu_id=parsed.gpu_id,
        warmup_iterations=parsed.warmup_iterations,
        repetitions=parsed.repetitions,
        iterations_per_repetition=parsed.iterations_per_repetition,
        fidelity_limits=FidelityLimits(
            minimum_policy_top1_agreement=parsed.minimum_policy_top1_agreement,
            maximum_mean_policy_kl_divergence=parsed.maximum_mean_policy_kl_divergence,
            maximum_wdl_mean_absolute_error=parsed.maximum_wdl_mean_absolute_error,
            maximum_expected_value_mean_absolute_error=parsed.maximum_expected_value_mean_absolute_error,
        ),
        acknowledge_gpu_load=parsed.acknowledge_gpu_load,
    )


def main() -> None:
    arguments = parse_arguments()
    report = run_comparison(arguments)
    write_text_atomically(arguments.output_path, report.model_dump_json(indent=2) + '\n')
    print(json.dumps(report.model_dump(mode='json'), indent=2))


if __name__ == '__main__':
    main()

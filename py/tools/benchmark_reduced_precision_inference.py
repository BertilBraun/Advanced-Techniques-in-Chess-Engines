"""Measures what reduced precision can and cannot buy the self-play forward pass.

Two arms answer two different questions.

The network arm times the production chess networks end to end in every precision and memory format
the native runtime can actually select, and reports the achieved rate against the device's dense
tensor-core peak. That is the number that decides whether the forward pass is anywhere near the
precision tier it already has.

The roofline arm times the bare implicit-GEMM shape of one trunk convolution in bfloat16, float16,
int8 and float8. There is no int8 or float8 convolution on this path, so this arm measures the
ceiling such a convolution could reach rather than a runnable network. Compare its int8 speedup
against the network arm's distance from peak before spending anything on a quantized runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.network import (
    DensePolicyHeadConfiguration,
    GlobalPoolingResidualContext,
    InferenceNetwork,
    Network,
    NetworkParams,
    ResidualContextPlacement,
)
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.provenance import SourceRevision, read_source_revision
from torch import Tensor, nn

# The three progressive-sizing rungs of vast-chess-4day-production-v9.
PRODUCTION_MODELS = (
    ('chess-cnn-12x128-dense4', 12, 128),
    ('chess-cnn-14x152-dense4', 14, 152),
    ('chess-cnn-18x176-dense4', 18, 176),
)
# 320 is the self-play cap, 241 its measured average fill, 64 the evaluation cap.
PRODUCTION_BATCH_SIZES = (64, 241, 320)
POLICY_HEAD_CHANNELS = 4
VALUE_CHANNELS = 2
VALUE_FC_SIZE = 48
# Ada consumer tensor cores run bfloat16 at 4x and int8/float8 at 8x the FP32 CUDA-core rate.
BFLOAT16_PEAK_MULTIPLIER = 4.0
INT8_PEAK_MULTIPLIER = 8.0
CONVOLUTION_KERNEL_AREA = 9


class Precision(str, Enum):
    BFLOAT16 = 'bfloat16'
    FLOAT16 = 'float16'
    FLOAT32 = 'float32'

    @property
    def torch_dtype(self) -> torch.dtype:
        match self:
            case Precision.BFLOAT16:
                return torch.bfloat16
            case Precision.FLOAT16:
                return torch.float16
            case Precision.FLOAT32:
                return torch.float32


class MemoryFormat(str, Enum):
    CONTIGUOUS = 'contiguous'
    CHANNELS_LAST = 'channels_last'

    @property
    def torch_memory_format(self) -> torch.memory_format:
        match self:
            case MemoryFormat.CONTIGUOUS:
                return torch.contiguous_format
            case MemoryFormat.CHANNELS_LAST:
                return torch.channels_last


class GemmPrecision(str, Enum):
    BFLOAT16 = 'bfloat16'
    FLOAT16 = 'float16'
    INT8 = 'int8'
    FLOAT8_E4M3 = 'float8_e4m3'


@dataclass(frozen=True)
class BenchmarkArguments:
    output_path: Path
    gpu_id: int
    batch_sizes: tuple[int, ...]
    warmup_iterations: int
    duration_seconds: float
    acknowledge_gpu_load: bool


class HardwareDescription(FrozenModel):
    gpu_id: int = Field(ge=0)
    device_name: str = Field(min_length=1)
    compute_capability: tuple[int, int]
    multiprocessor_count: int = Field(gt=0)
    torch_version: str = Field(min_length=1)
    cuda_version: str = Field(min_length=1)
    fp32_cuda_core_peak_tflops: float = Field(gt=0.0)
    bfloat16_tensor_peak_tflops: float = Field(gt=0.0)
    int8_tensor_peak_tops: float = Field(gt=0.0)


class ModelDescription(FrozenModel):
    model_id: str = Field(min_length=1)
    block_count: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    parameters: int = Field(gt=0)
    gflop_per_position: float = Field(gt=0.0)
    # int8 and float8 tensor-core convolution kernels need the channel count to be a multiple of 16.
    hidden_size_is_int8_aligned: bool


class NetworkMeasurement(FrozenModel):
    model_id: str = Field(min_length=1)
    precision: Precision
    memory_format: MemoryFormat
    cudnn_benchmark: bool
    batch_size: int = Field(gt=0)
    measured_iterations: int = Field(gt=0)
    mean_batch_milliseconds: float = Field(gt=0.0)
    positions_per_second: float = Field(gt=0.0)
    achieved_tflops: float = Field(gt=0.0)
    fraction_of_bfloat16_peak: float = Field(gt=0.0)


class GemmMeasurement(FrozenModel):
    model_id: str = Field(min_length=1)
    precision: GemmPrecision
    batch_size: int = Field(gt=0)
    m: int = Field(gt=0)
    n: int = Field(gt=0)
    k: int = Field(gt=0)
    available: bool
    unavailable_reason: str | None
    mean_milliseconds: float | None
    achieved_tops: float | None
    speedup_over_bfloat16: float | None


class ReducedPrecisionBenchmarkReport(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: SourceRevision
    hardware: HardwareDescription
    batch_sizes: tuple[int, ...] = Field(min_length=1)
    warmup_iterations: int = Field(ge=0)
    target_duration_seconds: float = Field(gt=0.0)
    models: tuple[ModelDescription, ...] = Field(min_length=1)
    network_measurements: tuple[NetworkMeasurement, ...] = Field(min_length=1)
    gemm_measurements: tuple[GemmMeasurement, ...] = Field(min_length=1)


def _build_inference_network(block_count: int, hidden_size: int, device: torch.device) -> InferenceNetwork:
    parameters = NetworkParams(
        num_layers=block_count,
        hidden_size=hidden_size,
        residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK),
        policy_head=DensePolicyHeadConfiguration(channels=POLICY_HEAD_CHANNELS),
        num_value_channels=VALUE_CHANNELS,
        value_fc_size=VALUE_FC_SIZE,
    )
    training_network = Network(parameters, device, CHESS_NETWORK_DIMENSIONS, ())
    inference_network = InferenceNetwork(training_network)
    inference_network.eval()
    inference_network.fuse_model()
    return inference_network


def _multiply_accumulate_count(inference_network: InferenceNetwork) -> int:
    total = 0
    handles = []

    def record(module: nn.Module, _inputs: object, output: Tensor) -> None:
        nonlocal total
        match module:
            case nn.Conv2d():
                total += (
                    module.in_channels
                    * module.out_channels
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                    * output.shape[2]
                    * output.shape[3]
                )
            case nn.Linear():
                total += module.in_features * module.out_features

    for module in inference_network.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(record))
    device = next(inference_network.parameters()).device
    with torch.inference_mode():
        inference_network(
            torch.zeros(
                1,
                CHESS_NETWORK_DIMENSIONS.channels,
                CHESS_NETWORK_DIMENSIONS.rows,
                CHESS_NETWORK_DIMENSIONS.columns,
                device=device,
            )
        )
    for handle in handles:
        handle.remove()
    return total


def _calibrated_iteration_count(run: object, device: torch.device, duration_seconds: float) -> int:
    calibration_iterations = 5
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(calibration_iterations):
        run()  # type: ignore[operator]
    torch.cuda.synchronize(device)
    calibration_seconds = time.perf_counter() - started
    return max(1, math.ceil(duration_seconds * calibration_iterations / calibration_seconds))


def _time_seconds_per_iteration(
    run: object,
    device: torch.device,
    warmup_iterations: int,
    duration_seconds: float,
) -> tuple[float, int]:
    for _ in range(warmup_iterations):
        run()  # type: ignore[operator]
    iterations = _calibrated_iteration_count(run, device, duration_seconds)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        run()  # type: ignore[operator]
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) / iterations, iterations


def _measure_network(
    model: ModelDescription,
    inference_network: InferenceNetwork,
    precision: Precision,
    memory_format: MemoryFormat,
    cudnn_benchmark: bool,
    batch_size: int,
    device: torch.device,
    arguments: BenchmarkArguments,
    bfloat16_peak_tflops: float,
) -> NetworkMeasurement:
    torch.backends.cudnn.benchmark = cudnn_benchmark
    network = inference_network.to(dtype=precision.torch_dtype, memory_format=memory_format.torch_memory_format)
    scripted = torch.jit.freeze(torch.jit.script(network))
    states = torch.zeros(
        batch_size,
        CHESS_NETWORK_DIMENSIONS.channels,
        CHESS_NETWORK_DIMENSIONS.rows,
        CHESS_NETWORK_DIMENSIONS.columns,
        device=device,
        dtype=precision.torch_dtype,
    ).to(memory_format=memory_format.torch_memory_format)

    with torch.inference_mode():
        seconds, iterations = _time_seconds_per_iteration(
            lambda: scripted(states), device, arguments.warmup_iterations, arguments.duration_seconds
        )
    achieved_tflops = model.gflop_per_position * batch_size / seconds / 1e3
    return NetworkMeasurement(
        model_id=model.model_id,
        precision=precision,
        memory_format=memory_format,
        cudnn_benchmark=cudnn_benchmark,
        batch_size=batch_size,
        measured_iterations=iterations,
        mean_batch_milliseconds=seconds * 1e3,
        positions_per_second=batch_size / seconds,
        achieved_tflops=achieved_tflops,
        fraction_of_bfloat16_peak=achieved_tflops / bfloat16_peak_tflops,
    )


def _gemm_runner(precision: GemmPrecision, m: int, n: int, k: int, device: torch.device) -> object:
    """Builds the timed callable for one precision, or raises with the reason it is unavailable."""
    match precision:
        case GemmPrecision.BFLOAT16 | GemmPrecision.FLOAT16:
            dtype = torch.bfloat16 if precision is GemmPrecision.BFLOAT16 else torch.float16
            left = torch.randn(m, k, device=device, dtype=dtype)
            right = torch.randn(k, n, device=device, dtype=dtype)
            return lambda: torch.matmul(left, right)
        case GemmPrecision.INT8:
            if k % 16 or n % 16:
                raise ValueError(f'int8 tensor-core GEMM needs K and N to be multiples of 16, got K={k}, N={n}')
            left = torch.randint(-127, 127, (m, k), device=device, dtype=torch.int8)
            right = torch.randint(-127, 127, (k, n), device=device, dtype=torch.int8)
            return lambda: torch._int_mm(left, right)
        case GemmPrecision.FLOAT8_E4M3:
            if k % 16 or n % 16:
                raise ValueError(f'float8 tensor-core GEMM needs K and N to be multiples of 16, got K={k}, N={n}')
            left = torch.randn(m, k, device=device).to(torch.float8_e4m3fn)
            # _scaled_mm requires a column-major right operand.
            right = torch.randn(k, n, device=device).to(torch.float8_e4m3fn).t().contiguous().t()
            scale = torch.tensor(1.0, device=device)
            return lambda: torch._scaled_mm(left, right, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)


def _measure_gemm(
    model: ModelDescription,
    precision: GemmPrecision,
    batch_size: int,
    device: torch.device,
    arguments: BenchmarkArguments,
    bfloat16_milliseconds: float | None,
) -> GemmMeasurement:
    # One trunk 3x3 convolution as its implicit GEMM: rows are board squares, K is the 3x3 gather.
    m = batch_size * CHESS_NETWORK_DIMENSIONS.rows * CHESS_NETWORK_DIMENSIONS.columns
    n = model.hidden_size
    k = model.hidden_size * CONVOLUTION_KERNEL_AREA
    unavailable = GemmMeasurement(
        model_id=model.model_id,
        precision=precision,
        batch_size=batch_size,
        m=m,
        n=n,
        k=k,
        available=False,
        unavailable_reason=None,
        mean_milliseconds=None,
        achieved_tops=None,
        speedup_over_bfloat16=None,
    )
    try:
        run = _gemm_runner(precision, m, n, k, device)
        with torch.inference_mode():
            seconds, _ = _time_seconds_per_iteration(
                run, device, arguments.warmup_iterations, arguments.duration_seconds
            )
    except Exception as failure:  # noqa: BLE001 - every backend gap must be recorded, not fatal
        return unavailable.model_copy(update={'unavailable_reason': f'{type(failure).__name__}: {failure}'})
    milliseconds = seconds * 1e3
    return GemmMeasurement(
        model_id=model.model_id,
        precision=precision,
        batch_size=batch_size,
        m=m,
        n=n,
        k=k,
        available=True,
        unavailable_reason=None,
        mean_milliseconds=milliseconds,
        achieved_tops=2.0 * m * n * k / seconds / 1e12,
        speedup_over_bfloat16=None if bfloat16_milliseconds is None else bfloat16_milliseconds / milliseconds,
    )


def _describe_hardware(gpu_id: int, device: torch.device) -> HardwareDescription:
    properties = torch.cuda.get_device_properties(device)
    # clock_rate is in kHz and reports the boost clock.
    boost_hertz = float(getattr(properties, 'clock_rate', 0) or 0) * 1e3
    if boost_hertz <= 0.0:
        raise ValueError('The CUDA device did not report a clock rate, so no peak can be derived.')
    cuda_cores = properties.multi_processor_count * 128
    fp32_peak = cuda_cores * 2 * boost_hertz / 1e12
    return HardwareDescription(
        gpu_id=gpu_id,
        device_name=properties.name,
        compute_capability=(properties.major, properties.minor),
        multiprocessor_count=properties.multi_processor_count,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or 'unknown',
        fp32_cuda_core_peak_tflops=fp32_peak,
        bfloat16_tensor_peak_tflops=fp32_peak * BFLOAT16_PEAK_MULTIPLIER,
        int8_tensor_peak_tops=fp32_peak * INT8_PEAK_MULTIPLIER,
    )


def run_benchmark(arguments: BenchmarkArguments) -> ReducedPrecisionBenchmarkReport:
    if not arguments.acknowledge_gpu_load:
        raise ValueError('The reduced-precision benchmark requires --acknowledge-gpu-load.')
    if not torch.cuda.is_available():
        raise ValueError('The reduced-precision benchmark requires CUDA.')
    if arguments.gpu_id >= torch.cuda.device_count():
        raise ValueError(f'GPU ID {arguments.gpu_id} is not available.')

    device = torch.device('cuda', arguments.gpu_id)
    torch.cuda.set_device(device)
    hardware = _describe_hardware(arguments.gpu_id, device)

    models: list[ModelDescription] = []
    network_measurements: list[NetworkMeasurement] = []
    gemm_measurements: list[GemmMeasurement] = []

    for model_id, block_count, hidden_size in PRODUCTION_MODELS:
        inference_network = _build_inference_network(block_count, hidden_size, device)
        model = ModelDescription(
            model_id=model_id,
            block_count=block_count,
            hidden_size=hidden_size,
            parameters=sum(parameter.numel() for parameter in inference_network.parameters()),
            gflop_per_position=2.0 * _multiply_accumulate_count(inference_network) / 1e9,
            hidden_size_is_int8_aligned=hidden_size % 16 == 0,
        )
        models.append(model)

        for batch_size in arguments.batch_sizes:
            for precision in Precision:
                for memory_format in MemoryFormat:
                    for cudnn_benchmark in (False, True):
                        network_measurements.append(
                            _measure_network(
                                model,
                                inference_network,
                                precision,
                                memory_format,
                                cudnn_benchmark,
                                batch_size,
                                device,
                                arguments,
                                hardware.bfloat16_tensor_peak_tflops,
                            )
                        )
            bfloat16_milliseconds: float | None = None
            for precision in GemmPrecision:
                measurement = _measure_gemm(model, precision, batch_size, device, arguments, bfloat16_milliseconds)
                if precision is GemmPrecision.BFLOAT16 and measurement.available:
                    bfloat16_milliseconds = measurement.mean_milliseconds
                gemm_measurements.append(measurement)

    return ReducedPrecisionBenchmarkReport(
        source_revision=read_source_revision(),
        hardware=hardware,
        batch_sizes=arguments.batch_sizes,
        warmup_iterations=arguments.warmup_iterations,
        target_duration_seconds=arguments.duration_seconds,
        models=tuple(models),
        network_measurements=tuple(network_measurements),
        gemm_measurements=tuple(gemm_measurements),
    )


def parse_arguments() -> BenchmarkArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=list(PRODUCTION_BATCH_SIZES))
    parser.add_argument('--warmup-iterations', type=int, default=16)
    parser.add_argument('--duration-seconds', type=float, default=2.0)
    parser.add_argument('--acknowledge-gpu-load', action='store_true')
    parsed = parser.parse_args()
    return BenchmarkArguments(
        output_path=parsed.output_path,
        gpu_id=parsed.gpu_id,
        batch_sizes=tuple(parsed.batch_sizes),
        warmup_iterations=parsed.warmup_iterations,
        duration_seconds=parsed.duration_seconds,
        acknowledge_gpu_load=parsed.acknowledge_gpu_load,
    )


def main() -> None:
    arguments = parse_arguments()
    report = run_benchmark(arguments)
    write_text_atomically(arguments.output_path, report.model_dump_json(indent=2) + '\n')
    summary = {
        'device': report.hardware.device_name,
        'bfloat16_tensor_peak_tflops': round(report.hardware.bfloat16_tensor_peak_tflops, 1),
        'int8_tensor_peak_tops': round(report.hardware.int8_tensor_peak_tops, 1),
    }
    print(json.dumps(summary, indent=2))
    for measurement in report.network_measurements:
        print(
            f'{measurement.model_id} b{measurement.batch_size} {measurement.precision.value}/'
            f'{measurement.memory_format.value} benchmark={int(measurement.cudnn_benchmark)}: '
            f'{measurement.positions_per_second:,.0f} pos/s, {measurement.achieved_tflops:.1f} TFLOPS, '
            f'{100 * measurement.fraction_of_bfloat16_peak:.1f}% of bf16 peak'
        )
    for measurement in report.gemm_measurements:
        if not measurement.available:
            print(f'{measurement.model_id} b{measurement.batch_size} {measurement.precision.value}: unavailable')
            continue
        speedup = (
            '' if measurement.speedup_over_bfloat16 is None else f', {measurement.speedup_over_bfloat16:.2f}x bf16'
        )
        print(
            f'{measurement.model_id} b{measurement.batch_size} GEMM {measurement.precision.value}: '
            f'{measurement.achieved_tops:.1f} TOPS{speedup}'
        )


if __name__ == '__main__':
    main()

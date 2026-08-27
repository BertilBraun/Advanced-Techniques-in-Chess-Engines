"""Profiles the CUDA kernels in one warmed convolutional inference call."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.provenance import SourceRevision, read_source_revision
from tools.benchmark_reduced_precision_inference import MemoryFormat, Precision, _build_inference_network
from torch.autograd.profiler_util import FunctionEvent


class KernelClass(str, Enum):
    CONVOLUTION = 'convolution'
    BATCH_NORM = 'batch_norm'
    ACTIVATION = 'activation'
    RESIDUAL_ADD = 'residual_add'
    GLOBAL_POOLING = 'global_pooling'
    MEMORY_COPY = 'memory_copy'
    OTHER = 'other'


@dataclass(frozen=True)
class ProfileArguments:
    output_path: Path
    gpu_id: int
    batch_size: int
    block_count: int
    hidden_size: int
    precision: Precision
    memory_format: MemoryFormat
    cudnn_benchmark: bool
    warmup_iterations: int
    acknowledge_gpu_load: bool


@dataclass(frozen=True)
class KernelIdentity:
    kernel_class: KernelClass
    name: str
    operation_path: tuple[str, ...]


class KernelMeasurement(FrozenModel):
    kernel_class: KernelClass
    name: str = Field(min_length=1)
    operation_path: tuple[str, ...]
    count: int = Field(gt=0)
    total_microseconds: float = Field(gt=0.0)
    fraction_of_gpu_span: float = Field(ge=0.0)


class KernelClassMeasurement(FrozenModel):
    kernel_class: KernelClass
    count: int = Field(ge=0)
    total_microseconds: float = Field(ge=0.0)
    fraction_of_gpu_span: float = Field(ge=0.0)


class KernelProfileReport(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: SourceRevision
    device_name: str = Field(min_length=1)
    batch_size: int = Field(gt=0)
    block_count: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    precision: Precision
    memory_format: MemoryFormat
    cudnn_benchmark: bool
    kernel_count: int = Field(gt=0)
    gpu_span_microseconds: float = Field(gt=0.0)
    gpu_busy_microseconds: float = Field(gt=0.0)
    launch_gap_microseconds: float = Field(ge=0.0)
    launch_gap_fraction: float = Field(ge=0.0)
    classes: tuple[KernelClassMeasurement, ...]
    kernels: tuple[KernelMeasurement, ...]


def operation_path(event: FunctionEvent) -> tuple[str, ...]:
    names: list[str] = []
    parent = event.cpu_parent
    while parent is not None:
        names.append(parent.name)
        parent = parent.cpu_parent
    return tuple(reversed(names))


def classify_kernel(name: str, operations: tuple[str, ...]) -> KernelClass:
    description = ' '.join((*operations, name)).lower()
    if 'memcpy' in description or 'memset' in description or 'aten::copy_' in description:
        return KernelClass.MEMORY_COPY
    if 'batch_norm' in description or 'batchnorm' in description:
        return KernelClass.BATCH_NORM
    if 'convolution' in description or 'conv2d' in description or 'cudnn' in description:
        return KernelClass.CONVOLUTION
    if 'relu' in description or 'threshold' in description or 'clamp' in description:
        return KernelClass.ACTIVATION
    if 'aten::add' in description:
        return KernelClass.RESIDUAL_ADD
    if any(operation in description for operation in ('aten::mean', 'aten::sum', 'adaptive_avg_pool', 'global_pool')):
        return KernelClass.GLOBAL_POOLING
    return KernelClass.OTHER


def interval_union_microseconds(intervals: tuple[tuple[float, float], ...]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    union = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        union += current_end - current_start
        current_start, current_end = start, end
    return union + current_end - current_start


def _cuda_events(profiler: torch.profiler.profile) -> tuple[FunctionEvent, ...]:
    return tuple(event for event in profiler.events() if event.device_type is torch.autograd.DeviceType.CUDA)


def _build_report(arguments: ProfileArguments, events: tuple[FunctionEvent, ...]) -> KernelProfileReport:
    if not events:
        raise ValueError('The profiler did not record any CUDA kernel or memory-copy events.')
    intervals = tuple((event.time_range.start, event.time_range.end) for event in events)
    first_start = min(start for start, _end in intervals)
    last_end = max(end for _start, end in intervals)
    gpu_span = last_end - first_start
    gpu_busy = interval_union_microseconds(intervals)

    aggregates: defaultdict[KernelIdentity, list[float]] = defaultdict(list)
    for event in events:
        operations = operation_path(event)
        identity = KernelIdentity(classify_kernel(event.name, operations), event.name, operations)
        aggregates[identity].append(event.time_range.elapsed_us())

    kernels = tuple(
        sorted(
            (
                KernelMeasurement(
                    kernel_class=identity.kernel_class,
                    name=identity.name,
                    operation_path=identity.operation_path,
                    count=len(durations),
                    total_microseconds=sum(durations),
                    fraction_of_gpu_span=sum(durations) / gpu_span,
                )
                for identity, durations in aggregates.items()
            ),
            key=lambda measurement: measurement.total_microseconds,
            reverse=True,
        )
    )
    classes = tuple(
        KernelClassMeasurement(
            kernel_class=kernel_class,
            count=sum(kernel.count for kernel in kernels if kernel.kernel_class is kernel_class),
            total_microseconds=sum(
                kernel.total_microseconds for kernel in kernels if kernel.kernel_class is kernel_class
            ),
            fraction_of_gpu_span=sum(
                kernel.total_microseconds for kernel in kernels if kernel.kernel_class is kernel_class
            )
            / gpu_span,
        )
        for kernel_class in KernelClass
    )
    launch_gap = max(0.0, gpu_span - gpu_busy)
    return KernelProfileReport(
        source_revision=read_source_revision(),
        device_name=torch.cuda.get_device_name(arguments.gpu_id),
        batch_size=arguments.batch_size,
        block_count=arguments.block_count,
        hidden_size=arguments.hidden_size,
        precision=arguments.precision,
        memory_format=arguments.memory_format,
        cudnn_benchmark=arguments.cudnn_benchmark,
        kernel_count=len(events),
        gpu_span_microseconds=gpu_span,
        gpu_busy_microseconds=gpu_busy,
        launch_gap_microseconds=launch_gap,
        launch_gap_fraction=launch_gap / gpu_span,
        classes=classes,
        kernels=kernels,
    )


def run_profile(arguments: ProfileArguments) -> KernelProfileReport:
    if not arguments.acknowledge_gpu_load:
        raise ValueError('The inference profile requires --acknowledge-gpu-load.')
    if not torch.cuda.is_available():
        raise ValueError('The inference profile requires CUDA.')
    if arguments.gpu_id >= torch.cuda.device_count():
        raise ValueError(f'GPU ID {arguments.gpu_id} is not available.')

    device = torch.device('cuda', arguments.gpu_id)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = arguments.cudnn_benchmark
    network = _build_inference_network(arguments.block_count, arguments.hidden_size, device).to(
        dtype=arguments.precision.torch_dtype,
        memory_format=arguments.memory_format.torch_memory_format,
    )
    scripted = torch.jit.freeze(torch.jit.script(network))
    states = torch.zeros(
        arguments.batch_size,
        CHESS_NETWORK_DIMENSIONS.channels,
        CHESS_NETWORK_DIMENSIONS.rows,
        CHESS_NETWORK_DIMENSIONS.columns,
        device=device,
        dtype=arguments.precision.torch_dtype,
    ).to(memory_format=arguments.memory_format.torch_memory_format)

    with torch.inference_mode():
        for _ in range(arguments.warmup_iterations):
            scripted(states)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
            record_shapes=True,
        ) as profiler:
            scripted(states)
            torch.cuda.synchronize(device)
    return _build_report(arguments, _cuda_events(profiler))


def parse_arguments() -> ProfileArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=320)
    parser.add_argument('--block-count', type=int, default=12)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--precision', type=Precision, choices=tuple(Precision), default=Precision.BFLOAT16)
    parser.add_argument(
        '--memory-format', type=MemoryFormat, choices=tuple(MemoryFormat), default=MemoryFormat.CONTIGUOUS
    )
    parser.add_argument('--cudnn-benchmark', action='store_true')
    parser.add_argument('--warmup-iterations', type=int, default=32)
    parser.add_argument('--acknowledge-gpu-load', action='store_true')
    parsed = parser.parse_args()
    return ProfileArguments(
        output_path=parsed.output_path,
        gpu_id=parsed.gpu_id,
        batch_size=parsed.batch_size,
        block_count=parsed.block_count,
        hidden_size=parsed.hidden_size,
        precision=parsed.precision,
        memory_format=parsed.memory_format,
        cudnn_benchmark=parsed.cudnn_benchmark,
        warmup_iterations=parsed.warmup_iterations,
        acknowledge_gpu_load=parsed.acknowledge_gpu_load,
    )


def main() -> None:
    arguments = parse_arguments()
    report = run_profile(arguments)
    write_text_atomically(arguments.output_path, report.model_dump_json(indent=2) + '\n')
    print(
        json.dumps(
            {'kernel_count': report.kernel_count, 'gpu_span_microseconds': report.gpu_span_microseconds}, indent=2
        )
    )
    for measurement in report.classes:
        print(
            f'{measurement.kernel_class.value:16s} {measurement.count:4d} kernels '
            f'{measurement.total_microseconds:9.1f} us {100 * measurement.fraction_of_gpu_span:6.2f}%'
        )
    print(f'launch_gap       {report.launch_gap_microseconds:9.1f} us {100 * report.launch_gap_fraction:6.2f}%')


if __name__ == '__main__':
    main()

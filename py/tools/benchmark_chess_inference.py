from __future__ import annotations

import argparse
import math
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field
from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.network import (
    AttentionNetworkParams,
    Chess76PlaneDirectPolicyHeadConfiguration,
    GlobalPoolingResidualContext,
    Network,
    NetworkConfiguration,
    NetworkParams,
    ResidualContextPlacement,
)
from src.training.targets import SearchCorrectionHeadLayout, build_training_target_layout
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from src.util.hashing import file_sha256
from src.util.provenance import SourceRevision, read_source_revision

DEFAULT_CONFIGURATION_PATH = Path(__file__).resolve().parents[1] / 'configs/production/vast-chess-8gpu-optimal.yaml'


class ExecutionMode(str, Enum):
    EAGER = 'eager'
    SCRIPTED = 'scripted'
    COMPILED = 'compiled'


class AttentionBackend(str, Enum):
    AUTOMATIC = 'automatic'
    FLASH = 'flash'
    MEMORY_EFFICIENT = 'memory_efficient'
    CUDNN = 'cudnn'
    MATH = 'math'


@dataclass(frozen=True)
class BenchmarkModel:
    model_id: str
    training_start_days: float
    network: NetworkConfiguration


@dataclass(frozen=True)
class BenchmarkArguments:
    configuration_path: Path
    output_path: Path
    gpu_id: int
    batch_sizes: tuple[int, ...]
    execution_modes: tuple[ExecutionMode, ...]
    attention_backend: AttentionBackend
    warmup_iterations: int
    duration_seconds: float
    include_diagnostic_controls: bool
    acknowledge_gpu_load: bool


class ParameterCounts(FrozenModel):
    backbone: int = Field(gt=0)
    primary_policy_head: int = Field(gt=0)
    value_head: int = Field(gt=0)
    auxiliary_heads: int = Field(ge=0)
    inference_total: int = Field(gt=0)
    training_total: int = Field(gt=0)


class ModelDescription(FrozenModel):
    model_id: str = Field(min_length=1)
    training_start_days: float = Field(ge=0.0)
    network: NetworkConfiguration
    parameters: ParameterCounts


class InferenceMeasurement(FrozenModel):
    model_id: str = Field(min_length=1)
    execution_mode: ExecutionMode
    batch_size: int = Field(gt=0)
    measured_iterations: int = Field(gt=0)
    elapsed_seconds: float = Field(gt=0.0)
    mean_batch_seconds: float = Field(gt=0.0)
    positions_per_second: float = Field(gt=0.0)
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)


class HardwareDescription(FrozenModel):
    gpu_id: int = Field(ge=0)
    device_name: str = Field(min_length=1)
    compute_capability: tuple[int, int]
    torch_version: str = Field(min_length=1)
    cuda_version: str = Field(min_length=1)


class ChessInferenceBenchmarkReport(FrozenModel):
    schema_version: Literal[2] = 2
    source_revision: SourceRevision
    configuration_path: str = Field(min_length=1)
    configuration_sha256: str = Field(min_length=64, max_length=64)
    policy_semantics: Literal['chess_76_plane_direct_v2_logits'] = 'chess_76_plane_direct_v2_logits'
    hardware: HardwareDescription
    attention_backend: AttentionBackend
    batch_sizes: tuple[int, ...] = Field(min_length=1)
    execution_modes: tuple[ExecutionMode, ...] = Field(min_length=1)
    warmup_iterations: int = Field(ge=0)
    target_duration_seconds: float = Field(gt=0.0)
    includes_diagnostic_controls: bool
    models: tuple[ModelDescription, ...] = Field(min_length=1)
    measurements: tuple[InferenceMeasurement, ...] = Field(min_length=1)


def parse_positive_integer_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('values must be comma-separated integers') from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError('values must be positive integers')
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('values must be unique')
    return values


def parse_execution_modes(value: str) -> tuple[ExecutionMode, ...]:
    try:
        modes = tuple(ExecutionMode(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('modes must be eager, scripted, and/or compiled') from error
    if not modes or len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError('modes must be nonempty and unique')
    return modes


def parse_arguments() -> BenchmarkArguments:
    parser = argparse.ArgumentParser(
        description='Benchmark inference for every progressive model in the retained Chess production config.'
    )
    parser.add_argument('--configuration', type=Path, default=DEFAULT_CONFIGURATION_PATH)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-id', type=int, required=True)
    parser.add_argument('--batch-sizes', type=parse_positive_integer_list, default=(1, 16, 64, 256))
    parser.add_argument('--modes', type=parse_execution_modes, default=(ExecutionMode.SCRIPTED,))
    parser.add_argument('--attention-backend', type=AttentionBackend, default=AttentionBackend.AUTOMATIC)
    parser.add_argument('--warmup-iterations', type=int, default=10)
    parser.add_argument('--duration-seconds', type=float, default=5.0)
    parser.add_argument('--include-diagnostic-controls', action='store_true')
    parser.add_argument('--acknowledge-gpu-load', action='store_true')
    parsed = parser.parse_args()
    if parsed.gpu_id < 0:
        parser.error('--gpu-id must be nonnegative')
    if parsed.warmup_iterations < 0:
        parser.error('--warmup-iterations must be nonnegative')
    if parsed.duration_seconds <= 0.0:
        parser.error('--duration-seconds must be positive')
    return BenchmarkArguments(
        configuration_path=parsed.configuration.resolve(),
        output_path=parsed.output.resolve(),
        gpu_id=parsed.gpu_id,
        batch_sizes=parsed.batch_sizes,
        execution_modes=parsed.modes,
        attention_backend=parsed.attention_backend,
        warmup_iterations=parsed.warmup_iterations,
        duration_seconds=parsed.duration_seconds,
        include_diagnostic_controls=parsed.include_diagnostic_controls,
        acknowledge_gpu_load=parsed.acknowledge_gpu_load,
    )


def module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def parameter_counts(network: Network) -> ParameterCounts:
    backbone = (
        module_parameter_count(network.startBlock)
        + module_parameter_count(network.backBone)
        + module_parameter_count(network.finishBlock)
    )
    primary_policy_head = module_parameter_count(network.policyHead)
    value_head = module_parameter_count(network.valueHead)
    auxiliary_heads = module_parameter_count(network.auxiliaryHeads)
    primary_total = backbone + primary_policy_head + value_head
    inference_total = primary_total + _search_correction_parameter_count(network)
    training_total = primary_total + auxiliary_heads
    assert training_total == module_parameter_count(network)
    return ParameterCounts(
        backbone=backbone,
        primary_policy_head=primary_policy_head,
        value_head=value_head,
        auxiliary_heads=auxiliary_heads,
        inference_total=inference_total,
        training_total=training_total,
    )


def _search_correction_parameter_count(network: Network) -> int:
    for head, module in zip(network.auxiliary_heads, network.auxiliaryHeads, strict=True):
        match head:
            case SearchCorrectionHeadLayout():
                return module_parameter_count(module)
    return 0


def _attention_backend_context(backend: AttentionBackend) -> AbstractContextManager[None]:
    match backend:
        case AttentionBackend.AUTOMATIC:
            return nullcontext()
        case AttentionBackend.FLASH:
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION, set_priority=True)
        case AttentionBackend.MEMORY_EFFICIENT:
            return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION, set_priority=True)
        case AttentionBackend.CUDNN:
            return sdpa_kernel(SDPBackend.CUDNN_ATTENTION, set_priority=True)
        case AttentionBackend.MATH:
            return sdpa_kernel(SDPBackend.MATH, set_priority=True)


def _benchmark_models(
    configuration_path: Path,
    include_diagnostic_controls: bool,
) -> tuple[BenchmarkModel, ...]:
    configuration = load_chess_experiment_configuration(configuration_path)
    progressive = configuration.training.progressive_model_sizing
    models = tuple(
        BenchmarkModel(
            model_id=model.model_id,
            training_start_days=float(model.training_start_days),
            network=model.network,
        )
        for model in progressive.models
    )
    if not include_diagnostic_controls:
        return models
    policy_head = Chess76PlaneDirectPolicyHeadConfiguration()
    controls = (
        BenchmarkModel(
            model_id='control-cnn-10x128-direct-policy',
            training_start_days=0.0,
            network=NetworkParams(
                num_layers=10,
                hidden_size=128,
                residual_context=GlobalPoolingResidualContext(
                    placement=ResidualContextPlacement.EVERY_SECOND_BLOCK,
                ),
                policy_head=policy_head,
            ),
        ),
        BenchmarkModel(
            model_id='control-attention-5x256-direct-policy',
            training_start_days=0.0,
            network=AttentionNetworkParams(
                num_layers=5,
                embedding_size=256,
                num_heads=8,
                feedforward_size=512,
                policy_head=policy_head,
            ),
        ),
    )
    return (*controls, *models)


def _prepare_inference_network(network: Network) -> Network:
    network.auxiliaryHeads = nn.ModuleList()
    network.auxiliary_heads = ()
    network.eval()
    network.fuse_model()
    network.to(dtype=torch.bfloat16)
    return network


def _calibrated_iteration_count(
    network: nn.Module,
    states: Tensor,
    device: torch.device,
    duration_seconds: float,
) -> int:
    calibration_iterations = 5
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(calibration_iterations):
        network(states)
    torch.cuda.synchronize(device)
    calibration_seconds = time.perf_counter() - started
    return max(1, math.ceil(duration_seconds * calibration_iterations / calibration_seconds))


def _measure(
    network: nn.Module,
    model_id: str,
    execution_mode: ExecutionMode,
    batch_size: int,
    warmup_iterations: int,
    duration_seconds: float,
    device: torch.device,
) -> InferenceMeasurement:
    states = torch.zeros(
        batch_size,
        CHESS_NETWORK_DIMENSIONS.channels,
        CHESS_NETWORK_DIMENSIONS.rows,
        CHESS_NETWORK_DIMENSIONS.columns,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            network(states)
        measured_iterations = _calibrated_iteration_count(network, states, device, duration_seconds)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(measured_iterations):
            network(states)
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started
    positions = batch_size * measured_iterations
    return InferenceMeasurement(
        model_id=model_id,
        execution_mode=execution_mode,
        batch_size=batch_size,
        measured_iterations=measured_iterations,
        elapsed_seconds=elapsed_seconds,
        mean_batch_seconds=elapsed_seconds / measured_iterations,
        positions_per_second=positions / elapsed_seconds,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def run_benchmark(arguments: BenchmarkArguments) -> ChessInferenceBenchmarkReport:
    if not arguments.acknowledge_gpu_load:
        raise ValueError('Inference benchmark requires --acknowledge-gpu-load.')
    if not torch.cuda.is_available():
        raise ValueError('Inference benchmark requires CUDA.')
    if arguments.gpu_id >= torch.cuda.device_count():
        raise ValueError(f'GPU ID {arguments.gpu_id} is not available.')

    configuration = load_chess_experiment_configuration(arguments.configuration_path)
    target_layout = build_training_target_layout(
        CHESS_NETWORK_DIMENSIONS.actions,
        configuration.chess.objective.auxiliary_targets,
    )
    device = torch.device('cuda', arguments.gpu_id)
    torch.cuda.set_device(device)
    descriptions: list[ModelDescription] = []
    measurements: list[InferenceMeasurement] = []

    with _attention_backend_context(arguments.attention_backend):
        for benchmark_model in _benchmark_models(
            arguments.configuration_path,
            arguments.include_diagnostic_controls,
        ):
            network = Network(
                benchmark_model.network,
                device,
                CHESS_NETWORK_DIMENSIONS,
                target_layout.auxiliary_heads,
            )
            descriptions.append(
                ModelDescription(
                    model_id=benchmark_model.model_id,
                    training_start_days=benchmark_model.training_start_days,
                    network=benchmark_model.network,
                    parameters=parameter_counts(network),
                )
            )
            inference_network = _prepare_inference_network(network)
            for execution_mode in arguments.execution_modes:
                benchmark_network: nn.Module = inference_network
                match execution_mode:
                    case ExecutionMode.EAGER:
                        pass
                    case ExecutionMode.SCRIPTED:
                        benchmark_network = torch.jit.script(inference_network)
                    case ExecutionMode.COMPILED:
                        benchmark_network = torch.compile(inference_network)
                for batch_size in arguments.batch_sizes:
                    measurements.append(
                        _measure(
                            benchmark_network,
                            benchmark_model.model_id,
                            execution_mode,
                            batch_size,
                            arguments.warmup_iterations,
                            arguments.duration_seconds,
                            device,
                        )
                    )
                del benchmark_network
            del inference_network, network
            torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(device)
    return ChessInferenceBenchmarkReport(
        source_revision=read_source_revision(),
        configuration_path=str(arguments.configuration_path),
        configuration_sha256=file_sha256(arguments.configuration_path),
        hardware=HardwareDescription(
            gpu_id=arguments.gpu_id,
            device_name=properties.name,
            compute_capability=(properties.major, properties.minor),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda or 'none',
        ),
        attention_backend=arguments.attention_backend,
        batch_sizes=arguments.batch_sizes,
        execution_modes=arguments.execution_modes,
        warmup_iterations=arguments.warmup_iterations,
        target_duration_seconds=arguments.duration_seconds,
        includes_diagnostic_controls=arguments.include_diagnostic_controls,
        models=tuple(descriptions),
        measurements=tuple(measurements),
    )


def main() -> None:
    arguments = parse_arguments()
    report = run_benchmark(arguments)
    write_text_atomically(arguments.output_path, report.model_dump_json(indent=2) + '\n')
    print(arguments.output_path)


if __name__ == '__main__':
    main()

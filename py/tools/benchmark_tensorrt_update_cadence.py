from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort
import tensorrt as trt
import torch
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.util.atomic_file import write_bytes_atomically, write_text_atomically
from src.util.hashing import file_sha256
from tools.benchmark_tensorrt_inference import (
    BATCH_SIZE,
    TENSORRT_BUILDER_OPTIMIZATION_LEVEL,
    TENSORRT_WORKSPACE_BYTES,
    _measure_runner,
    _TensorRtCudaGraphRunner,
)
from tools.tensorrt_benchmark_metrics import ModelOutputs, measure_fidelity


@dataclass(frozen=True)
class BuildResult:
    engine_path: Path
    build_seconds: float
    timing_cache: bytes


@dataclass(frozen=True)
class RefitStep:
    name: str
    onnx_path: Path
    engine_path: Path
    seconds: float
    missing_weights: tuple[str, ...]


def _parse_onnx(network: trt.INetworkDefinition, parser: trt.OnnxParser, path: Path) -> None:
    if parser.parse_from_file(str(path)):
        return
    errors = tuple(str(parser.get_error(index)) for index in range(parser.num_errors))
    raise ValueError(f'TensorRT ONNX conversion failed for {path}: {" | ".join(errors)}')


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    timing_cache: bytes,
    refittable: bool,
) -> BuildResult:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    _parse_onnx(network, parser, onnx_path)
    configuration = builder.create_builder_config()
    configuration.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, TENSORRT_WORKSPACE_BYTES)
    configuration.builder_optimization_level = TENSORRT_BUILDER_OPTIMIZATION_LEVEL
    configuration.set_flag(trt.BuilderFlag.FP16)
    configuration.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
    if refittable:
        configuration.set_flag(trt.BuilderFlag.REFIT)
    cache = configuration.create_timing_cache(timing_cache)
    if not configuration.set_timing_cache(cache, False):
        raise ValueError('TensorRT rejected the serialized timing cache.')
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, configuration)
    build_seconds = time.perf_counter() - started
    if serialized is None:
        raise ValueError(f'TensorRT failed to build an engine for {onnx_path}.')
    write_bytes_atomically(engine_path, bytes(serialized))
    serialized_cache = bytes(configuration.get_timing_cache().serialize())
    return BuildResult(engine_path=engine_path, build_seconds=build_seconds, timing_cache=serialized_cache)


def _refit_engine_sequence(
    source_engine_path: Path,
    source_onnx_path: Path,
    updated_onnx_path: Path,
    output: Path,
) -> tuple[tuple[str, ...], tuple[RefitStep, ...]]:
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(source_engine_path.read_bytes())
    if engine is None:
        raise ValueError(f'TensorRT failed to deserialize {source_engine_path}.')
    refitter = trt.Refitter(engine, logger)
    refittable_weights = tuple(sorted(refitter.get_all_weights()))
    steps: list[RefitStep] = []
    targets = (
        ('updated-first', updated_onnx_path),
        ('source-return', source_onnx_path),
        ('updated-second', updated_onnx_path),
    )
    for name, onnx_path in targets:
        parser_refitter = trt.OnnxParserRefitter(refitter, logger)
        started = time.perf_counter()
        if not parser_refitter.refit_from_file(str(onnx_path)):
            missing = tuple(sorted(refitter.get_missing_weights()))
            raise ValueError(f'TensorRT ONNX refit failed for {name}; missing weights: {missing}.')
        missing = tuple(sorted(refitter.get_missing_weights()))
        if missing:
            raise ValueError(f'TensorRT ONNX refit left missing weights for {name}: {missing}.')
        if not refitter.refit_cuda_engine():
            missing = tuple(sorted(refitter.get_missing_weights()))
            raise ValueError(f'TensorRT engine refit failed for {name}; missing weights: {missing}.')
        seconds = time.perf_counter() - started
        engine_path = output / f'{name}.engine'
        write_bytes_atomically(engine_path, bytes(engine.serialize()))
        steps.append(
            RefitStep(
                name=name,
                onnx_path=onnx_path,
                engine_path=engine_path,
                seconds=seconds,
                missing_weights=missing,
            )
        )
    return refittable_weights, tuple(steps)


def _onnx_outputs(path: Path, states: torch.Tensor, device_id: int) -> ModelOutputs:
    session = ort.InferenceSession(
        str(path),
        providers=[('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider'],
    )
    policy, value = session.run(None, {'states': states.float().cpu().numpy()})
    return ModelOutputs(torch.from_numpy(policy).float(), torch.from_numpy(value).float())


def _engine_outputs(path: Path, states: torch.Tensor, device: torch.device) -> tuple[ModelOutputs, dict[str, object]]:
    runner = _TensorRtCudaGraphRunner(path, states, device, 10)
    runner.load_states(states)
    outputs = runner.outputs()
    timing = _measure_runner(runner, 10, 5, 100, device)
    return outputs, timing.model_dump(mode='json')


def _error_summary(reference: ModelOutputs, candidate: ModelOutputs) -> dict[str, float]:
    policy_error = (reference.policy_logits - candidate.policy_logits).abs()
    value_error = (reference.wdl_probabilities - candidate.wdl_probabilities).abs()
    return {
        'policy_mean_absolute_error': float(policy_error.mean()),
        'policy_maximum_absolute_error': float(policy_error.max()),
        'wdl_mean_absolute_error': float(value_error.mean()),
        'wdl_maximum_absolute_error': float(value_error.max()),
    }


def _fidelity_summary(reference: ModelOutputs, candidate: ModelOutputs) -> dict[str, object]:
    legal_mask = torch.ones(reference.policy_logits.shape, dtype=torch.bool)
    return measure_fidelity(reference, candidate, legal_mask).model_dump(mode='json')


def main() -> None:
    parser = argparse.ArgumentParser(description='Measure TensorRT timing-cache rebuild and ONNX refit cadence.')
    parser.add_argument('--source-onnx', type=Path, required=True)
    parser.add_argument('--updated-onnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-id', type=int, required=True)
    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda', arguments.device_id)
    generator = torch.Generator(device=device).manual_seed(20260912)
    states = torch.randint(
        0,
        2,
        (
            BATCH_SIZE,
            CHESS_NETWORK_DIMENSIONS.channels,
            CHESS_NETWORK_DIMENSIONS.rows,
            CHESS_NETWORK_DIMENSIONS.columns,
        ),
        generator=generator,
        device=device,
        dtype=torch.int32,
    ).float()

    source = _build_engine(arguments.source_onnx, arguments.output / 'source.engine', b'', False)
    cache_path = arguments.output / 'editable-timing.cache'
    write_bytes_atomically(cache_path, source.timing_cache)
    cached = _build_engine(
        arguments.updated_onnx,
        arguments.output / 'updated-cached.engine',
        source.timing_cache,
        False,
    )
    uncached = _build_engine(arguments.updated_onnx, arguments.output / 'updated-uncached.engine', b'', False)
    refit_source = _build_engine(
        arguments.source_onnx,
        arguments.output / 'source-refittable.engine',
        source.timing_cache,
        True,
    )
    refittable_weights, refit_steps = _refit_engine_sequence(
        refit_source.engine_path,
        arguments.source_onnx,
        arguments.updated_onnx,
        arguments.output,
    )

    reference = _onnx_outputs(arguments.updated_onnx, states, arguments.device_id)
    cached_outputs, cached_timing = _engine_outputs(cached.engine_path, states, device)
    uncached_outputs, uncached_timing = _engine_outputs(uncached.engine_path, states, device)
    first_refitted_outputs, first_refitted_timing = _engine_outputs(
        arguments.output / 'updated-first.engine', states, device
    )
    source_return_reference = _onnx_outputs(arguments.source_onnx, states, arguments.device_id)
    source_return_outputs, source_return_timing = _engine_outputs(
        arguments.output / 'source-return.engine', states, device
    )
    second_refitted_outputs, second_refitted_timing = _engine_outputs(
        arguments.output / 'updated-second.engine', states, device
    )
    report = {
        'source_onnx_sha256': file_sha256(arguments.source_onnx),
        'updated_onnx_sha256': file_sha256(arguments.updated_onnx),
        'timing_cache_sha256': file_sha256(cache_path),
        'timing_cache_bytes': len(source.timing_cache),
        'source_build_seconds': source.build_seconds,
        'updated_cached_build_seconds': cached.build_seconds,
        'updated_uncached_build_seconds': uncached.build_seconds,
        'refittable_source_build_seconds': refit_source.build_seconds,
        'refittable_weight_count': len(refittable_weights),
        'refittable_quantizer_constant_count': sum('quantizer' in name for name in refittable_weights),
        'refittable_weights': refittable_weights,
        'refit_steps': [
            {
                'name': step.name,
                'onnx_sha256': file_sha256(step.onnx_path),
                'engine_sha256': file_sha256(step.engine_path),
                'seconds': step.seconds,
                'missing_weights': step.missing_weights,
            }
            for step in refit_steps
        ],
        'updated_cached_timing': cached_timing,
        'updated_uncached_timing': uncached_timing,
        'updated_first_refitted_timing': first_refitted_timing,
        'source_return_refitted_timing': source_return_timing,
        'updated_second_refitted_timing': second_refitted_timing,
        'updated_cached_error_to_onnx': _error_summary(reference, cached_outputs),
        'updated_uncached_error_to_onnx': _error_summary(reference, uncached_outputs),
        'updated_cached_fidelity_to_onnx': _fidelity_summary(reference, cached_outputs),
        'updated_uncached_fidelity_to_onnx': _fidelity_summary(reference, uncached_outputs),
        'updated_first_refitted_error_to_onnx': _error_summary(reference, first_refitted_outputs),
        'updated_first_refitted_fidelity_to_onnx': _fidelity_summary(reference, first_refitted_outputs),
        'source_return_refitted_error_to_onnx': _error_summary(source_return_reference, source_return_outputs),
        'source_return_refitted_fidelity_to_onnx': _fidelity_summary(source_return_reference, source_return_outputs),
        'updated_second_refitted_error_to_onnx': _error_summary(reference, second_refitted_outputs),
        'updated_second_refitted_fidelity_to_onnx': _fidelity_summary(reference, second_refitted_outputs),
        'updated_first_refitted_error_to_cached': _error_summary(cached_outputs, first_refitted_outputs),
        'updated_second_refitted_error_to_cached': _error_summary(cached_outputs, second_refitted_outputs),
        'artifacts': {
            path.name: file_sha256(path)
            for path in sorted(arguments.output.iterdir())
            if path.is_file() and path.name != 'report.json'
        },
    }
    write_text_atomically(arguments.output / 'report.json', json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

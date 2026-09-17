"""Verify one checkpoint through the exact TensorRT publication and native inference path."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import onnxruntime as ort
import torch
from AlphaZeroCpp import (
    ChessSelfPlaySearch,
    InferenceBackend,
    InferenceConfiguration,
    InferenceDimensions,
    InferenceRunner,
)
from pydantic import Field
from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.training import ChessImplementation
from src.training.checkpoint.contracts import CheckpointReference
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from tools.build_tensorrt_refit_template import build_template
from tools.export_checkpoint_float_inference import export_float_inference
from tools.measure_inference_precision_agreement import load_positions
from tools.publish_tensorrt_engine import export_onnx, export_onnx_with_example
from tools.tensorrt_benchmark_metrics import (
    FidelityLimits,
    FidelityMetrics,
    ModelOutputs,
    fidelity_failures,
    measure_fidelity,
)
from torch import Tensor

DEFAULT_BATCH_SIZES = (1, 64, 241, 320)


@dataclass(frozen=True)
class Arguments:
    configuration_path: Path
    checkpoint_directory: Path
    checkpoint_generation: int
    dataset_path: Path
    artifact_directory: Path
    output_path: Path
    gpu_id: int
    batch_sizes: tuple[int, ...]
    limits: FidelityLimits
    acknowledge_gpu_load: bool


class StageComparison(FrozenModel):
    stage: str = Field(min_length=1)
    metrics: FidelityMetrics
    failures: tuple[str, ...]


class BatchReport(FrozenModel):
    batch_size: int = Field(gt=0)
    comparisons: tuple[StageComparison, ...] = Field(min_length=1)


class ArtifactIdentity(FrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class ProductionEquivalenceReport(FrozenModel):
    schema_version: Literal[1] = 1
    configuration_path: str = Field(min_length=1)
    checkpoint_generation: int = Field(ge=0)
    dataset_path: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    limits: FidelityLimits
    source_inference_artifact: ArtifactIdentity
    reference_torchscript: ArtifactIdentity
    zero_example_onnx: ArtifactIdentity
    real_example_onnx: ArtifactIdentity
    example_export_graphs_match: bool
    zero_vs_real_example_export: FidelityMetrics
    fresh_tensorrt_engine: ArtifactIdentity
    production_refitted_engine: ArtifactIdentity
    batches: tuple[BatchReport, ...] = Field(min_length=1)
    passed: bool


def _artifact(path: Path) -> ArtifactIdentity:
    return ArtifactIdentity(path=str(path), sha256=file_sha256(path))


def _onnx_outputs(path: Path, states: Tensor, gpu_id: int) -> ModelOutputs:
    providers: list[str | tuple[str, dict[str, str]]] = [
        ('CUDAExecutionProvider', {'device_id': str(gpu_id)}),
        'CPUExecutionProvider',
    ]
    session = ort.InferenceSession(str(path), providers=providers)
    input_metadata = session.get_inputs()
    if len(input_metadata) != 1 or input_metadata[0].name != 'states':
        raise ValueError('The deployed ONNX model must have exactly one input named states.')
    match input_metadata[0].type:
        case 'tensor(float)':
            numpy_states = states.cpu().numpy().astype(np.float32, copy=False)
        case 'tensor(float16)':
            numpy_states = states.cpu().numpy().astype(np.float16, copy=False)
        case input_type:
            raise ValueError(f'Unsupported deployed ONNX input type: {input_type}.')
    policy_logits, wdl_probabilities = session.run(
        ('policy_logits', 'wdl_probabilities'),
        {'states': numpy_states},
    )
    return ModelOutputs(
        policy_logits=torch.from_numpy(policy_logits).float(),
        wdl_probabilities=torch.from_numpy(wdl_probabilities).float(),
    )


def _onnx_graph_signature(path: Path) -> tuple[tuple[str, ...], tuple[tuple[str, tuple[int, ...]], ...]]:
    model = onnx.load(path, load_external_data=False)
    operations = tuple(node.op_type for node in model.graph.node)
    initializers = tuple(
        sorted((initializer.name, tuple(initializer.dims)) for initializer in model.graph.initializer)
    )
    return operations, initializers


def _native_outputs(runner: InferenceRunner, states: Tensor, batch_size: int) -> ModelOutputs:
    policy_logits, wdl_probabilities = runner.forward(states[:batch_size].contiguous())
    return ModelOutputs(policy_logits=policy_logits, wdl_probabilities=wdl_probabilities)


def _comparison(
    stage: str,
    reference: ModelOutputs,
    candidate: ModelOutputs,
    legal_action_mask: Tensor,
    limits: FidelityLimits,
) -> StageComparison:
    metrics = measure_fidelity(reference, candidate, legal_action_mask)
    return StageComparison(stage=stage, metrics=metrics, failures=fidelity_failures(metrics, limits))


def _runner(
    model_path: Path,
    backend: InferenceBackend,
    native_configuration: InferenceConfiguration,
    maximum_batch_size: int,
    dimensions: InferenceDimensions,
) -> InferenceRunner:
    return InferenceRunner(
        model_path=str(model_path),
        device=native_configuration.device,
        device_id=native_configuration.device_id,
        maximum_batch_size=maximum_batch_size,
        use_dedicated_cuda_stream=True,
        dimensions=dimensions,
        execution_options=native_configuration.execution_options,
        backend=backend,
    )


def run(arguments: Arguments) -> ProductionEquivalenceReport:
    if not arguments.acknowledge_gpu_load:
        raise ValueError('The production equivalence test requires --acknowledge-gpu-load.')
    if not torch.cuda.is_available() or arguments.gpu_id >= torch.cuda.device_count():
        raise ValueError(f'CUDA device {arguments.gpu_id} is unavailable.')
    if not arguments.batch_sizes or any(batch_size <= 0 for batch_size in arguments.batch_sizes):
        raise ValueError('At least one positive batch size is required.')

    configuration = load_chess_experiment_configuration(arguments.configuration_path)
    game = ChessImplementation(configuration)
    checkpoint = CheckpointReference.load(arguments.checkpoint_directory, arguments.checkpoint_generation)
    if not checkpoint.inference_model_path.name.endswith('.fp16.onnx'):
        raise ValueError('This diagnostic currently requires a floating-point ONNX checkpoint artifact.')

    maximum_batch_size = configuration.chess.self_play.inference.inference_batch_size
    if max(arguments.batch_sizes) > maximum_batch_size:
        raise ValueError(f'Requested batch {max(arguments.batch_sizes)} exceeds production batch {maximum_batch_size}.')
    arguments.artifact_directory.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(arguments.gpu_id)
    states, legal_action_mask = load_positions(arguments.dataset_path, maximum_batch_size)
    if states.dtype != torch.int8:
        raise ValueError(f'Production encoded states must be int8, found {states.dtype}.')

    reference_path = arguments.artifact_directory / f'model-{arguments.checkpoint_generation}.reference.jit.pt'
    export_float_inference(checkpoint.manifest_path, checkpoint.generation, reference_path)
    zero_example_onnx_path = arguments.artifact_directory / f'model-{checkpoint.generation}.zero-example.onnx'
    real_example_onnx_path = arguments.artifact_directory / f'model-{checkpoint.generation}.real-example.onnx'
    input_shape = (
        maximum_batch_size,
        configuration.network_dimensions.channels,
        configuration.network_dimensions.rows,
        configuration.network_dimensions.columns,
    )
    export_onnx(reference_path, zero_example_onnx_path, input_shape)
    export_onnx_with_example(
        reference_path,
        real_example_onnx_path,
        states.to(dtype=torch.float16),
    )

    native_configuration = game.native_inference_configuration(arguments.gpu_id, checkpoint)
    production_engine_path = Path(native_configuration.model_path)
    if native_configuration.backend is not InferenceBackend.TENSORRT:
        raise ValueError('The supplied production configuration did not resolve to TensorRT.')

    dimensions = ChessSelfPlaySearch.inference_dimensions()
    fresh_engine_path = arguments.artifact_directory / f'model-{checkpoint.generation}.fresh.engine'
    build_template(
        checkpoint.inference_model_path,
        fresh_engine_path,
        maximum_batch_size,
        dimensions.channels,
        dimensions.rows,
        dimensions.columns,
        5,
        None,
    )

    onnx_outputs = _onnx_outputs(checkpoint.inference_model_path, states, arguments.gpu_id)
    zero_example_outputs = _onnx_outputs(zero_example_onnx_path, states, arguments.gpu_id)
    real_example_outputs = _onnx_outputs(real_example_onnx_path, states, arguments.gpu_id)
    reference_runner = _runner(
        reference_path,
        InferenceBackend.TORCHSCRIPT,
        native_configuration,
        maximum_batch_size,
        dimensions,
    )
    fresh_runner = _runner(
        fresh_engine_path,
        InferenceBackend.TENSORRT,
        native_configuration,
        maximum_batch_size,
        dimensions,
    )
    production_runner = _runner(
        production_engine_path,
        InferenceBackend.TENSORRT,
        native_configuration,
        maximum_batch_size,
        dimensions,
    )

    batch_reports: list[BatchReport] = []
    for batch_size in arguments.batch_sizes:
        mask = legal_action_mask[:batch_size]
        reference_outputs = _native_outputs(reference_runner, states, batch_size)
        fresh_outputs = _native_outputs(fresh_runner, states, batch_size)
        production_outputs = _native_outputs(production_runner, states, batch_size)
        batch_reports.append(
            BatchReport(
                batch_size=batch_size,
                comparisons=(
                    _comparison(
                        'pytorch_torchscript_vs_checkpoint_onnx',
                        reference_outputs,
                        ModelOutputs(
                            onnx_outputs.policy_logits[:batch_size],
                            onnx_outputs.wdl_probabilities[:batch_size],
                        ),
                        mask,
                        arguments.limits,
                    ),
                    _comparison(
                        'checkpoint_onnx_vs_fresh_tensorrt_native',
                        ModelOutputs(
                            onnx_outputs.policy_logits[:batch_size],
                            onnx_outputs.wdl_probabilities[:batch_size],
                        ),
                        fresh_outputs,
                        mask,
                        arguments.limits,
                    ),
                    _comparison(
                        'fresh_tensorrt_vs_production_refit_native',
                        fresh_outputs,
                        production_outputs,
                        mask,
                        arguments.limits,
                    ),
                    _comparison(
                        'pytorch_torchscript_vs_production_refit_native',
                        reference_outputs,
                        production_outputs,
                        mask,
                        arguments.limits,
                    ),
                ),
            )
        )

    report = ProductionEquivalenceReport(
        configuration_path=str(arguments.configuration_path),
        checkpoint_generation=checkpoint.generation,
        dataset_path=str(arguments.dataset_path),
        device_name=torch.cuda.get_device_properties(arguments.gpu_id).name,
        limits=arguments.limits,
        source_inference_artifact=_artifact(checkpoint.inference_model_path),
        reference_torchscript=_artifact(reference_path),
        zero_example_onnx=_artifact(zero_example_onnx_path),
        real_example_onnx=_artifact(real_example_onnx_path),
        example_export_graphs_match=(
            _onnx_graph_signature(zero_example_onnx_path) == _onnx_graph_signature(real_example_onnx_path)
        ),
        zero_vs_real_example_export=measure_fidelity(
            zero_example_outputs,
            real_example_outputs,
            legal_action_mask,
        ),
        fresh_tensorrt_engine=_artifact(fresh_engine_path),
        production_refitted_engine=_artifact(production_engine_path),
        batches=tuple(batch_reports),
        passed=(
            _onnx_graph_signature(zero_example_onnx_path) == _onnx_graph_signature(real_example_onnx_path)
            and not fidelity_failures(
                measure_fidelity(zero_example_outputs, real_example_outputs, legal_action_mask),
                arguments.limits,
            )
            and all(not comparison.failures for batch in batch_reports for comparison in batch.comparisons)
        ),
    )
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(arguments.output_path, report.model_dump_json(indent=2) + '\n')
    return report


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', required=True, type=Path)
    parser.add_argument('--checkpoint-directory', required=True, type=Path)
    parser.add_argument('--checkpoint-generation', required=True, type=int)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--artifact-directory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', action='append', type=int)
    parser.add_argument('--minimum-policy-top1-agreement', type=float, default=0.98)
    parser.add_argument('--maximum-mean-policy-kl-divergence', type=float, default=0.005)
    parser.add_argument('--maximum-wdl-mean-absolute-error', type=float, default=0.01)
    parser.add_argument('--maximum-expected-value-mean-absolute-error', type=float, default=0.015)
    parser.add_argument('--acknowledge-gpu-load', action='store_true')
    parsed = parser.parse_args()
    return Arguments(
        configuration_path=parsed.configuration.resolve(),
        checkpoint_directory=parsed.checkpoint_directory.resolve(),
        checkpoint_generation=parsed.checkpoint_generation,
        dataset_path=parsed.dataset.resolve(),
        artifact_directory=parsed.artifact_directory.resolve(),
        output_path=parsed.output.resolve(),
        gpu_id=parsed.gpu_id,
        batch_sizes=tuple(parsed.batch_size or DEFAULT_BATCH_SIZES),
        limits=FidelityLimits(
            minimum_policy_top1_agreement=parsed.minimum_policy_top1_agreement,
            maximum_mean_policy_kl_divergence=parsed.maximum_mean_policy_kl_divergence,
            maximum_wdl_mean_absolute_error=parsed.maximum_wdl_mean_absolute_error,
            maximum_expected_value_mean_absolute_error=parsed.maximum_expected_value_mean_absolute_error,
        ),
        acknowledge_gpu_load=parsed.acknowledge_gpu_load,
    )


def main() -> None:
    report = run(parse_arguments())
    print(json.dumps(report.model_dump(mode='json'), indent=2))
    if not report.passed:
        raise ValueError('TensorRT production equivalence failed; inspect the written stage report.')


if __name__ == '__main__':
    main()

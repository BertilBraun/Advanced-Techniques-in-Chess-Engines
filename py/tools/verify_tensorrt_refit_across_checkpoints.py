"""Verify that a TensorRT template can be refitted across two checkpoint generations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from AlphaZeroCpp import ChessSelfPlaySearch, InferenceBackend
from pydantic import Field
from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.training import ChessImplementation
from src.training.checkpoint.contracts import CheckpointReference
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from tools.build_tensorrt_refit_template import RefitMode, build_template
from tools.measure_inference_precision_agreement import load_positions
from tools.publish_tensorrt_engine import refit_engine
from tools.tensorrt_benchmark_metrics import FidelityLimits, ModelOutputs
from tools.verify_tensorrt_production_equivalence import (
    ArtifactIdentity,
    BatchReport,
    _artifact,
    _comparison,
    _native_outputs,
    _onnx_graph_signature,
    _onnx_outputs,
    _runner,
)

DEFAULT_BATCH_SIZES = (1, 64, 241, 320)


@dataclass(frozen=True)
class Arguments:
    configuration_path: Path
    checkpoint_directory: Path
    template_generation: int
    candidate_generation: int
    dataset_path: Path
    artifact_directory: Path
    output_path: Path
    gpu_id: int
    batch_sizes: tuple[int, ...]
    limits: FidelityLimits
    acknowledge_gpu_load: bool


class RefitAcrossCheckpointsReport(FrozenModel):
    schema_version: Literal[1] = 1
    template_generation: int = Field(ge=0)
    candidate_generation: int = Field(ge=0)
    refit_mode: Literal['all'] = 'all'
    graph_signatures_match: bool
    template_onnx: ArtifactIdentity
    candidate_onnx: ArtifactIdentity
    refittable_template_engine: ArtifactIdentity
    refitted_candidate_engine: ArtifactIdentity
    batches: tuple[BatchReport, ...] = Field(min_length=1)
    passed: bool


def run(arguments: Arguments) -> RefitAcrossCheckpointsReport:
    if not arguments.acknowledge_gpu_load:
        raise ValueError('The refit equivalence test requires --acknowledge-gpu-load.')
    if not torch.cuda.is_available() or arguments.gpu_id >= torch.cuda.device_count():
        raise ValueError(f'CUDA device {arguments.gpu_id} is unavailable.')
    if arguments.template_generation >= arguments.candidate_generation:
        raise ValueError('The template generation must precede the candidate generation.')

    configuration = load_chess_experiment_configuration(arguments.configuration_path)
    maximum_batch_size = configuration.chess.self_play.inference.inference_batch_size
    if max(arguments.batch_sizes) > maximum_batch_size:
        raise ValueError(f'Requested batch {max(arguments.batch_sizes)} exceeds production batch {maximum_batch_size}.')
    template_checkpoint = CheckpointReference.load(
        arguments.checkpoint_directory,
        arguments.template_generation,
    )
    candidate_checkpoint = CheckpointReference.load(
        arguments.checkpoint_directory,
        arguments.candidate_generation,
    )
    states, legal_action_mask = load_positions(arguments.dataset_path, maximum_batch_size)
    arguments.artifact_directory.mkdir(parents=True, exist_ok=True)
    template_onnx_path = template_checkpoint.inference_model_path
    candidate_onnx_path = candidate_checkpoint.inference_model_path

    graph_signatures_match = _onnx_graph_signature(template_onnx_path) == _onnx_graph_signature(candidate_onnx_path)
    dimensions = ChessSelfPlaySearch.inference_dimensions()
    template_engine_path = arguments.artifact_directory / 'full-refit-template.engine'
    refitted_engine_path = arguments.artifact_directory / f'model-{arguments.candidate_generation}.full-refit.engine'
    build_template(
        template_onnx_path,
        template_engine_path,
        maximum_batch_size,
        dimensions.channels,
        dimensions.rows,
        dimensions.columns,
        5,
        None,
        RefitMode.ALL,
    )
    refit_engine(template_engine_path, candidate_onnx_path, refitted_engine_path)

    game = ChessImplementation(configuration)
    native_configuration = game.native_inference_configuration(arguments.gpu_id, candidate_checkpoint)
    runner = _runner(
        refitted_engine_path,
        InferenceBackend.TENSORRT,
        native_configuration,
        maximum_batch_size,
        dimensions,
    )
    candidate_outputs = _onnx_outputs(candidate_onnx_path, states, arguments.gpu_id)
    batch_reports: list[BatchReport] = []
    for batch_size in arguments.batch_sizes:
        mask = legal_action_mask[:batch_size]
        refitted_outputs = _native_outputs(runner, states.to(dtype=torch.int8), batch_size)
        batch_reports.append(
            BatchReport(
                batch_size=batch_size,
                comparisons=(
                    _comparison(
                        'checkpoint_onnx_vs_full_refit_tensorrt_native',
                        ModelOutputs(
                            candidate_outputs.policy_logits[:batch_size],
                            candidate_outputs.wdl_probabilities[:batch_size],
                        ),
                        refitted_outputs,
                        mask,
                        arguments.limits,
                    ),
                ),
            )
        )

    report = RefitAcrossCheckpointsReport(
        template_generation=arguments.template_generation,
        candidate_generation=arguments.candidate_generation,
        graph_signatures_match=graph_signatures_match,
        template_onnx=_artifact(template_onnx_path),
        candidate_onnx=_artifact(candidate_onnx_path),
        refittable_template_engine=_artifact(template_engine_path),
        refitted_candidate_engine=_artifact(refitted_engine_path),
        batches=tuple(batch_reports),
        passed=all(not comparison.failures for batch in batch_reports for comparison in batch.comparisons),
    )
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(arguments.output_path, report.model_dump_json(indent=2) + '\n')
    return report


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', required=True, type=Path)
    parser.add_argument('--checkpoint-directory', required=True, type=Path)
    parser.add_argument('--template-generation', required=True, type=int)
    parser.add_argument('--candidate-generation', required=True, type=int)
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
        template_generation=parsed.template_generation,
        candidate_generation=parsed.candidate_generation,
        dataset_path=parsed.dataset.resolve(),
        artifact_directory=parsed.artifact_directory.resolve(),
        output_path=parsed.output.resolve(),
        gpu_id=parsed.gpu_id,
        batch_sizes=tuple(DEFAULT_BATCH_SIZES if parsed.batch_size is None else parsed.batch_size),
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
    print(report.model_dump_json(indent=2))
    if not report.passed:
        raise ValueError('TensorRT cross-checkpoint refit failed; inspect the written stage report.')


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import Field
from src.experiment.configuration import load_experiment_configuration
from src.training.checkpoint.contracts import CheckpointManifest, CheckpointReference, load_checkpoint_manifest
from src.training.checkpoint.paths import checkpoint_manifest_path
from src.training.checkpoint.persistence import load_model_state_dict
from src.training.network import Network
from src.training.quantization.configuration import TensorRtInt8QatConfiguration
from src.training.quantization.runtime import restore_qat_model
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from tools.benchmark_tensorrt_inference import _TensorRtCudaGraphRunner
from tools.distill_train_student import (
    OpenedProductionReplay,
    ProductionReplayInput,
    close_training_dataset,
    dataset_split,
    open_training_dataset,
)
from tools.run_int8_architecture_screen import (
    _held_out_batches,
    _legal_action_mask,
    _model_outputs,
    _onnx_outputs,
    _tensorrt_outputs,
)
from tools.tensorrt_benchmark_metrics import FidelityMetrics, measure_fidelity


@dataclass(frozen=True)
class Arguments:
    run_configuration: Path
    run_directory: Path
    generation: int
    replay_store: Path
    replay_experiment: Path
    replay_sha256: str
    engine: Path
    output: Path
    device_id: int
    positions: int
    batch_size: int
    holdout_fraction: float
    onnx_only: bool


class DeploymentFidelityReport(FrozenModel):
    schema_version: int = 1
    source_checkpoint_manifest: str = Field(min_length=1)
    checkpoint_model_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    checkpoint_onnx_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    production_engine_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    replay_store_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    positions: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    framework_to_onnx: FidelityMetrics | None
    onnx_to_production_engine: FidelityMetrics
    framework_to_production_engine: FidelityMetrics | None


def run(arguments: Arguments) -> DeploymentFidelityReport:
    configuration = load_experiment_configuration(arguments.run_configuration)
    quantization = configuration.training.trainer.quantization
    if not isinstance(quantization, TensorRtInt8QatConfiguration):
        raise ValueError('Deployment fidelity audit requires a QAT experiment.')
    device = torch.device('cuda', arguments.device_id)
    torch.cuda.set_device(device)
    if arguments.onnx_only:
        manifest_path = checkpoint_manifest_path(arguments.generation, arguments.run_directory)
        manifest = CheckpointManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
        checkpoint_onnx_path = arguments.run_directory / manifest.inference_model_path
        model = None
    else:
        manifest = load_checkpoint_manifest(arguments.generation, arguments.run_directory)
        checkpoint = CheckpointReference.from_manifest(arguments.run_directory, manifest)
        if checkpoint.qat_state is None:
            raise ValueError('Deployment fidelity audit requires QAT checkpoint state.')
        checkpoint_onnx_path = checkpoint.inference_model_path
        model = Network(
            manifest.network.architecture,
            device,
            manifest.network.dimensions,
            manifest.network.auxiliary_heads,
        )
        model = restore_qat_model(model, checkpoint.qat_state, quantization).model
        weights = torch.load(checkpoint.model_path, map_location=device, weights_only=True)
        load_model_state_dict(model, weights, checkpoint.model_path)

    dataset_input = ProductionReplayInput(
        kind='production_replay',
        path=arguments.replay_store,
        experiment=arguments.replay_experiment,
        orchestrator_recorded_sha256=arguments.replay_sha256,
    )
    opened = open_training_dataset(dataset_input)
    if not isinstance(opened, OpenedProductionReplay):
        raise AssertionError('Deployment fidelity audit requires a production replay store.')
    try:
        split = dataset_split(opened.row_count, arguments.holdout_fraction, 1.0)
        if arguments.positions > split.held_out_row_count:
            raise ValueError('Requested positions exceed the untouched replay holdout.')
        batches = _held_out_batches(
            opened,
            split.held_out_start_row,
            arguments.positions,
            arguments.batch_size,
            device,
        )
        legal_mask = torch.cat(tuple(_legal_action_mask(batch.policy_legal_action_ids) for batch in batches))
        framework_outputs = None if model is None else _model_outputs(model, batches, device)[0]
        onnx_outputs = _onnx_outputs(checkpoint_onnx_path, batches, arguments.device_id)
        runner = _TensorRtCudaGraphRunner(arguments.engine, batches[0].states, device, 10)
        engine_outputs = _tensorrt_outputs(runner, batches)
        report = DeploymentFidelityReport(
            source_checkpoint_manifest=str(checkpoint_manifest_path(arguments.generation, arguments.run_directory)),
            checkpoint_model_sha256=manifest.model_sha256,
            checkpoint_onnx_sha256=manifest.inference_model_sha256,
            production_engine_sha256=file_sha256(arguments.engine),
            replay_store_sha256=arguments.replay_sha256,
            positions=arguments.positions,
            batch_size=arguments.batch_size,
            framework_to_onnx=(
                None if framework_outputs is None else measure_fidelity(framework_outputs, onnx_outputs, legal_mask)
            ),
            onnx_to_production_engine=measure_fidelity(onnx_outputs, engine_outputs, legal_mask),
            framework_to_production_engine=(
                None if framework_outputs is None else measure_fidelity(framework_outputs, engine_outputs, legal_mask)
            ),
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomically(arguments.output, report.model_dump_json(indent=2) + '\n')
        return report
    finally:
        close_training_dataset(opened)


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Audit a deployed QAT checkpoint against its production engine.')
    parser.add_argument('--run-configuration', required=True, type=Path)
    parser.add_argument('--run-directory', required=True, type=Path)
    parser.add_argument('--generation', required=True, type=int)
    parser.add_argument('--replay-store', required=True, type=Path)
    parser.add_argument('--replay-experiment', required=True, type=Path)
    parser.add_argument('--replay-sha256', required=True)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--positions', default=3_200, type=int)
    parser.add_argument('--batch-size', default=320, type=int)
    parser.add_argument('--holdout-fraction', default=0.02, type=float)
    parser.add_argument('--onnx-only', action='store_true')
    namespace = parser.parse_args()
    if namespace.generation <= 0 or namespace.device_id < 0:
        raise ValueError('Generation must be positive and device ID must be nonnegative.')
    if namespace.positions <= 0 or namespace.batch_size <= 0 or namespace.positions % namespace.batch_size:
        raise ValueError('Positions must be a positive multiple of batch size.')
    return Arguments(
        run_configuration=namespace.run_configuration,
        run_directory=namespace.run_directory,
        generation=namespace.generation,
        replay_store=namespace.replay_store,
        replay_experiment=namespace.replay_experiment,
        replay_sha256=namespace.replay_sha256,
        engine=namespace.engine,
        output=namespace.output,
        device_id=namespace.device_id,
        positions=namespace.positions,
        batch_size=namespace.batch_size,
        holdout_fraction=namespace.holdout_fraction,
        onnx_only=namespace.onnx_only,
    )


def main() -> None:
    print(run(parse_arguments()).model_dump_json(indent=2))


if __name__ == '__main__':
    main()

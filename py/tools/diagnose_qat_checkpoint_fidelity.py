from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort
import torch
from pydantic import Field
from src.experiment.configuration import load_chess_experiment_configuration
from src.training.checkpoint.contracts import CheckpointReference, read_checkpoint_manifest
from src.training.checkpoint.persistence import load_model_state_dict
from src.training.network import Network
from src.training.quantization.configuration import TensorRtInt8QatConfiguration
from src.training.quantization.runtime import export_qat_onnx, quantizers_disabled, restore_qat_model
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from tools.benchmark_tensorrt_inference import _TensorRtCudaGraphRunner
from tools.measure_inference_precision_agreement import load_positions
from tools.publish_tensorrt_engine import publish
from tools.tensorrt_benchmark_metrics import FidelityMetrics, ModelOutputs, measure_fidelity
from torch import Tensor


@dataclass(frozen=True)
class Arguments:
    configuration: Path
    run_directory: Path
    dataset: Path
    output_directory: Path
    output: Path
    generations: tuple[int, ...]
    pre_fold_template: Path
    deployment_template: Path
    positions: int
    device_id: int


class OutputSummary(FrozenModel):
    mean_legal_policy_entropy: float = Field(ge=0.0)
    mean_legal_policy_entropy_ratio: float = Field(ge=0.0, le=1.0)
    mean_legal_top1_mass: float = Field(ge=0.0, le=1.0)
    mean_legal_top3_mass: float = Field(ge=0.0, le=1.0)
    mean_expected_value: float = Field(ge=-1.0, le=1.0)
    expected_value_standard_deviation: float = Field(ge=0.0)
    mean_wdl_probabilities: tuple[float, float, float]


class BackendReport(FrozenModel):
    name: str = Field(min_length=1)
    output: OutputSummary


class GenerationReport(FrozenModel):
    generation: int = Field(ge=0)
    qat_phase: str = Field(min_length=1)
    completed_optimizer_steps: int = Field(ge=0)
    model_artifact_available: bool
    original_onnx_artifact_available: bool
    outputs: tuple[BackendReport, ...]
    float_vs_fake_quant: FidelityMetrics | None
    fake_quant_vs_onnx: FidelityMetrics | None
    onnx_vs_tensorrt: FidelityMetrics
    float_vs_onnx: FidelityMetrics | None
    float_vs_tensorrt: FidelityMetrics | None
    diagnostic_onnx_path: str = Field(min_length=1)
    diagnostic_engine_path: str = Field(min_length=1)


class DiagnosticReport(FrozenModel):
    schema_version: int = 1
    configuration: str = Field(min_length=1)
    run_directory: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    positions: int = Field(gt=0)
    device_id: int = Field(ge=0)
    generations: tuple[GenerationReport, ...]


def _framework_outputs(model: Network, states: Tensor) -> ModelOutputs:
    with torch.inference_mode():
        policy_logits, wdl_probabilities = model(states)
    return ModelOutputs(policy_logits.float().cpu(), wdl_probabilities.float().cpu())


def _onnx_outputs(path: Path, states: Tensor) -> ModelOutputs:
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    policy_logits, wdl_probabilities = session.run(None, {'states': states.float().cpu().numpy()})
    return ModelOutputs(torch.from_numpy(policy_logits).float(), torch.from_numpy(wdl_probabilities).float())


def _output_summary(outputs: ModelOutputs, legal_action_mask: Tensor) -> OutputSummary:
    masked_logits = outputs.policy_logits.to(torch.float64).masked_fill(~legal_action_mask, float('-inf'))
    legal_log_probabilities = torch.log_softmax(masked_logits, dim=1)
    legal_probabilities = legal_log_probabilities.exp()
    entropy = -(legal_probabilities * legal_log_probabilities.nan_to_num()).sum(dim=1)
    legal_counts = legal_action_mask.sum(dim=1).to(torch.float64)
    entropy_ratio = entropy / legal_counts.log()
    top_probabilities = legal_probabilities.topk(3, dim=1).values
    expected_values = outputs.wdl_probabilities[:, 0] - outputs.wdl_probabilities[:, 2]
    mean_wdl = outputs.wdl_probabilities.to(torch.float64).mean(dim=0)
    return OutputSummary(
        mean_legal_policy_entropy=float(entropy.mean()),
        mean_legal_policy_entropy_ratio=float(entropy_ratio.mean()),
        mean_legal_top1_mass=float(top_probabilities[:, 0].mean()),
        mean_legal_top3_mass=float(top_probabilities.sum(dim=1).mean()),
        mean_expected_value=float(expected_values.to(torch.float64).mean()),
        expected_value_standard_deviation=float(expected_values.to(torch.float64).std()),
        mean_wdl_probabilities=(float(mean_wdl[0]), float(mean_wdl[1]), float(mean_wdl[2])),
    )


def _copy_onnx(source: Path, output_directory: Path, generation: int) -> Path:
    destination = output_directory / f'generation-{generation}.int8.onnx'
    shutil.copy2(source, destination)
    return destination


def _generation_report(
    arguments: Arguments, states: Tensor, legal_action_mask: Tensor, generation: int
) -> GenerationReport:
    configuration = load_chess_experiment_configuration(arguments.configuration)
    quantization = configuration.training.trainer.quantization
    if not isinstance(quantization, TensorRtInt8QatConfiguration):
        raise ValueError('The checkpoint fidelity diagnostic requires TensorRT INT8 QAT configuration.')
    manifest = read_checkpoint_manifest(generation, arguments.run_directory)
    checkpoint = CheckpointReference.from_manifest(arguments.run_directory, manifest)
    if checkpoint.qat_state is None:
        raise ValueError(f'Generation {generation} is not a QAT checkpoint.')
    device = torch.device('cuda', arguments.device_id)
    model_artifact_available = checkpoint.model_path.is_file()
    original_onnx_artifact_available = checkpoint.inference_model_path.is_file()
    float_outputs: ModelOutputs | None = None
    fake_quant_outputs: ModelOutputs | None = None
    restored: Network | None = None
    if model_artifact_available:
        if file_sha256(checkpoint.model_path) != manifest.model_sha256:
            raise ValueError(f'Checkpoint model hash does not match: {checkpoint.model_path}')
        model = Network(
            manifest.network.architecture,
            device,
            manifest.network.dimensions,
            manifest.network.auxiliary_heads,
        )
        restored = restore_qat_model(model, checkpoint.qat_state, quantization).model
        state_dict: dict[str, Tensor] = torch.load(checkpoint.model_path, map_location=device, weights_only=True)
        load_model_state_dict(restored, state_dict, checkpoint.model_path)
        restored.eval()
        device_states = states.to(device=device, dtype=torch.float32)
        fake_quant_outputs = _framework_outputs(restored, device_states)
        with quantizers_disabled(restored):
            float_outputs = _framework_outputs(restored, device_states)

    diagnostic_onnx = arguments.output_directory / f'generation-{generation}.int8.onnx'
    if original_onnx_artifact_available:
        if file_sha256(checkpoint.inference_model_path) != manifest.inference_model_sha256:
            raise ValueError(f'Checkpoint ONNX hash does not match: {checkpoint.inference_model_path}')
        diagnostic_onnx = _copy_onnx(checkpoint.inference_model_path, arguments.output_directory, generation)
    elif restored is not None:
        export_qat_onnx(restored, diagnostic_onnx, states.to(device=device, dtype=torch.float32))
    else:
        raise ValueError(f'Generation {generation} retains neither its model nor its ONNX artifact.')
    onnx_outputs = _onnx_outputs(diagnostic_onnx, states)
    template = (
        arguments.pre_fold_template if checkpoint.qat_state.phase.value == 'pre_fold' else arguments.deployment_template
    )
    published = publish(diagnostic_onnx, (template,))
    engine_path = Path(str(published['engine_path']))
    runner = _TensorRtCudaGraphRunner(engine_path, states.to(device=device, dtype=torch.int8), device, 2)
    tensorrt_outputs = runner.outputs()

    named_outputs: list[tuple[str, ModelOutputs]] = []
    if float_outputs is not None:
        named_outputs.append(('float_quantizers_disabled', float_outputs))
    if fake_quant_outputs is not None:
        named_outputs.append(('pytorch_fake_quant', fake_quant_outputs))
    named_outputs.extend((('onnx_qdq', onnx_outputs), ('tensorrt_int8', tensorrt_outputs)))
    return GenerationReport(
        generation=generation,
        qat_phase=checkpoint.qat_state.phase.value,
        completed_optimizer_steps=checkpoint.qat_state.completed_optimizer_steps,
        model_artifact_available=model_artifact_available,
        original_onnx_artifact_available=original_onnx_artifact_available,
        outputs=tuple(
            BackendReport(name=name, output=_output_summary(outputs, legal_action_mask))
            for name, outputs in named_outputs.items()
        ),
        float_vs_fake_quant=(
            measure_fidelity(float_outputs, fake_quant_outputs, legal_action_mask)
            if float_outputs is not None and fake_quant_outputs is not None
            else None
        ),
        fake_quant_vs_onnx=(
            measure_fidelity(fake_quant_outputs, onnx_outputs, legal_action_mask)
            if fake_quant_outputs is not None
            else None
        ),
        onnx_vs_tensorrt=measure_fidelity(onnx_outputs, tensorrt_outputs, legal_action_mask),
        float_vs_onnx=(
            measure_fidelity(float_outputs, onnx_outputs, legal_action_mask) if float_outputs is not None else None
        ),
        float_vs_tensorrt=(
            measure_fidelity(float_outputs, tensorrt_outputs, legal_action_mask) if float_outputs is not None else None
        ),
        diagnostic_onnx_path=str(diagnostic_onnx),
        diagnostic_engine_path=str(engine_path),
    )


def run(arguments: Arguments) -> DiagnosticReport:
    if arguments.positions <= 0:
        raise ValueError('Positions must be positive.')
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(arguments.device_id)
    states, legal_action_mask = load_positions(arguments.dataset, arguments.positions)
    reports = tuple(
        _generation_report(arguments, states, legal_action_mask, generation) for generation in arguments.generations
    )
    report = DiagnosticReport(
        configuration=str(arguments.configuration),
        run_directory=str(arguments.run_directory),
        dataset=str(arguments.dataset),
        positions=arguments.positions,
        device_id=arguments.device_id,
        generations=reports,
    )
    write_text_atomically(arguments.output, report.model_dump_json(indent=2) + '\n')
    return report


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Compare one QAT checkpoint across framework, ONNX, and TensorRT.')
    parser.add_argument('--configuration', required=True, type=Path)
    parser.add_argument('--run-directory', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--generation', required=True, action='append', type=int)
    parser.add_argument('--pre-fold-template', required=True, type=Path)
    parser.add_argument('--deployment-template', required=True, type=Path)
    parser.add_argument('--positions', default=320, type=int)
    parser.add_argument('--device-id', default=7, type=int)
    namespace = parser.parse_args()
    return Arguments(
        configuration=namespace.configuration,
        run_directory=namespace.run_directory,
        dataset=namespace.dataset,
        output_directory=namespace.output_directory,
        output=namespace.output,
        generations=tuple(namespace.generation),
        pre_fold_template=namespace.pre_fold_template,
        deployment_template=namespace.deployment_template,
        positions=namespace.positions,
        device_id=namespace.device_id,
    )


def main() -> None:
    print(run(parse_arguments()).model_dump_json(indent=2))


if __name__ == '__main__':
    main()

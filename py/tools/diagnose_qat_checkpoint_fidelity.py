from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import onnxruntime as ort
import tensorrt as trt
import torch
from modelopt.torch.quantization.nn import TensorQuantizer
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
from tools.publish_tensorrt_engine import (
    _build_automatic_template,
    engine_input_shape,
    exclusive_lock,
    onnx_graph_signature,
    refit_engine,
)
from tools.tensorrt_benchmark_metrics import FidelityMetrics, ModelOutputs, measure_fidelity
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


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


class FiniteOutputSummary(FrozenModel):
    kind: Literal['finite'] = 'finite'
    policy_absolute_maximum: float = Field(ge=0.0)
    wdl_absolute_maximum: float = Field(ge=0.0)
    mean_legal_policy_entropy: float = Field(ge=0.0)
    mean_legal_policy_entropy_ratio: float = Field(ge=0.0, le=1.0)
    mean_legal_top1_mass: float = Field(ge=0.0, le=1.0)
    mean_legal_top3_mass: float = Field(ge=0.0, le=1.0)
    mean_expected_value: float = Field(ge=-1.0, le=1.0)
    expected_value_standard_deviation: float = Field(ge=0.0)
    mean_wdl_probabilities: tuple[float, float, float]


class NonFiniteOutputSummary(FrozenModel):
    kind: Literal['nonfinite'] = 'nonfinite'
    policy_nonfinite_values: int = Field(ge=0)
    wdl_nonfinite_values: int = Field(ge=0)
    policy_finite_absolute_maximum: float = Field(ge=0.0)
    wdl_finite_absolute_maximum: float = Field(ge=0.0)


OutputSummary = Annotated[FiniteOutputSummary | NonFiniteOutputSummary, Field(discriminator='kind')]


class SuccessfulFidelityComparison(FrozenModel):
    kind: Literal['success'] = 'success'
    metrics: FidelityMetrics


class FailedFidelityComparison(FrozenModel):
    kind: Literal['failed'] = 'failed'
    reason: str = Field(min_length=1)


FidelityComparison = Annotated[
    SuccessfulFidelityComparison | FailedFidelityComparison,
    Field(discriminator='kind'),
]


class BackendReport(FrozenModel):
    name: str = Field(min_length=1)
    output: OutputSummary


class QuantizerGroupSummary(FrozenModel):
    quantizer_count: int = Field(gt=0)
    amax_minimum: float = Field(ge=0.0)
    amax_mean: float = Field(ge=0.0)
    amax_maximum: float = Field(ge=0.0)
    observed_absolute_maximum: float = Field(ge=0.0)
    saturation_rate: float = Field(ge=0.0, le=1.0)


class QuantizerReport(FrozenModel):
    activation: QuantizerGroupSummary
    weight: QuantizerGroupSummary
    dynamic_quantizer_names: tuple[str, ...]


class GenerationReport(FrozenModel):
    generation: int = Field(ge=0)
    qat_phase: str = Field(min_length=1)
    completed_optimizer_steps: int = Field(ge=0)
    model_artifact_available: bool
    original_onnx_artifact_available: bool
    outputs: tuple[BackendReport, ...]
    quantizers: QuantizerReport | None
    float_vs_fake_quant: FidelityComparison | None
    float_vs_onnx: FidelityComparison | None
    onnx_vs_tensorrt_fp16: FidelityComparison
    fake_quant_vs_qdq_onnx: FidelityComparison | None
    qdq_onnx_vs_tensorrt_int8: FidelityComparison | None
    float_onnx_path: str = Field(min_length=1)
    tensorrt_fp16_engine_path: str = Field(min_length=1)
    qdq_onnx_path: str | None
    tensorrt_int8_engine_path: str | None


class DiagnosticReport(FrozenModel):
    schema_version: Literal[2] = 2
    configuration: str = Field(min_length=1)
    run_directory: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    positions: int = Field(gt=0)
    device_id: int = Field(ge=0)
    generations: tuple[GenerationReport, ...]


@dataclass
class _QuantizerObservation:
    name: str
    kind: Literal['activation', 'weight']
    amax: Tensor
    observed_absolute_maximum: float = 0.0
    saturated_values: int = 0
    total_values: int = 0

    def observe(self, values: Tensor) -> None:
        absolute_values = values.detach().abs()
        broadcast_values, broadcast_amax = torch.broadcast_tensors(absolute_values, self.amax)
        self.observed_absolute_maximum = max(self.observed_absolute_maximum, float(absolute_values.max()))
        self.saturated_values += int((broadcast_values >= broadcast_amax).sum())
        self.total_values += broadcast_values.numel()


class _QuantizerPreHook:
    def __init__(self, observation: _QuantizerObservation) -> None:
        self.observation = observation

    def __call__(self, module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        del module
        if len(inputs) != 1:
            raise ValueError(f'Quantizer {self.observation.name} received {len(inputs)} inputs.')
        self.observation.observe(inputs[0])


def _framework_outputs(model: Network, states: Tensor) -> ModelOutputs:
    with torch.inference_mode():
        policy_logits, wdl_probabilities = model(states)
    return ModelOutputs(policy_logits.float().cpu(), wdl_probabilities.float().cpu())


def _onnx_outputs(path: Path, states: Tensor) -> ModelOutputs:
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    input_metadata = session.get_inputs()
    if len(input_metadata) != 1 or input_metadata[0].name != 'states':
        raise ValueError('The diagnostic ONNX graph must have exactly one input named states.')
    match input_metadata[0].type:
        case 'tensor(float)':
            encoded_states = states.float().cpu().numpy()
        case 'tensor(float16)':
            encoded_states = states.to(torch.float16).cpu().numpy()
        case input_type:
            raise ValueError(f'Unsupported diagnostic ONNX input type: {input_type}.')
    policy_logits, wdl_probabilities = session.run(None, {'states': encoded_states})
    return ModelOutputs(torch.from_numpy(policy_logits).float(), torch.from_numpy(wdl_probabilities).float())


def _finite_absolute_maximum(values: Tensor) -> float:
    finite_values = values[torch.isfinite(values)]
    return 0.0 if finite_values.numel() == 0 else float(finite_values.abs().max())


def _output_summary(outputs: ModelOutputs, legal_action_mask: Tensor) -> OutputSummary:
    policy_nonfinite_values = int((~torch.isfinite(outputs.policy_logits)).sum())
    wdl_nonfinite_values = int((~torch.isfinite(outputs.wdl_probabilities)).sum())
    if policy_nonfinite_values or wdl_nonfinite_values:
        return NonFiniteOutputSummary(
            policy_nonfinite_values=policy_nonfinite_values,
            wdl_nonfinite_values=wdl_nonfinite_values,
            policy_finite_absolute_maximum=_finite_absolute_maximum(outputs.policy_logits),
            wdl_finite_absolute_maximum=_finite_absolute_maximum(outputs.wdl_probabilities),
        )
    masked_logits = outputs.policy_logits.to(torch.float64).masked_fill(~legal_action_mask, float('-inf'))
    legal_log_probabilities = torch.log_softmax(masked_logits, dim=1)
    legal_probabilities = legal_log_probabilities.exp()
    entropy = -(legal_probabilities * legal_log_probabilities.nan_to_num()).sum(dim=1)
    legal_counts = legal_action_mask.sum(dim=1).to(torch.float64)
    maximum_entropy = legal_counts.log()
    entropy_ratio = torch.where(maximum_entropy > 0.0, entropy / maximum_entropy, torch.ones_like(entropy))
    top_probabilities = legal_probabilities.topk(3, dim=1).values
    expected_values = outputs.wdl_probabilities[:, 0] - outputs.wdl_probabilities[:, 2]
    mean_wdl = outputs.wdl_probabilities.to(torch.float64).mean(dim=0)
    return FiniteOutputSummary(
        policy_absolute_maximum=float(outputs.policy_logits.abs().max()),
        wdl_absolute_maximum=float(outputs.wdl_probabilities.abs().max()),
        mean_legal_policy_entropy=float(entropy.mean()),
        mean_legal_policy_entropy_ratio=float(entropy_ratio.clamp(0.0, 1.0).mean()),
        mean_legal_top1_mass=float(top_probabilities[:, 0].mean()),
        mean_legal_top3_mass=float(top_probabilities.sum(dim=1).mean()),
        mean_expected_value=float(expected_values.to(torch.float64).mean()),
        expected_value_standard_deviation=float(expected_values.to(torch.float64).std()),
        mean_wdl_probabilities=(float(mean_wdl[0]), float(mean_wdl[1]), float(mean_wdl[2])),
    )


def _compare_outputs(
    reference: ModelOutputs,
    candidate: ModelOutputs,
    legal_action_mask: Tensor,
) -> FidelityComparison:
    failures: list[str] = []
    if isinstance(_output_summary(reference, legal_action_mask), NonFiniteOutputSummary):
        failures.append('reference outputs are non-finite')
    if isinstance(_output_summary(candidate, legal_action_mask), NonFiniteOutputSummary):
        failures.append('candidate outputs are non-finite')
    if failures:
        return FailedFidelityComparison(reason='; '.join(failures))
    return SuccessfulFidelityComparison(metrics=measure_fidelity(reference, candidate, legal_action_mask))


def _copy_float_onnx(source: Path, output_directory: Path, generation: int) -> Path:
    destination = output_directory / f'generation-{generation}.fp16.onnx'
    shutil.copy2(source, destination)
    return destination


def _diagnostic_engine(onnx_path: Path, configured_template_path: Path) -> Path:
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    configured_template = runtime.deserialize_cuda_engine(configured_template_path.read_bytes())
    if configured_template is None:
        raise ValueError(f'Could not deserialize TensorRT template: {configured_template_path}')
    input_shape = engine_input_shape(configured_template)
    signature = onnx_graph_signature(onnx_path)
    selected_template_path = _build_automatic_template(
        onnx_path,
        configured_template_path,
        input_shape,
        signature,
    )
    template_sha256 = file_sha256(selected_template_path)
    engine_path = onnx_path.with_suffix(f'.trt-{template_sha256[:16]}.engine')
    with exclusive_lock(engine_path.with_suffix('.lock')):
        refit_engine(selected_template_path, onnx_path, engine_path)
    return engine_path


def _quantizer_observations(model: Network) -> tuple[tuple[_QuantizerObservation, ...], tuple[str, ...]]:
    observations: list[_QuantizerObservation] = []
    dynamic_quantizer_names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or not module.is_enabled:
            continue
        amax = module.amax
        if amax is None:
            dynamic_quantizer_names.append(name)
            continue
        kind: Literal['activation', 'weight'] = 'weight' if 'weight_quantizer' in name else 'activation'
        observations.append(_QuantizerObservation(name=name, kind=kind, amax=amax.detach()))
    return tuple(observations), tuple(dynamic_quantizer_names)


def _group_quantizers(
    observations: tuple[_QuantizerObservation, ...],
    kind: Literal['activation', 'weight'],
) -> QuantizerGroupSummary:
    selected = tuple(observation for observation in observations if observation.kind == kind)
    if not selected:
        raise ValueError(f'QAT model contains no enabled {kind} quantizers.')
    amax_values = torch.cat(tuple(observation.amax.float().cpu().reshape(-1) for observation in selected))
    total_values = sum(observation.total_values for observation in selected)
    if total_values <= 0:
        raise ValueError(f'QAT {kind} quantizers observed no values.')
    return QuantizerGroupSummary(
        quantizer_count=len(selected),
        amax_minimum=float(amax_values.min()),
        amax_mean=float(amax_values.mean()),
        amax_maximum=float(amax_values.max()),
        observed_absolute_maximum=max(observation.observed_absolute_maximum for observation in selected),
        saturation_rate=sum(observation.saturated_values for observation in selected) / total_values,
    )


def _fake_quant_outputs_and_report(model: Network, states: Tensor) -> tuple[ModelOutputs, QuantizerReport]:
    observations, dynamic_quantizer_names = _quantizer_observations(model)
    handles: list[RemovableHandle] = []
    observation_by_name = {observation.name: observation for observation in observations}
    for name, module in model.named_modules():
        observation = observation_by_name.get(name)
        if observation is not None:
            handles.append(module.register_forward_pre_hook(_QuantizerPreHook(observation)))
    try:
        outputs = _framework_outputs(model, states)
    finally:
        for handle in handles:
            handle.remove()
    return outputs, QuantizerReport(
        activation=_group_quantizers(observations, 'activation'),
        weight=_group_quantizers(observations, 'weight'),
        dynamic_quantizer_names=dynamic_quantizer_names,
    )


def _generation_report(
    arguments: Arguments,
    states: Tensor,
    legal_action_mask: Tensor,
    generation: int,
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
    device_states = states.to(device=device, dtype=torch.float32)
    model_artifact_available = checkpoint.model_path.is_file()
    original_onnx_artifact_available = checkpoint.inference_model_path.is_file()
    float_outputs: ModelOutputs | None = None
    fake_quant_outputs: ModelOutputs | None = None
    quantizer_report: QuantizerReport | None = None
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
        fake_quant_outputs, quantizer_report = _fake_quant_outputs_and_report(restored, device_states)
        with quantizers_disabled(restored):
            float_outputs = _framework_outputs(restored, device_states)

    if not original_onnx_artifact_available:
        raise ValueError(f'Generation {generation} does not retain its floating-point ONNX artifact.')
    if file_sha256(checkpoint.inference_model_path) != manifest.inference_model_sha256:
        raise ValueError(f'Checkpoint ONNX hash does not match: {checkpoint.inference_model_path}')
    float_onnx_path = _copy_float_onnx(checkpoint.inference_model_path, arguments.output_directory, generation)
    float_onnx_outputs = _onnx_outputs(float_onnx_path, states)
    template = (
        arguments.pre_fold_template if checkpoint.qat_state.phase.value == 'pre_fold' else arguments.deployment_template
    )
    tensorrt_fp16_engine_path = _diagnostic_engine(float_onnx_path, template)
    tensorrt_fp16_outputs = _TensorRtCudaGraphRunner(
        tensorrt_fp16_engine_path,
        states.to(device=device, dtype=torch.int8),
        device,
        2,
    ).outputs()

    qdq_onnx_path: Path | None = None
    tensorrt_int8_engine_path: Path | None = None
    qdq_onnx_outputs: ModelOutputs | None = None
    tensorrt_int8_outputs: ModelOutputs | None = None
    if restored is not None:
        qdq_onnx_path = arguments.output_directory / f'generation-{generation}.int8.onnx'
        export_qat_onnx(restored, qdq_onnx_path, device_states)
        qdq_onnx_outputs = _onnx_outputs(qdq_onnx_path, states)
        tensorrt_int8_engine_path = _diagnostic_engine(qdq_onnx_path, template)
        tensorrt_int8_outputs = _TensorRtCudaGraphRunner(
            tensorrt_int8_engine_path,
            states.to(device=device, dtype=torch.int8),
            device,
            2,
        ).outputs()

    named_outputs: list[tuple[str, ModelOutputs]] = []
    if float_outputs is not None:
        named_outputs.append(('pytorch_float_quantizers_disabled', float_outputs))
    if fake_quant_outputs is not None:
        named_outputs.append(('pytorch_fake_quant', fake_quant_outputs))
    named_outputs.extend((('onnx_float', float_onnx_outputs), ('tensorrt_fp16', tensorrt_fp16_outputs)))
    if qdq_onnx_outputs is not None and tensorrt_int8_outputs is not None:
        named_outputs.extend((('onnx_qdq', qdq_onnx_outputs), ('tensorrt_int8', tensorrt_int8_outputs)))
    return GenerationReport(
        generation=generation,
        qat_phase=checkpoint.qat_state.phase.value,
        completed_optimizer_steps=checkpoint.qat_state.completed_optimizer_steps,
        model_artifact_available=model_artifact_available,
        original_onnx_artifact_available=original_onnx_artifact_available,
        outputs=tuple(
            BackendReport(name=name, output=_output_summary(outputs, legal_action_mask))
            for name, outputs in named_outputs
        ),
        quantizers=quantizer_report,
        float_vs_fake_quant=(
            _compare_outputs(float_outputs, fake_quant_outputs, legal_action_mask)
            if float_outputs is not None and fake_quant_outputs is not None
            else None
        ),
        float_vs_onnx=(
            _compare_outputs(float_outputs, float_onnx_outputs, legal_action_mask)
            if float_outputs is not None
            else None
        ),
        onnx_vs_tensorrt_fp16=_compare_outputs(float_onnx_outputs, tensorrt_fp16_outputs, legal_action_mask),
        fake_quant_vs_qdq_onnx=(
            _compare_outputs(fake_quant_outputs, qdq_onnx_outputs, legal_action_mask)
            if fake_quant_outputs is not None and qdq_onnx_outputs is not None
            else None
        ),
        qdq_onnx_vs_tensorrt_int8=(
            _compare_outputs(qdq_onnx_outputs, tensorrt_int8_outputs, legal_action_mask)
            if qdq_onnx_outputs is not None and tensorrt_int8_outputs is not None
            else None
        ),
        float_onnx_path=str(float_onnx_path),
        tensorrt_fp16_engine_path=str(tensorrt_fp16_engine_path),
        qdq_onnx_path=None if qdq_onnx_path is None else str(qdq_onnx_path),
        tensorrt_int8_engine_path=(None if tensorrt_int8_engine_path is None else str(tensorrt_int8_engine_path)),
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

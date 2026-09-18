"""Compare QAT calibration corpora on one retained checkpoint."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch
from pydantic import Field
from src.experiment.configuration import ChessExperimentConfiguration, load_chess_experiment_configuration
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS, CHESS_STATE_CONTRACT
from src.replay.batch_loader import decode_states
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.training.checkpoint.contracts import CheckpointReference, read_checkpoint_manifest
from src.training.checkpoint.persistence import load_model_state_dict
from src.training.network import Network
from src.training.quantization.configuration import TensorRtInt8QatConfiguration
from src.training.quantization.runtime import (
    export_qat_onnx,
    fixed_batch_example_states,
    quantizers_disabled,
    recalibrate_qat,
    restore_qat_model,
)
from src.training.targets import build_training_target_layout
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from tools.benchmark_tensorrt_inference import _TensorRtCudaGraphRunner
from tools.diagnose_qat_checkpoint_fidelity import _diagnostic_engine
from tools.measure_inference_precision_agreement import load_positions
from tools.tensorrt_benchmark_metrics import FidelityMetrics, ModelOutputs, measure_fidelity
from torch import Tensor

INFERENCE_BATCH_SIZE = 320


@dataclass(frozen=True)
class Arguments:
    configuration: Path
    run_directory: Path
    replay: Path
    calibration_dataset: Path
    template: Path
    output_directory: Path
    output: Path
    generation: int
    replay_calibration_positions: tuple[int, ...]
    fidelity_positions: int
    random_seed: int
    device_id: int


class CalibrationVariantReport(FrozenModel):
    name: str = Field(min_length=1)
    calibration_positions: int = Field(gt=0)
    calibration_seconds: float = Field(gt=0.0)
    export_seconds: float = Field(gt=0.0)
    engine_seconds: float = Field(gt=0.0)
    float_vs_fake_quant: FidelityMetrics
    fake_quant_vs_qdq_onnx: FidelityMetrics
    qdq_onnx_vs_tensorrt_int8: FidelityMetrics
    float_vs_tensorrt_int8: FidelityMetrics
    qdq_onnx_path: str = Field(min_length=1)
    tensorrt_engine_path: str = Field(min_length=1)


class CalibrationComparisonReport(FrozenModel):
    schema_version: Literal[1] = 1
    configuration: str = Field(min_length=1)
    run_directory: str = Field(min_length=1)
    replay: str = Field(min_length=1)
    calibration_dataset: str = Field(min_length=1)
    generation: int = Field(ge=0)
    fidelity_positions: int = Field(gt=0)
    fidelity_logical_indices: tuple[int, ...] = Field(min_length=1)
    replay_calibration_logical_indices: tuple[int, ...] = Field(min_length=1)
    variants: tuple[CalibrationVariantReport, ...] = Field(min_length=1)


@dataclass(frozen=True)
class ReplaySamples:
    states: Tensor
    legal_action_mask: Tensor
    fidelity_indices: npt.NDArray[np.int64]
    calibration_indices: npt.NDArray[np.int64]


def _replay_layout(configuration: ChessExperimentConfiguration) -> ReplayLayout:
    return ReplayLayout(
        packed_planes=CHESS_STATE_CONTRACT.packed_plane_layout,
        targets=build_training_target_layout(
            CHESS_NETWORK_DIMENSIONS.actions,
            configuration.chess.objective.auxiliary_targets,
        ),
        maximum_policy_entries=configuration.training.lifecycle.replay.maximum_policy_entries,
        maximum_legal_actions=CHESS_STATE_CONTRACT.maximum_legal_action_count,
    )


def _legal_action_mask(
    legal_counts: npt.NDArray[np.uint8],
    legal_action_ids: npt.NDArray[np.uint16],
) -> Tensor:
    position_count = legal_counts.shape[0]
    slots = np.arange(legal_action_ids.shape[1])[None, :]
    populated = slots < legal_counts[:, None]
    row_indices = np.broadcast_to(np.arange(position_count)[:, None], legal_action_ids.shape)[populated]
    mask = np.zeros((position_count, CHESS_NETWORK_DIMENSIONS.actions), dtype=np.bool_)
    mask[row_indices, legal_action_ids[populated]] = True
    if not mask.any(axis=1).all():
        raise ValueError('Every replay fidelity position must contain a legal action.')
    return torch.from_numpy(mask)


def _load_replay_samples(
    arguments: Arguments,
    configuration: ChessExperimentConfiguration,
) -> ReplaySamples:
    maximum_calibration_positions = max(arguments.replay_calibration_positions)
    requested_positions = arguments.fidelity_positions + maximum_calibration_positions
    store = ReplayStore.open(arguments.replay, _replay_layout(configuration), writable=False)
    try:
        if requested_positions > store.state.size:
            raise ValueError(
                f'Replay contains {store.state.size} positions, fewer than the {requested_positions} requested.'
            )
        generator = np.random.default_rng(arguments.random_seed)
        selected = generator.choice(store.state.size, size=requested_positions, replace=False)
        fidelity_indices = selected[: arguments.fidelity_positions]
        calibration_indices = selected[arguments.fidelity_positions :]
        fidelity_columns = store.gather_logical(fidelity_indices)
        calibration_columns = store.gather_logical(calibration_indices)
    finally:
        store.close()
    fidelity_states = torch.from_numpy(decode_states(fidelity_columns.encoded_state, CHESS_STATE_CONTRACT).astype(np.int8))
    calibration_states = torch.from_numpy(
        decode_states(calibration_columns.encoded_state, CHESS_STATE_CONTRACT).astype(np.int8)
    )
    return ReplaySamples(
        states=torch.cat((fidelity_states, calibration_states), dim=0),
        legal_action_mask=_legal_action_mask(
            fidelity_columns.policy.legal_count,
            fidelity_columns.policy.legal_action_ids,
        ),
        fidelity_indices=fidelity_indices,
        calibration_indices=calibration_indices,
    )


def _restore_model(
    arguments: Arguments,
    quantization: TensorRtInt8QatConfiguration,
    device: torch.device,
) -> Network:
    manifest = read_checkpoint_manifest(arguments.generation, arguments.run_directory)
    checkpoint = CheckpointReference.from_manifest(arguments.run_directory, manifest)
    if checkpoint.qat_state is None:
        raise ValueError(f'Generation {arguments.generation} is not a QAT checkpoint.')
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
    return restored


def _framework_outputs(model: Network, states: Tensor, device: torch.device) -> ModelOutputs:
    policies: list[Tensor] = []
    wdls: list[Tensor] = []
    with torch.inference_mode():
        for batch in states.split(INFERENCE_BATCH_SIZE):
            policy, wdl = model(batch.to(device=device, dtype=torch.float32))
            policies.append(policy.float().cpu())
            wdls.append(wdl.float().cpu())
    return ModelOutputs(torch.cat(policies), torch.cat(wdls))


def _onnx_outputs(path: Path, states: Tensor, device_id: int) -> ModelOutputs:
    session = ort.InferenceSession(
        str(path),
        providers=[('CUDAExecutionProvider', {'device_id': str(device_id)}), 'CPUExecutionProvider'],
    )
    input_metadata = session.get_inputs()
    if len(input_metadata) != 1 or input_metadata[0].name != 'states':
        raise ValueError('The QAT ONNX graph must have exactly one input named states.')
    policies: list[Tensor] = []
    wdls: list[Tensor] = []
    match input_metadata[0].type:
        case 'tensor(float)':
            input_dtype = torch.float32
        case 'tensor(float16)':
            input_dtype = torch.float16
        case input_type:
            raise ValueError(f'Unsupported QAT ONNX input type: {input_type}.')
    for batch in states.split(INFERENCE_BATCH_SIZE):
        policy, wdl = session.run(None, {'states': batch.to(input_dtype).numpy()})
        policies.append(torch.from_numpy(policy).float())
        wdls.append(torch.from_numpy(wdl).float())
    return ModelOutputs(torch.cat(policies), torch.cat(wdls))


def _tensorrt_outputs(path: Path, states: Tensor, device: torch.device) -> ModelOutputs:
    batches = states.split(INFERENCE_BATCH_SIZE)
    if any(batch.shape[0] != INFERENCE_BATCH_SIZE for batch in batches):
        raise ValueError(f'Fidelity positions must be divisible by {INFERENCE_BATCH_SIZE}.')
    runner = _TensorRtCudaGraphRunner(path, batches[0], device, 2)
    policies: list[Tensor] = []
    wdls: list[Tensor] = []
    for batch in batches:
        runner.load_states(batch)
        outputs = runner.outputs()
        policies.append(outputs.policy_logits)
        wdls.append(outputs.wdl_probabilities)
    return ModelOutputs(torch.cat(policies), torch.cat(wdls))


def _calibrate(model: Network, states: Tensor, device: torch.device) -> float:
    def calibration_loop(calibration_model: torch.nn.Module) -> None:
        with torch.inference_mode():
            for batch in states.split(INFERENCE_BATCH_SIZE):
                calibration_model(batch.to(device=device, dtype=torch.float32))

    started_at = time.perf_counter()
    recalibrate_qat(model, calibration_loop, distributed_sync=False)
    torch.cuda.synchronize(device)
    return time.perf_counter() - started_at


def _variant_report(
    arguments: Arguments,
    quantization: TensorRtInt8QatConfiguration,
    name: str,
    calibration_states: Tensor,
    fidelity_states: Tensor,
    legal_action_mask: Tensor,
    float_outputs: ModelOutputs,
    device: torch.device,
) -> CalibrationVariantReport:
    model = _restore_model(arguments, quantization, device)
    calibration_seconds = _calibrate(model, calibration_states, device)
    fake_quant_outputs = _framework_outputs(model, fidelity_states, device)
    example_states = fixed_batch_example_states(calibration_states, INFERENCE_BATCH_SIZE).to(
        device=device,
        dtype=torch.float32,
    )
    qdq_onnx_path = arguments.output_directory / f'{name}.int8.onnx'
    export_started_at = time.perf_counter()
    export_qat_onnx(model, qdq_onnx_path, example_states)
    export_seconds = time.perf_counter() - export_started_at
    qdq_onnx_outputs = _onnx_outputs(qdq_onnx_path, fidelity_states, arguments.device_id)
    engine_started_at = time.perf_counter()
    engine_path = _diagnostic_engine(qdq_onnx_path, arguments.template)
    engine_seconds = time.perf_counter() - engine_started_at
    tensorrt_outputs = _tensorrt_outputs(engine_path, fidelity_states, device)
    return CalibrationVariantReport(
        name=name,
        calibration_positions=calibration_states.shape[0],
        calibration_seconds=calibration_seconds,
        export_seconds=export_seconds,
        engine_seconds=engine_seconds,
        float_vs_fake_quant=measure_fidelity(float_outputs, fake_quant_outputs, legal_action_mask),
        fake_quant_vs_qdq_onnx=measure_fidelity(fake_quant_outputs, qdq_onnx_outputs, legal_action_mask),
        qdq_onnx_vs_tensorrt_int8=measure_fidelity(qdq_onnx_outputs, tensorrt_outputs, legal_action_mask),
        float_vs_tensorrt_int8=measure_fidelity(float_outputs, tensorrt_outputs, legal_action_mask),
        qdq_onnx_path=str(qdq_onnx_path),
        tensorrt_engine_path=str(engine_path),
    )


def run(arguments: Arguments) -> CalibrationComparisonReport:
    if arguments.fidelity_positions <= 0 or arguments.fidelity_positions % INFERENCE_BATCH_SIZE:
        raise ValueError(f'Fidelity positions must be a positive multiple of {INFERENCE_BATCH_SIZE}.')
    if not arguments.replay_calibration_positions or any(
        position_count <= 0 for position_count in arguments.replay_calibration_positions
    ):
        raise ValueError('Replay calibration position counts must be positive.')
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    configuration = load_chess_experiment_configuration(arguments.configuration)
    quantization = configuration.training.trainer.quantization
    if not isinstance(quantization, TensorRtInt8QatConfiguration):
        raise ValueError('Calibration comparison requires a TensorRT INT8 QAT configuration.')
    device = torch.device('cuda', arguments.device_id)
    torch.cuda.set_device(device)
    replay_samples = _load_replay_samples(arguments, configuration)
    fidelity_states = replay_samples.states[: arguments.fidelity_positions]
    replay_calibration_states = replay_samples.states[arguments.fidelity_positions :]
    fixed_calibration_states, _ = load_positions(
        arguments.calibration_dataset,
        quantization.calibration_positions,
    )
    reference_model = _restore_model(arguments, quantization, device)
    with quantizers_disabled(reference_model):
        float_outputs = _framework_outputs(reference_model, fidelity_states, device)
    variants = [
        _variant_report(
            arguments,
            quantization,
            f'fixed-evaluation-{fixed_calibration_states.shape[0]}',
            fixed_calibration_states,
            fidelity_states,
            replay_samples.legal_action_mask,
            float_outputs,
            device,
        )
    ]
    for position_count in arguments.replay_calibration_positions:
        variants.append(
            _variant_report(
                arguments,
                quantization,
                f'replay-{position_count}',
                replay_calibration_states[:position_count],
                fidelity_states,
                replay_samples.legal_action_mask,
                float_outputs,
                device,
            )
        )
    report = CalibrationComparisonReport(
        configuration=str(arguments.configuration),
        run_directory=str(arguments.run_directory),
        replay=str(arguments.replay),
        calibration_dataset=str(arguments.calibration_dataset),
        generation=arguments.generation,
        fidelity_positions=arguments.fidelity_positions,
        fidelity_logical_indices=tuple(int(index) for index in replay_samples.fidelity_indices),
        replay_calibration_logical_indices=tuple(int(index) for index in replay_samples.calibration_indices),
        variants=tuple(variants),
    )
    write_text_atomically(arguments.output, report.model_dump_json(indent=2) + '\n')
    return report


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', required=True, type=Path)
    parser.add_argument('--run-directory', required=True, type=Path)
    parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--calibration-dataset', required=True, type=Path)
    parser.add_argument('--template', required=True, type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--generation', required=True, type=int)
    parser.add_argument('--replay-calibration-positions', action='append', required=True, type=int)
    parser.add_argument('--fidelity-positions', default=3_200, type=int)
    parser.add_argument('--random-seed', default=20260918, type=int)
    parser.add_argument('--device-id', default=7, type=int)
    namespace = parser.parse_args()
    return Arguments(
        configuration=namespace.configuration.resolve(),
        run_directory=namespace.run_directory.resolve(),
        replay=namespace.replay.resolve(),
        calibration_dataset=namespace.calibration_dataset.resolve(),
        template=namespace.template.resolve(),
        output_directory=namespace.output_directory.resolve(),
        output=namespace.output.resolve(),
        generation=namespace.generation,
        replay_calibration_positions=tuple(namespace.replay_calibration_positions),
        fidelity_positions=namespace.fidelity_positions,
        random_seed=namespace.random_seed,
        device_id=namespace.device_id,
    )


def main() -> None:
    print(run(parse_arguments()).model_dump_json(indent=2))


if __name__ == '__main__':
    main()

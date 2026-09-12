from __future__ import annotations

import copy
import argparse
import json
import time
from pathlib import Path

import modelopt.torch.quantization as mtq
import numpy as np
import onnx
import torch
import torch.nn.functional as functional
from torch import nn

from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS, CHESS_STATE_CONTRACT
from src.replay.batch_loader import decode_states
from src.replay.store import ReplayStore
from src.training.checkpoint.contracts import load_checkpoint_manifest_path
from src.training.checkpoint.persistence import load_model
from src.training.network import InferenceNetwork
from tools.benchmark_tensorrt_inference import (
    BATCH_SIZE,
    Backend,
    _TensorRtCudaGraphRunner,
    _build_engine,
    _measure_runner,
    _replay_layout,
)
from tools.tensorrt_benchmark_metrics import FidelityLimits, ModelOutputs, measure_fidelity

CONFIGURATION = Path('/workspace/run-control/configs/vast-chess-8gpu-integrated-v34-resume-g1702.yaml')
RUN = Path(
    '/workspace/alphazero-engine-v34-lr-001/py/training_data/production/vast-chess-8gpu-integrated-v34'
)
REPLAY = RUN / 'replay.bin'
MANIFEST = RUN / 'checkpoint_1785.json'
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', choices=('early8', 'full'), required=True)
    parser.add_argument('--steps', type=int, default=1_000)
    parser.add_argument('--sample-count', type=int, default=128_000)
    parser.add_argument('--learning-rate', type=float, default=3e-6)
    parser.add_argument('--freeze-quantized-only', action='store_true')
    parser.add_argument('--smoothquant', action='store_true')
    parser.add_argument('--per-channel-activation', action='store_true')
    parser.add_argument('--weight-only', action='store_true')
    parser.add_argument('--value-channels', type=int, choices=(2, 32), default=2)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def load_sample(sample_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    configuration = load_chess_experiment_configuration(CONFIGURATION)
    store = ReplayStore.open(REPLAY, _replay_layout(configuration), writable=False)
    try:
        indices = np.sort(np.random.default_rng(20260913).choice(store.state.size, sample_count, replace=False))
        columns = store.gather_logical(indices)
        states = torch.from_numpy(decode_states(columns.encoded_state.copy(), CHESS_STATE_CONTRACT).astype(np.int8))
        legal_ids = columns.policy.legal_action_ids.astype(np.int64)
        legal_counts = columns.policy.legal_count.astype(np.int64)
    finally:
        store.close()
    legal_mask = torch.zeros((sample_count, CHESS_NETWORK_DIMENSIONS.actions), dtype=torch.bool)
    positions = np.arange(legal_ids.shape[1])[None, :] < legal_counts[:, None]
    rows = np.broadcast_to(np.arange(sample_count)[:, None], legal_ids.shape)[positions]
    legal_mask[torch.from_numpy(rows), torch.from_numpy(legal_ids[positions])] = True
    return states, legal_mask


def load_inference_model(device: torch.device) -> InferenceNetwork:
    manifest = load_checkpoint_manifest_path(MANIFEST, 1785)
    training_model = load_model(
        RUN / manifest.model_path,
        manifest.network.architecture,
        device,
        manifest.network.dimensions,
        manifest.network.auxiliary_heads,
    )
    model = InferenceNetwork(training_model)
    model.eval()
    model.fuse_model()
    return model


def outputs(model: torch.nn.Module, states: torch.Tensor, data_type: torch.dtype, device: torch.device) -> ModelOutputs:
    policies: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(states), BATCH_SIZE):
            batch = states[start : start + BATCH_SIZE].to(device=device, dtype=data_type, memory_format=torch.channels_last)
            policy, value = model(batch)
            policies.append(policy.float().cpu())
            values.append(value.float().cpu())
    return ModelOutputs(torch.cat(policies), torch.cat(values))


def fidelity(reference: ModelOutputs, candidate: ModelOutputs, legal_mask: torch.Tensor) -> dict[str, float | int]:
    return measure_fidelity(reference, candidate, legal_mask).model_dump()


def main() -> None:
    arguments = parse_arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    training_count = arguments.sample_count * 4 // 5
    selection_count = arguments.sample_count // 10
    device = torch.device('cuda', 0)
    states, legal_mask = load_sample(arguments.sample_count)
    teacher = load_inference_model(device).to(dtype=torch.bfloat16, memory_format=torch.channels_last)
    student = copy.deepcopy(teacher).to(dtype=torch.float32, memory_format=torch.channels_last)
    if arguments.value_channels == 32:
        student.value_head = nn.Sequential(
            nn.Conv2d(160, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 3),
        ).to(device=device, dtype=torch.float32)

    calibration_states = states[: arguments.sample_count // 10]

    def calibration_loop(model: torch.nn.Module) -> None:
        with torch.inference_mode():
            for start in range(0, len(calibration_states), BATCH_SIZE):
                model(
                    calibration_states[start : start + BATCH_SIZE].to(
                        device=device, dtype=torch.float32, memory_format=torch.channels_last
                    )
                )

    configuration = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG if arguments.smoothquant else mtq.INT8_DEFAULT_CFG)
    configuration['quant_cfg'].extend(
        (
            {'quantizer_name': '*', 'parent_class': 'nn.Linear', 'enable': False},
            {'quantizer_name': '*policy_head*', 'enable': False},
            {'quantizer_name': '*value_head*', 'enable': False},
        )
    )
    if arguments.per_channel_activation:
        configuration['quant_cfg'].append(
            {'quantizer_name': '*input_quantizer', 'cfg': {'num_bits': 8, 'axis': 1}}
        )
    if arguments.weight_only:
        configuration['quant_cfg'].append({'quantizer_name': '*input_quantizer', 'enable': False})
    if arguments.partition == 'early8':
        configuration['quant_cfg'].append({'quantizer_name': '*start_block*', 'enable': False})
        configuration['quant_cfg'].extend(
            {'quantizer_name': f'*backbone.{block}.*', 'enable': False} for block in range(4, 14)
        )
    student = mtq.quantize(student, configuration, calibration_loop)

    selection_states = states[training_count : training_count + selection_count]
    selection_mask = legal_mask[training_count : training_count + selection_count]
    holdout_states = states[training_count + selection_count :]
    holdout_mask = legal_mask[training_count + selection_count :]
    teacher_selection = outputs(teacher, selection_states, torch.bfloat16, device)
    teacher_holdout = outputs(teacher, holdout_states, torch.bfloat16, device)
    before_selection = outputs(student, selection_states, torch.float32, device)
    before_holdout = outputs(student, holdout_states, torch.float32, device)

    generator = torch.Generator(device='cpu').manual_seed(20260913)
    if arguments.value_channels == 32:
        for name, parameter in student.named_parameters():
            parameter.requires_grad = name.startswith('value_head.')
    elif arguments.partition == 'early8' and arguments.freeze_quantized_only:
        for name, parameter in student.named_parameters():
            parameter.requires_grad = name.startswith(('backbone.0.', 'backbone.1.', 'backbone.2.', 'backbone.3.'))
    elif arguments.partition == 'early8':
        for parameter in student.parameters():
            parameter.requires_grad = True
    trainable_parameters = tuple(parameter for parameter in student.parameters() if parameter.requires_grad)
    print(f'Trainable parameters: {sum(parameter.numel() for parameter in trainable_parameters):,}', flush=True)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=arguments.learning_rate, weight_decay=0.0)
    student.train()
    started = time.perf_counter()
    losses: list[float] = []
    selection_history: list[dict[str, object]] = []
    initial_selection_fidelity = fidelity(teacher_selection, before_selection, selection_mask)
    best_score = (
        -float(initial_selection_fidelity['expected_value_mean_absolute_error'])
        if arguments.value_channels == 32
        else float(initial_selection_fidelity['policy_top1_agreement'])
    )
    best_state = copy.deepcopy(student.state_dict())
    for step in range(arguments.steps):
        indices = torch.randint(0, training_count, (BATCH_SIZE,), generator=generator)
        batch_states = states[indices].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        batch_mask = legal_mask[indices].to(device=device)
        with torch.inference_mode():
            teacher_policy, teacher_wdl = teacher(batch_states.to(dtype=torch.bfloat16))
        student_policy, student_wdl = student(batch_states)
        teacher_probability = torch.softmax(teacher_policy.float().masked_fill(~batch_mask, -10_000.0), dim=1)
        student_log_probability = torch.log_softmax(
            student_policy.float().masked_fill(~batch_mask, -10_000.0), dim=1
        )
        policy_loss = -(teacher_probability * student_log_probability).sum(dim=1).mean()
        value_loss = functional.mse_loss(student_wdl.float(), teacher_wdl.float())
        loss = value_loss if arguments.value_channels == 32 else policy_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if not (step + 1) % 100:
            student.eval()
            selection_outputs = outputs(student, selection_states, torch.float32, device)
            selection_fidelity = fidelity(teacher_selection, selection_outputs, selection_mask)
            selection_history.append({'step': step + 1, 'fidelity': selection_fidelity})
            score = (
                -float(selection_fidelity['expected_value_mean_absolute_error'])
                if arguments.value_channels == 32
                else float(selection_fidelity['policy_top1_agreement'])
            )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(student.state_dict())
            student.train()
            print(step + 1, sum(losses[-100:]) / 100, selection_fidelity, flush=True)
    training_seconds = time.perf_counter() - started
    student.load_state_dict(best_state)
    student.eval()
    after_holdout = outputs(student, holdout_states, torch.float32, device)

    onnx_path = arguments.output / 'qat-int8.onnx'
    temporary_path = arguments.output / '.qat-int8.onnx.tmp'
    with torch.inference_mode():
        torch.onnx.export(
            student,
            (torch.zeros((BATCH_SIZE, *states.shape[1:]), device=device, dtype=torch.float32),),
            str(temporary_path),
            input_names=('states',),
            output_names=('policy_logits', 'wdl_probabilities'),
            opset_version=20,
            do_constant_folding=True,
            dynamo=False,
        )
    temporary_path.replace(onnx_path)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    engine_path = arguments.output / 'qat-int8.engine'
    engine, _, _ = _build_engine(onnx_path, engine_path, Backend.TENSORRT_INT8)
    runner_states = holdout_states[:BATCH_SIZE]
    runner = _TensorRtCudaGraphRunner(engine_path, runner_states, device, 10)
    tensorrt_outputs = runner.outputs()
    reference_batch = ModelOutputs(
        teacher_holdout.policy_logits[:BATCH_SIZE], teacher_holdout.wdl_probabilities[:BATCH_SIZE]
    )
    report = {
        'sample_count': arguments.sample_count,
        'training_positions': training_count,
        'selection_positions': selection_count,
        'holdout_positions': len(holdout_states),
        'partition': arguments.partition,
        'steps': arguments.steps,
        'learning_rate': arguments.learning_rate,
        'freeze_quantized_only': arguments.freeze_quantized_only,
        'smoothquant': arguments.smoothquant,
        'per_channel_activation': arguments.per_channel_activation,
        'weight_only': arguments.weight_only,
        'value_channels': arguments.value_channels,
        'training_seconds': training_seconds,
        'initial_selection_fidelity': initial_selection_fidelity,
        'initial_holdout_fidelity': fidelity(teacher_holdout, before_holdout, holdout_mask),
        'selection_history': selection_history,
        'selected_score': best_score,
        'final_holdout_fidelity': fidelity(teacher_holdout, after_holdout, holdout_mask),
        'tensorrt_holdout_fidelity': fidelity(reference_batch, tensorrt_outputs, holdout_mask[:BATCH_SIZE]),
        'tensorrt_timing': _measure_runner(runner, 10, 3, 20, device).model_dump(),
        'onnx_quantize_linear_nodes': sum(node.op_type == 'QuantizeLinear' for node in onnx_model.graph.node),
        'onnx_dequantize_linear_nodes': sum(node.op_type == 'DequantizeLinear' for node in onnx_model.graph.node),
        'engine_path': engine.path,
        'loss_tail_mean': sum(losses[-200:]) / len(losses[-200:]) if losses else None,
    }
    (arguments.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

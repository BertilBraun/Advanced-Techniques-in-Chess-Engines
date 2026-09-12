from __future__ import annotations

import argparse
import copy
import hashlib
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Literal

import modelopt.torch.quantization as mtq
import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import torch
from modelopt.torch.quantization.config import QuantizeConfig, QuantizerCfgEntry
from modelopt.torch.quantization.nn import TensorQuantizer
from pydantic import Field
from src.distillation.dataset import build_replay_training_batch
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS, CHESS_STATE_CONTRACT
from src.training.batch import TrainingBatch
from src.training.model_cost import ModelCost, measure_model_cost
from src.training.network import (
    ChessFromToAttentionPolicyHeadConfiguration,
    GlobalPoolingResBlock,
    GlobalPoolingResidualContext,
    Network,
    NetworkParams,
    PostActivationResidualBlockConfiguration,
    ResBlock,
    ResidualContextPlacement,
    ScaledPostActivationGlobalPoolingResBlock,
    ScaledPostActivationResBlock,
    ScaledPostActivationResidualBlockConfiguration,
    ScaledPreActivationResidualBlockConfiguration,
)
from src.training.objective import ResolvedTrainingObjective
from src.util.atomic_file import write_bytes_atomically, write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from src.util.provenance import SourceRevision, read_source_revision
from tools.benchmark_tensorrt_inference import (
    BATCH_SIZE as TENSORRT_BATCH_SIZE,
)
from tools.benchmark_tensorrt_inference import (
    ArtifactIdentity,
    Backend,
    _build_engine,
    _measure_runner,
    _TensorRtCudaGraphRunner,
    _TorchScriptCudaGraphRunner,
)
from tools.benchmark_training_overfit import LossValues, achievable_loss_floor
from tools.distill_train_student import (
    OpenedProductionReplay,
    OptimizerKind,
    ProductionReplayInput,
    ProductionReplaySnapshot,
    close_training_dataset,
    create_student_optimizer,
    dataset_split,
    distillation_objective,
    learning_rate_at,
    mean_loss_values,
    observed_losses,
    open_training_dataset,
    source_held_out_batches,
)
from tools.tensorrt_benchmark_metrics import FidelityMetrics, ModelOutputs, TimingDistribution, measure_fidelity
from torch import Tensor, nn

DEFAULT_LAYERS = 12
DEFAULT_HIDDEN_SIZE = 128
POLICY_KEY_SIZE = 128
VALUE_CHANNELS = 2
VALUE_FULLY_CONNECTED_SIZE = 48
# The published 3,451,655 count used the former 29-plane encoding. The v34 replay has 52 input planes,
# adding 23 * 128 * 3 * 3 start-convolution weights while preserving the 12x128 trunk and heads.
EXPECTED_PARAMETER_COUNT = 3_478_151
ACTIVATION_CAP = 6.0
QAT_RECALIBRATION_INTERVAL = 1_000
QAT_CALIBRATION_POSITIONS = 3_200
FINAL_FIDELITY_POSITIONS = 51_200
ACTIVATION_RANGE_POSITIONS = 64


class ScreenCell(str, Enum):
    POST_FLOAT = 'post_float'
    POST_QAT = 'post_qat'
    POST_SCALED_FLOAT = 'post_scaled_float'
    POST_SCALED_QAT = 'post_scaled_qat'
    PRE_SCALED_FLOAT = 'pre_scaled_float'
    PRE_SCALED_QAT = 'pre_scaled_qat'

    @property
    def uses_quantization_friendly_trunk(self) -> bool:
        return self in (
            ScreenCell.POST_SCALED_FLOAT,
            ScreenCell.POST_SCALED_QAT,
            ScreenCell.PRE_SCALED_FLOAT,
            ScreenCell.PRE_SCALED_QAT,
        )

    @property
    def uses_pre_activation_trunk(self) -> bool:
        return self in (ScreenCell.PRE_SCALED_FLOAT, ScreenCell.PRE_SCALED_QAT)

    @property
    def uses_qat(self) -> bool:
        return self in (ScreenCell.POST_QAT, ScreenCell.POST_SCALED_QAT, ScreenCell.PRE_SCALED_QAT)


@dataclass(frozen=True)
class Arguments:
    replay_store: Path
    replay_experiment: Path
    replay_sha256: str
    output: Path
    cell: ScreenCell
    random_seed: int
    device_id: int
    steps: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    evaluate_every: int
    holdout_fraction: float
    final_fidelity_positions: int
    layers: int
    hidden_size: int
    quantized_convolutions: int
    final_normalization: bool
    fold_post_activation_batch_norm: bool
    strongly_typed_tensorrt: bool
    constrain_tensorrt_float16_islands: bool


class TrainingObservation(FrozenModel):
    step: int = Field(ge=0)
    learning_rate: float = Field(ge=0.0)
    elapsed_seconds: float = Field(ge=0.0)
    samples_per_second: float = Field(gt=0.0)
    training: LossValues
    held_out: LossValues
    held_out_policy_gap_above_floor: float


class ActivationDistribution(FrozenModel):
    name: str = Field(min_length=1)
    median_absolute: float = Field(ge=0.0)
    p95_absolute: float = Field(ge=0.0)
    p99_absolute: float = Field(ge=0.0)
    p999_absolute: float = Field(ge=0.0)
    maximum_absolute: float = Field(ge=0.0)


class TensorRtMeasurement(FrozenModel):
    onnx_path: str = Field(min_length=1)
    onnx_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    engine_path: str = Field(min_length=1)
    engine_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    timing: TimingDistribution
    fidelity_to_floating: FidelityMetrics
    onnx_fidelity_to_floating: FidelityMetrics
    onnx_fidelity_to_framework: FidelityMetrics
    tensorrt_fidelity_to_onnx: FidelityMetrics


class ArchitectureScreenReport(FrozenModel):
    schema_version: Literal[4] = 4
    source_revision: SourceRevision
    cell: ScreenCell
    random_seed: int
    device_id: int
    replay: ProductionReplaySnapshot
    sampled_index_sequence_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    architecture: NetworkParams
    quantized_convolutions: int = Field(ge=0)
    folded_post_activation_batch_norm: bool
    strongly_typed_tensorrt: bool
    constrained_tensorrt_float16_islands: bool
    model_cost: ModelCost
    steps: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    optimizer: Literal['adamw'] = 'adamw'
    peak_learning_rate: float = Field(gt=0.0)
    learning_rate_schedule: Literal['cosine'] = 'cosine'
    warmup_steps: int = Field(gt=0)
    holdout_fraction: float = Field(gt=0.0, lt=1.0)
    held_out_positions: int = Field(gt=0)
    held_out_loss_floor: LossValues
    observations: tuple[TrainingObservation, ...]
    training_wall_seconds: float = Field(ge=0.0)
    average_training_samples_per_second: float | None = Field(default=None, gt=0.0)
    qat_recalibration_interval: int | None = Field(default=None, gt=0)
    qat_calibration_positions: int | None = Field(default=None, gt=0)
    activation_ranges: tuple[ActivationDistribution, ...]
    floating_held_out_loss: LossValues
    fake_quant_held_out_loss: LossValues | None
    floating_to_fake_quant_fidelity: FidelityMetrics | None
    torchscript_bfloat16_timing: TimingDistribution
    tensorrt_float16: TensorRtMeasurement
    tensorrt_int8: TensorRtMeasurement | None
    state_dict_path: str = Field(min_length=1)
    state_dict_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


def architecture(
    cell: ScreenCell,
    layers: int = DEFAULT_LAYERS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    final_normalization: bool = False,
) -> NetworkParams:
    match cell:
        case ScreenCell.POST_FLOAT | ScreenCell.POST_QAT:
            residual_block = PostActivationResidualBlockConfiguration()
        case ScreenCell.POST_SCALED_FLOAT | ScreenCell.POST_SCALED_QAT:
            residual_block = ScaledPostActivationResidualBlockConfiguration(
                branch_scale=layers**-0.5,
                activation_cap=ACTIVATION_CAP,
            )
        case ScreenCell.PRE_SCALED_FLOAT | ScreenCell.PRE_SCALED_QAT:
            residual_block = ScaledPreActivationResidualBlockConfiguration(
                branch_scale=layers**-0.5,
                activation_cap=ACTIVATION_CAP,
                final_activation_cap=ACTIVATION_CAP if final_normalization else None,
            )
    return NetworkParams(
        num_layers=layers,
        hidden_size=hidden_size,
        residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK),
        residual_block=residual_block,
        policy_head=ChessFromToAttentionPolicyHeadConfiguration(key_size=POLICY_KEY_SIZE),
        num_value_channels=VALUE_CHANNELS,
        value_fc_size=VALUE_FULLY_CONNECTED_SIZE,
    )


def _qat_configuration(cell: ScreenCell, layers: int, quantized_convolutions: int) -> QuantizeConfig:
    configuration = QuantizeConfig.model_validate(copy.deepcopy(mtq.INT8_DEFAULT_CFG))
    configuration.quant_cfg.extend(
        (
            QuantizerCfgEntry(quantizer_name='*', parent_class='nn.Linear', enable=False),
            QuantizerCfgEntry(quantizer_name='*start_block*', enable=False),
            QuantizerCfgEntry(quantizer_name='*policy_head*', enable=False),
            QuantizerCfgEntry(quantizer_name='*value_head*', enable=False),
        )
    )
    convolution_module_index = 2 if cell.uses_pre_activation_trunk else 0
    for convolution_index in range(quantized_convolutions, layers * 2):
        block_index, convolution_in_block = divmod(convolution_index, 2)
        configuration.quant_cfg.append(
            QuantizerCfgEntry(
                quantizer_name=(
                    f'backbone.{block_index}.conv_block{convolution_in_block + 1}.{convolution_module_index}.*'
                ),
                enable=False,
            )
        )
    return configuration


def _replay_batch(dataset: OpenedProductionReplay, indices: np.ndarray, device: torch.device) -> TrainingBatch:
    return build_replay_training_batch(
        dataset.store.gather_logical(indices),
        CHESS_STATE_CONTRACT,
        dataset.action_size,
        device,
    )


def _calibration_loop(
    dataset: OpenedProductionReplay,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Callable[[nn.Module], None]:
    def run(model: nn.Module) -> None:
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for start in range(0, len(indices), batch_size):
                    batch = _replay_batch(dataset, indices[start : start + batch_size], device)
                    model(batch.states)
        finally:
            model.train(was_training)

    return run


def _configure_qat(
    model: Network,
    dataset: OpenedProductionReplay,
    calibration_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    cell: ScreenCell,
    layers: int,
    quantized_convolutions: int,
) -> Network:
    quantized = mtq.quantize(
        model,
        _qat_configuration(cell, layers, quantized_convolutions),
        _calibration_loop(dataset, calibration_indices, batch_size, device),
    )
    assert isinstance(quantized, Network)
    return quantized


def _fold_convolution_batch_norm(block: nn.Sequential) -> None:
    convolution = block[0]
    batch_norm = block[1]
    assert isinstance(convolution, nn.Conv2d)
    assert isinstance(batch_norm, nn.BatchNorm2d)
    block[0] = torch.nn.utils.fusion.fuse_conv_bn_eval(convolution, batch_norm)
    block[1] = nn.Identity()


def _fold_post_activation_batch_norm(model: Network) -> None:
    model.eval()
    for block in model.backbone:
        match block:
            case (
                ResBlock()
                | GlobalPoolingResBlock()
                | ScaledPostActivationResBlock()
                | ScaledPostActivationGlobalPoolingResBlock()
            ):
                _fold_convolution_batch_norm(block.conv_block1)
                _fold_convolution_batch_norm(block.conv_block2)
            case _:
                raise ValueError('Batch-normalization folding requires the post-activation residual trunk.')
    model.train()


@contextmanager
def _quantizers_disabled(model: Network) -> Iterator[None]:
    quantizers = tuple(module for module in model.modules() if isinstance(module, TensorQuantizer))
    originally_enabled = tuple(quantizer.is_enabled for quantizer in quantizers)
    try:
        for quantizer in quantizers:
            quantizer.disable()
        yield
    finally:
        for quantizer, was_enabled in zip(quantizers, originally_enabled):
            if was_enabled:
                quantizer.enable()
            else:
                quantizer.disable()


def _evaluate(
    model: Network,
    batches: tuple[TrainingBatch, ...],
    objective: ResolvedTrainingObjective,
    device: torch.device,
) -> LossValues:
    model.eval()
    with (
        torch.inference_mode(),
        torch.autocast(device_type='cuda', dtype=torch.bfloat16),
    ):
        values = tuple(
            observed_losses(objective.calculate_loss(model.training_output(batch.states), batch)) for batch in batches
        )
    model.train()
    return mean_loss_values(values)


def _legal_action_mask(legal_action_ids: Tensor) -> Tensor:
    mask = torch.zeros((len(legal_action_ids), CHESS_NETWORK_DIMENSIONS.actions), dtype=torch.bool)
    valid = legal_action_ids.cpu() >= 0
    rows = torch.arange(len(legal_action_ids))[:, None].expand_as(valid)[valid]
    mask[rows, legal_action_ids.cpu()[valid]] = True
    return mask


def _model_outputs(
    model: Network,
    batches: tuple[TrainingBatch, ...],
    device: torch.device,
) -> tuple[ModelOutputs, Tensor]:
    policies: list[Tensor] = []
    values: list[Tensor] = []
    legal_masks: list[Tensor] = []
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for batch in batches:
            policy, value = model(batch.states)
            policies.append(policy.float().cpu())
            values.append(value.float().cpu())
            legal_masks.append(_legal_action_mask(batch.policy_legal_action_ids))
    model.train()
    return ModelOutputs(torch.cat(policies), torch.cat(values)), torch.cat(legal_masks)


def _held_out_batches(
    dataset: OpenedProductionReplay,
    held_out_start: int,
    position_count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[TrainingBatch, ...]:
    return tuple(
        _replay_batch(
            dataset,
            np.arange(start, min(start + batch_size, held_out_start + position_count), dtype=np.int64),
            device,
        )
        for start in range(held_out_start, held_out_start + position_count, batch_size)
    )


def _activation_ranges(
    model: Network,
    batch: TrainingBatch,
    device: torch.device,
) -> tuple[ActivationDistribution, ...]:
    observed: dict[str, Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture(name: str):
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
            observed[f'{name}.input'] = inputs[0].detach().float().abs().cpu().flatten()
            observed[f'{name}.output'] = output.detach().float().abs().cpu().flatten()

        return hook

    for index, block in enumerate(model.backbone):
        handles.append(block.register_forward_hook(capture(f'backbone.{index}')))
        for name, module in block.named_modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(capture(f'backbone.{index}.{name}')))
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        model(batch.states[:ACTIVATION_RANGE_POSITIONS])
    for handle in handles:
        handle.remove()
    model.train()
    quantiles = torch.tensor((0.5, 0.95, 0.99, 0.999))
    return tuple(
        ActivationDistribution(
            name=name,
            median_absolute=float(torch.quantile(values, quantiles[0])),
            p95_absolute=float(torch.quantile(values, quantiles[1])),
            p99_absolute=float(torch.quantile(values, quantiles[2])),
            p999_absolute=float(torch.quantile(values, quantiles[3])),
            maximum_absolute=float(values.max()),
        )
        for name, values in sorted(observed.items())
    )


def _export_onnx(model: Network, path: Path, device: torch.device) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.unlink(missing_ok=True)
    model.eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (
                torch.zeros(
                    (
                        TENSORRT_BATCH_SIZE,
                        CHESS_NETWORK_DIMENSIONS.channels,
                        CHESS_NETWORK_DIMENSIONS.rows,
                        CHESS_NETWORK_DIMENSIONS.columns,
                    ),
                    device=device,
                ),
            ),
            str(temporary),
            input_names=('states',),
            output_names=('policy_logits', 'wdl_probabilities'),
            opset_version=20,
            do_constant_folding=True,
            dynamo=False,
        )
    exported = onnx.load(temporary)
    onnx.checker.check_model(exported, full_check=True)
    temporary.replace(path)
    model.train()


def _tensorrt_outputs(
    runner: _TensorRtCudaGraphRunner,
    batches: tuple[TrainingBatch, ...],
) -> ModelOutputs:
    policies: list[Tensor] = []
    values: list[Tensor] = []
    for batch in batches:
        runner.load_states(batch.states)
        outputs = runner.outputs()
        policies.append(outputs.policy_logits)
        values.append(outputs.wdl_probabilities)
    return ModelOutputs(torch.cat(policies), torch.cat(values))


def _onnx_outputs(
    onnx_path: Path,
    batches: tuple[TrainingBatch, ...],
    device_id: int,
) -> ModelOutputs:
    session = ort.InferenceSession(
        str(onnx_path),
        providers=[('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider'],
    )
    policies: list[Tensor] = []
    values: list[Tensor] = []
    for batch in batches:
        policy, value = session.run(None, {'states': batch.states.float().cpu().numpy()})
        policies.append(torch.from_numpy(policy).float())
        values.append(torch.from_numpy(value).float())
    return ModelOutputs(torch.cat(policies), torch.cat(values))


def _measure_tensorrt(
    model: Network,
    output: Path,
    name: str,
    backend: Backend,
    reference: ModelOutputs,
    framework_outputs: ModelOutputs,
    legal_mask: Tensor,
    fidelity_batches: tuple[TrainingBatch, ...],
    timing_states: Tensor,
    device: torch.device,
    strongly_typed: bool,
    constrain_float16_islands: bool,
) -> TensorRtMeasurement:
    onnx_path = output / f'{name}.onnx'
    engine_path = output / f'{name}.engine'
    _export_onnx(model, onnx_path, device)
    if strongly_typed and backend == Backend.TENSORRT_INT8:
        engine = _build_strongly_typed_engine(onnx_path, engine_path)
    elif constrain_float16_islands and backend == Backend.TENSORRT_INT8:
        engine = _build_precision_constrained_engine(onnx_path, engine_path)
    else:
        engine = _build_engine(onnx_path, engine_path, backend)[0]
    onnx_outputs = _onnx_outputs(onnx_path, fidelity_batches, device.index or 0)
    runner = _TensorRtCudaGraphRunner(engine_path, timing_states, device, 10)
    candidate = _tensorrt_outputs(runner, fidelity_batches)
    timing = _measure_runner(runner, 10, 5, 100, device)
    return TensorRtMeasurement(
        onnx_path=str(onnx_path),
        onnx_sha256=file_sha256(onnx_path),
        engine_path=engine.path,
        engine_sha256=engine.sha256,
        timing=timing,
        fidelity_to_floating=measure_fidelity(reference, candidate, legal_mask),
        onnx_fidelity_to_floating=measure_fidelity(reference, onnx_outputs, legal_mask),
        onnx_fidelity_to_framework=measure_fidelity(framework_outputs, onnx_outputs, legal_mask),
        tensorrt_fidelity_to_onnx=measure_fidelity(onnx_outputs, candidate, legal_mask),
    )


def _build_strongly_typed_engine(onnx_path: Path, engine_path: Path) -> ArtifactIdentity:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | (
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = tuple(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise ValueError(f'TensorRT ONNX conversion failed for {onnx_path}: {" | ".join(errors)}')
    configuration = builder.create_builder_config()
    configuration.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
    configuration.builder_optimization_level = 3
    serialized = builder.build_serialized_network(network, configuration)
    if serialized is None:
        raise ValueError(f'TensorRT failed to build a strongly typed engine for {onnx_path}.')
    write_bytes_atomically(engine_path, bytes(serialized))
    return ArtifactIdentity(path=str(engine_path), sha256=file_sha256(engine_path))


def _requires_float16_island(layer: trt.ILayer) -> bool:
    name = layer.name
    if 'start_block' in name or 'policy_head' in name or 'value_head' in name or 'global_pooling_bias' in name:
        return True
    return layer.type == trt.LayerType.ELEMENTWISE and '/backbone.' in name and name.endswith('/Add')


def _build_precision_constrained_engine(onnx_path: Path, engine_path: Path) -> ArtifactIdentity:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = tuple(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise ValueError(f'TensorRT ONNX conversion failed for {onnx_path}: {" | ".join(errors)}')
    constrained_types = {
        trt.LayerType.ACTIVATION,
        trt.LayerType.CONVOLUTION,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.MATRIX_MULTIPLY,
        trt.LayerType.REDUCE,
    }
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if layer.type not in constrained_types or not _requires_float16_island(layer):
            continue
        layer.precision = trt.float16
        for output_index in range(layer.num_outputs):
            layer.set_output_type(output_index, trt.float16)
    configuration = builder.create_builder_config()
    configuration.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
    configuration.builder_optimization_level = 3
    configuration.set_flag(trt.BuilderFlag.FP16)
    configuration.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    serialized = builder.build_serialized_network(network, configuration)
    if serialized is None:
        raise ValueError(f'TensorRT failed to build a precision-constrained engine for {onnx_path}.')
    write_bytes_atomically(engine_path, bytes(serialized))
    return ArtifactIdentity(path=str(engine_path), sha256=file_sha256(engine_path))


def _measure_torchscript_bfloat16(
    model: Network,
    architecture: NetworkParams,
    output: Path,
    timing_states: Tensor,
    device: torch.device,
) -> TimingDistribution:
    floating_model = Network(architecture, device, CHESS_NETWORK_DIMENSIONS)
    quantized_state = model.state_dict()
    floating_state = floating_model.state_dict()
    floating_model.load_state_dict({name: quantized_state[name] for name in floating_state})
    floating_model.eval()
    model_path = output / 'float-reference.jit.pt'
    torch.jit.save(torch.jit.script(floating_model), model_path)
    runner = _TorchScriptCudaGraphRunner(model_path, timing_states, device, 10)
    return _measure_runner(runner, 10, 5, 100, device)


def run(arguments: Arguments) -> ArchitectureScreenReport:
    arguments.output.mkdir(parents=True, exist_ok=True)
    dataset_input = ProductionReplayInput(
        kind='production_replay',
        path=arguments.replay_store,
        experiment=arguments.replay_experiment,
        orchestrator_recorded_sha256=arguments.replay_sha256,
    )
    opened = open_training_dataset(dataset_input)
    if not isinstance(opened, OpenedProductionReplay):
        raise AssertionError('The architecture screen requires a production replay store.')
    try:
        split = dataset_split(opened.row_count, arguments.holdout_fraction, 1.0)
        if arguments.final_fidelity_positions > split.held_out_row_count:
            raise ValueError('Final fidelity positions exceed the untouched holdout.')
        device = torch.device('cuda', arguments.device_id)
        torch.manual_seed(arguments.random_seed)
        torch.cuda.manual_seed_all(arguments.random_seed)
        network_architecture = architecture(
            arguments.cell,
            arguments.layers,
            arguments.hidden_size,
            arguments.final_normalization,
        )
        model = Network(network_architecture, device, CHESS_NETWORK_DIMENSIONS)
        cost = measure_model_cost(model)
        if (
            arguments.layers == DEFAULT_LAYERS
            and arguments.hidden_size == DEFAULT_HIDDEN_SIZE
            and not arguments.final_normalization
            and cost.parameters.total != EXPECTED_PARAMETER_COUNT
        ):
            raise ValueError(f'Expected {EXPECTED_PARAMETER_COUNT:,} parameters, found {cost.parameters.total:,}.')

        calibration_generator = np.random.default_rng(arguments.random_seed + 10_000_000)
        calibration_indices = np.sort(
            calibration_generator.choice(
                split.training_row_count,
                QAT_CALIBRATION_POSITIONS,
                replace=False,
            )
        )
        if arguments.cell.uses_qat:
            model = _configure_qat(
                model,
                opened,
                calibration_indices,
                arguments.batch_size,
                device,
                arguments.cell,
                arguments.layers,
                arguments.quantized_convolutions,
            )

        optimizer = create_student_optimizer(model, OptimizerKind.ADAMW, arguments.learning_rate)
        objective = distillation_objective()
        evaluation_batches = source_held_out_batches(
            opened,
            split.held_out_start_row,
            arguments.batch_size,
            device,
            (),
        )
        floor = mean_loss_values(tuple(achievable_loss_floor(batch, objective) for batch in evaluation_batches))
        generator = np.random.default_rng(arguments.random_seed)
        sampled_index_hash = hashlib.sha256()
        model.train()
        observations: list[TrainingObservation] = []
        recent_losses: list[LossValues] = []
        started = time.perf_counter()
        window_started = started
        window_samples = 0
        for step in range(1, arguments.steps + 1):
            learning_rate = learning_rate_at(
                step,
                arguments.steps,
                arguments.learning_rate,
                arguments.warmup_steps,
            )
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            indices = np.sort(generator.integers(0, split.training_row_count, size=arguments.batch_size))
            sampled_index_hash.update(indices.tobytes())
            batch = _replay_batch(opened, indices, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = objective.calculate_loss(model.training_output(batch.states), batch)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            recent_losses.append(observed_losses(loss))
            window_samples += arguments.batch_size
            if arguments.cell.uses_qat and not step % QAT_RECALIBRATION_INTERVAL:
                mtq.calibrate(
                    model,
                    'max',
                    _calibration_loop(opened, calibration_indices, arguments.batch_size, device),
                )
            if step % arguments.evaluate_every and step != arguments.steps:
                continue
            now = time.perf_counter()
            held_out = _evaluate(model, evaluation_batches, objective, device)
            observation = TrainingObservation(
                step=step,
                learning_rate=learning_rate,
                elapsed_seconds=now - started,
                samples_per_second=window_samples / (now - window_started),
                training=mean_loss_values(tuple(recent_losses)),
                held_out=held_out,
                held_out_policy_gap_above_floor=held_out.policy - floor.policy,
            )
            observations.append(observation)
            print(observation.model_dump_json(), flush=True)
            recent_losses.clear()
            window_started = now
            window_samples = 0
            state_path = arguments.output / 'latest-state.pt'
            torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, state_path)

        training_wall = time.perf_counter() - started
        if arguments.fold_post_activation_batch_norm:
            _fold_post_activation_batch_norm(model)
            mtq.calibrate(
                model,
                'max',
                _calibration_loop(opened, calibration_indices, arguments.batch_size, device),
            )
        state_path = arguments.output / 'final-state.pt'
        torch.save(model.state_dict(), state_path)
        activation_batch = _replay_batch(
            opened,
            np.arange(split.held_out_start_row, split.held_out_start_row + arguments.batch_size, dtype=np.int64),
            device,
        )
        activation_ranges = _activation_ranges(model, activation_batch, device)
        fidelity_batches = _held_out_batches(
            opened,
            split.held_out_start_row,
            arguments.final_fidelity_positions,
            TENSORRT_BATCH_SIZE,
            device,
        )
        torchscript_timing = _measure_torchscript_bfloat16(
            model,
            network_architecture,
            arguments.output,
            activation_batch.states[:TENSORRT_BATCH_SIZE],
            device,
        )
        if arguments.cell.uses_qat:
            with _quantizers_disabled(model):
                floating_outputs, legal_mask = _model_outputs(model, fidelity_batches, device)
                floating_loss = _evaluate(model, evaluation_batches, objective, device)
                float16_measurement = _measure_tensorrt(
                    model,
                    arguments.output,
                    'float16',
                    Backend.TENSORRT_FLOAT16,
                    floating_outputs,
                    floating_outputs,
                    legal_mask,
                    fidelity_batches,
                    activation_batch.states[:TENSORRT_BATCH_SIZE],
                    device,
                    arguments.strongly_typed_tensorrt,
                    arguments.constrain_tensorrt_float16_islands,
                )
            fake_outputs, _ = _model_outputs(model, fidelity_batches, device)
            fake_loss = _evaluate(model, evaluation_batches, objective, device)
            int8_measurement = _measure_tensorrt(
                model,
                arguments.output,
                'int8',
                Backend.TENSORRT_INT8,
                floating_outputs,
                fake_outputs,
                legal_mask,
                fidelity_batches,
                activation_batch.states[:TENSORRT_BATCH_SIZE],
                device,
                arguments.strongly_typed_tensorrt,
                arguments.constrain_tensorrt_float16_islands,
            )
            float_to_fake = measure_fidelity(floating_outputs, fake_outputs, legal_mask)
        else:
            floating_outputs, legal_mask = _model_outputs(model, fidelity_batches, device)
            floating_loss = _evaluate(model, evaluation_batches, objective, device)
            fake_loss = None
            float_to_fake = None
            int8_measurement = None
            float16_measurement = _measure_tensorrt(
                model,
                arguments.output,
                'float16',
                Backend.TENSORRT_FLOAT16,
                floating_outputs,
                floating_outputs,
                legal_mask,
                fidelity_batches,
                activation_batch.states[:TENSORRT_BATCH_SIZE],
                device,
                arguments.strongly_typed_tensorrt,
                arguments.constrain_tensorrt_float16_islands,
            )

        report = ArchitectureScreenReport(
            source_revision=read_source_revision(),
            cell=arguments.cell,
            random_seed=arguments.random_seed,
            device_id=arguments.device_id,
            replay=opened.snapshot,
            sampled_index_sequence_sha256=sampled_index_hash.hexdigest(),
            architecture=network_architecture,
            quantized_convolutions=arguments.quantized_convolutions if arguments.cell.uses_qat else 0,
            folded_post_activation_batch_norm=arguments.fold_post_activation_batch_norm,
            strongly_typed_tensorrt=arguments.strongly_typed_tensorrt,
            constrained_tensorrt_float16_islands=arguments.constrain_tensorrt_float16_islands,
            model_cost=cost,
            steps=arguments.steps,
            batch_size=arguments.batch_size,
            peak_learning_rate=arguments.learning_rate,
            warmup_steps=arguments.warmup_steps,
            holdout_fraction=arguments.holdout_fraction,
            held_out_positions=split.held_out_row_count,
            held_out_loss_floor=floor,
            observations=tuple(observations),
            training_wall_seconds=training_wall,
            average_training_samples_per_second=(
                arguments.steps * arguments.batch_size / training_wall if arguments.steps else None
            ),
            qat_recalibration_interval=QAT_RECALIBRATION_INTERVAL if arguments.cell.uses_qat else None,
            qat_calibration_positions=QAT_CALIBRATION_POSITIONS if arguments.cell.uses_qat else None,
            activation_ranges=activation_ranges,
            floating_held_out_loss=floating_loss,
            fake_quant_held_out_loss=fake_loss,
            floating_to_fake_quant_fidelity=float_to_fake,
            torchscript_bfloat16_timing=torchscript_timing,
            tensorrt_float16=float16_measurement,
            tensorrt_int8=int8_measurement,
            state_dict_path=str(state_path),
            state_dict_sha256=file_sha256(state_path),
        )
        write_text_atomically(arguments.output / 'report.json', report.model_dump_json(indent=2) + '\n')
        return report
    finally:
        close_training_dataset(opened)


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Run one arm of the frozen-replay INT8 architecture screen.')
    parser.add_argument('--replay-store', required=True, type=Path)
    parser.add_argument('--replay-experiment', required=True, type=Path)
    parser.add_argument('--replay-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cell', required=True, choices=tuple(cell.value for cell in ScreenCell))
    parser.add_argument('--random-seed', required=True, type=int)
    parser.add_argument('--device-id', required=True, type=int)
    parser.add_argument('--steps', default=100_000, type=int)
    parser.add_argument('--batch-size', default=1_024, type=int)
    parser.add_argument('--learning-rate', default=0.002, type=float)
    parser.add_argument('--warmup-steps', default=200, type=int)
    parser.add_argument('--evaluate-every', default=1_000, type=int)
    parser.add_argument('--holdout-fraction', default=0.02, type=float)
    parser.add_argument('--final-fidelity-positions', default=FINAL_FIDELITY_POSITIONS, type=int)
    parser.add_argument('--layers', default=DEFAULT_LAYERS, type=int)
    parser.add_argument('--hidden-size', default=DEFAULT_HIDDEN_SIZE, type=int)
    parser.add_argument('--quantized-convolutions', default=DEFAULT_LAYERS * 2, type=int)
    parser.add_argument('--final-normalization', action='store_true')
    parser.add_argument('--fold-post-activation-batch-norm', action='store_true')
    parser.add_argument('--strongly-typed-tensorrt', action='store_true')
    parser.add_argument('--constrain-tensorrt-float16-islands', action='store_true')
    namespace = parser.parse_args()
    if namespace.steps < 0 or namespace.batch_size <= 0 or namespace.evaluate_every <= 0:
        raise ValueError('Steps must be nonnegative; batch size and evaluation interval must be positive.')
    if namespace.layers <= 0 or namespace.hidden_size <= 0:
        raise ValueError('Layers and hidden size must be positive.')
    if namespace.quantized_convolutions <= 0 or namespace.quantized_convolutions > namespace.layers * 2:
        raise ValueError('Quantized convolutions must be between one and twice the residual-block count.')
    if namespace.final_normalization and not ScreenCell(namespace.cell).uses_quantization_friendly_trunk:
        raise ValueError('Final normalization is defined only for the scaled pre-activation trunk.')
    if namespace.fold_post_activation_batch_norm and ScreenCell(namespace.cell) not in (
        ScreenCell.POST_QAT,
        ScreenCell.POST_SCALED_QAT,
    ):
        raise ValueError('Batch-normalization folding is defined only for post-activation QAT cells.')
    return Arguments(
        replay_store=namespace.replay_store,
        replay_experiment=namespace.replay_experiment,
        replay_sha256=namespace.replay_sha256,
        output=namespace.output,
        cell=ScreenCell(namespace.cell),
        random_seed=namespace.random_seed,
        device_id=namespace.device_id,
        steps=namespace.steps,
        batch_size=namespace.batch_size,
        learning_rate=namespace.learning_rate,
        warmup_steps=namespace.warmup_steps,
        evaluate_every=namespace.evaluate_every,
        holdout_fraction=namespace.holdout_fraction,
        final_fidelity_positions=namespace.final_fidelity_positions,
        layers=namespace.layers,
        hidden_size=namespace.hidden_size,
        quantized_convolutions=namespace.quantized_convolutions,
        final_normalization=namespace.final_normalization,
        fold_post_activation_batch_norm=namespace.fold_post_activation_batch_norm,
        strongly_typed_tensorrt=namespace.strongly_typed_tensorrt,
        constrain_tensorrt_float16_islands=namespace.constrain_tensorrt_float16_islands,
    )


def main() -> None:
    report = run(parse_arguments())
    print(report.model_dump_json(indent=2))


if __name__ == '__main__':
    main()

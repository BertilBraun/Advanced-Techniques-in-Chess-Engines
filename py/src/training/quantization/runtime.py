from __future__ import annotations

import copy
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import modelopt.torch.opt as modelopt
import modelopt.torch.quantization as quantization
import numpy as np
import onnx
import torch
from modelopt.torch.quantization.config import QuantizeConfig, QuantizerCfgEntry
from modelopt.torch.quantization.nn import TensorQuantizer
from pydantic import Field
from src.self_play.tensorrt_refit import canonicalize_onnx_refit_names
from src.training.network import (
    GlobalPoolingResBlock,
    Network,
    ResBlock,
    ScaledPostActivationGlobalPoolingResBlock,
    ScaledPostActivationResBlock,
)
from src.training.quantization.configuration import (
    QatCheckpointPhase,
    QatStateIdentity,
    TensorRtInt8QatConfiguration,
    expected_qat_phase,
)
from src.util.atomic_file import write_bytes_atomically
from src.util.frozen_model import ConfigurationPath, FrozenModel
from src.util.hashing import file_sha256
from torch import Tensor, nn

CalibrationLoop = Callable[[nn.Module], None]


class QatOnnxArtifact(FrozenModel):
    path: ConfigurationPath
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    quantize_linear_nodes: int = Field(gt=0)
    dequantize_linear_nodes: int = Field(gt=0)


class FloatOnnxArtifact(FrozenModel):
    path: ConfigurationPath
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


@dataclass(frozen=True)
class RestoredQatModel:
    model: Network
    phase: QatCheckpointPhase


@dataclass(frozen=True)
class _SeededBatchNormTensor:
    name: str
    tensor: Tensor
    original_value: Tensor


def fixed_batch_example_states(states: Tensor, batch_size: int) -> Tensor:
    if states.shape[0] <= 0:
        raise ValueError('QAT deployment export requires at least one calibration position.')
    repetitions = (batch_size + states.shape[0] - 1) // states.shape[0]
    return states.repeat((repetitions, 1, 1, 1))[:batch_size]


def _qat_configuration() -> QuantizeConfig:
    configuration = QuantizeConfig.model_validate(copy.deepcopy(quantization.INT8_DEFAULT_CFG))
    configuration.quant_cfg.extend(
        (
            QuantizerCfgEntry(quantizer_name='*', parent_class='nn.Linear', enable=False),
            QuantizerCfgEntry(quantizer_name='*start_block*', enable=False),
            QuantizerCfgEntry(quantizer_name='*policy_head*', enable=False),
            QuantizerCfgEntry(quantizer_name='*value_head*', enable=False),
        )
    )
    return configuration


def configure_qat(model: Network, calibration_loop: CalibrationLoop) -> Network:
    configured = quantization.quantize(model, _qat_configuration(), calibration_loop)
    if not isinstance(configured, Network):
        raise ValueError('ModelOpt returned a model that does not preserve the training network contract.')
    return configured


def recalibrate_qat(model: Network, calibration_loop: CalibrationLoop) -> None:
    quantization.calibrate(model, 'max', calibration_loop)


def _fold_convolution_batch_norm(block: nn.Sequential) -> None:
    convolution = block[0]
    batch_norm = block[1]
    if not isinstance(convolution, nn.Conv2d) or not isinstance(batch_norm, nn.BatchNorm2d):
        raise ValueError('QAT deployment folding requires Conv2d-BatchNorm2d residual branches.')
    block[0] = torch.nn.utils.fusion.fuse_conv_bn_eval(convolution, batch_norm)
    block[1] = nn.Identity()


def fold_post_activation_batch_norm(model: Network) -> None:
    model.eval()
    for block in model.backbone:
        if not isinstance(
            block,
            (ResBlock, GlobalPoolingResBlock, ScaledPostActivationResBlock, ScaledPostActivationGlobalPoolingResBlock),
        ):
            raise ValueError('QAT deployment folding requires a post-activation residual trunk.')
        if isinstance(block, (ScaledPostActivationResBlock, ScaledPostActivationGlobalPoolingResBlock)):
            second_batch_norm = block.conv_block2[1]
            if not isinstance(second_batch_norm, nn.BatchNorm2d):
                raise ValueError('The scaled residual branch has already been folded.')
            if second_batch_norm.weight is None or second_batch_norm.bias is None:
                raise ValueError('The scaled residual BatchNorm must have affine parameters.')
            with torch.no_grad():
                second_batch_norm.weight.mul_(block.branch_scale)
                second_batch_norm.bias.mul_(block.branch_scale)
            block.branch_scale = 1.0
        _fold_convolution_batch_norm(block.conv_block1)
        _fold_convolution_batch_norm(block.conv_block2)
    model.train()


def save_qat_state(
    model: Network,
    path: Path,
    completed_optimizer_steps: int,
) -> QatStateIdentity:
    temporary_path = path.with_name(f'.{path.name}.modelopt')
    torch.save(modelopt.modelopt_state(model), temporary_path)
    write_bytes_atomically(path, temporary_path.read_bytes())
    temporary_path.unlink(missing_ok=True)
    return QatStateIdentity(
        phase=QatCheckpointPhase.PRE_FOLD,
        completed_optimizer_steps=completed_optimizer_steps,
        path=path,
        sha256=file_sha256(path),
    )


def deployment_qat_state(
    pre_fold_state: QatStateIdentity,
    completed_optimizer_steps: int,
) -> QatStateIdentity:
    if pre_fold_state.phase is not QatCheckpointPhase.PRE_FOLD:
        raise ValueError('Deployment QAT must retain the ModelOpt conversion state captured before folding.')
    return QatStateIdentity(
        phase=QatCheckpointPhase.DEPLOYMENT,
        completed_optimizer_steps=completed_optimizer_steps,
        path=pre_fold_state.path,
        sha256=pre_fold_state.sha256,
    )


def restore_qat_model(
    model: Network,
    state: QatStateIdentity,
    configuration: TensorRtInt8QatConfiguration,
) -> RestoredQatModel:
    expected_phase = expected_qat_phase(configuration, state.completed_optimizer_steps)
    if state.phase is not expected_phase:
        raise ValueError(f'QAT checkpoint phase {state.phase} does not match expected phase {expected_phase}.')
    if file_sha256(state.path) != state.sha256:
        raise ValueError(f'QAT state hash does not match: {state.path}')
    restored = modelopt.restore_from_modelopt_state(model, modelopt_state_path=state.path)
    if not isinstance(restored, Network):
        raise ValueError('ModelOpt restore did not preserve the training network contract.')
    if state.phase is QatCheckpointPhase.DEPLOYMENT:
        fold_post_activation_batch_norm(restored)
    return RestoredQatModel(restored, state.phase)


@contextmanager
def quantizers_disabled(model: Network) -> Iterator[None]:
    quantizers = tuple(module for module in model.modules() if isinstance(module, TensorQuantizer))
    enabled = tuple(quantizer.is_enabled for quantizer in quantizers)
    try:
        for quantizer in quantizers:
            quantizer.disable()
        yield
    finally:
        for quantizer, was_enabled in zip(quantizers, enabled, strict=True):
            if was_enabled:
                quantizer.enable()
            else:
                quantizer.disable()


def _explicit_batch_norm_tensor_names(exported: onnx.ModelProto) -> frozenset[str]:
    return frozenset(
        tensor_name
        for node in exported.graph.node
        if node.op_type == 'BatchNormalization'
        for tensor_name in node.input[1:5]
    )


@contextmanager
def _seed_batch_norm_tensors_for_onnx_export(
    model: nn.Module,
    tensor_names: frozenset[str],
) -> Iterator[tuple[_SeededBatchNormTensor, ...]]:
    seeded_tensors: list[_SeededBatchNormTensor] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        if module.weight is None or module.bias is None or module.running_mean is None or module.running_var is None:
            raise ValueError('QAT ONNX export requires affine BatchNorm2d modules with tracked running statistics.')
        named_tensors = (
            ('weight', module.weight, 0.875),
            ('bias', module.bias, 0.03125),
            ('running_mean', module.running_mean, 0.0625),
            ('running_var', module.running_var, 1.125),
        )
        for tensor_name, tensor, sentinel in named_tensors:
            qualified_name = f'{module_name}.{tensor_name}'
            if qualified_name not in tensor_names:
                continue
            unique_sentinel = sentinel + (len(seeded_tensors) + 1) / 4096
            seeded_tensors.append(
                _SeededBatchNormTensor(
                    name=qualified_name,
                    tensor=tensor,
                    original_value=tensor.detach().clone(),
                )
            )
            tensor.fill_(unique_sentinel)
    seeded_names = frozenset(seeded_tensor.name for seeded_tensor in seeded_tensors)
    if seeded_names != tensor_names:
        missing_names = sorted(tensor_names - seeded_names)
        raise ValueError(f'QAT ONNX BatchNorm inputs do not map to model tensors: {missing_names}.')
    try:
        yield tuple(seeded_tensors)
    finally:
        for seeded_tensor in seeded_tensors:
            seeded_tensor.tensor.copy_(seeded_tensor.original_value)


def _restore_exported_batch_norm_tensors(
    exported: onnx.ModelProto,
    seeded_tensors: tuple[_SeededBatchNormTensor, ...],
) -> None:
    initializers = {initializer.name: initializer for initializer in exported.graph.initializer}
    for seeded_tensor in seeded_tensors:
        initializer = initializers.get(seeded_tensor.name)
        if initializer is None:
            raise ValueError(f'QAT ONNX export omitted BatchNorm tensor {seeded_tensor.name}.')
        restored = np.ascontiguousarray(seeded_tensor.original_value.detach().cpu().numpy())
        initializer.CopyFrom(onnx.numpy_helper.from_array(restored, name=seeded_tensor.name))


def _export_onnx_graph(model: nn.Module, path: Path, example_states: Tensor) -> None:
    torch.onnx.export(
        model,
        (example_states,),
        str(path),
        input_names=('states',),
        output_names=('policy_logits', 'wdl_probabilities'),
        opset_version=20,
        do_constant_folding=True,
        dynamo=False,
    )


def _export_modelopt_onnx(model: nn.Module, path: Path, example_states: Tensor) -> onnx.ModelProto:
    temporary_path = path.with_name(f'.{path.name}.tmp')
    was_training = model.training
    model.eval()
    seeded_tensors: tuple[_SeededBatchNormTensor, ...] = ()
    try:
        with torch.inference_mode():
            _export_onnx_graph(model, temporary_path, example_states)
            probe = onnx.load(temporary_path)
            explicit_batch_norm_tensor_names = _explicit_batch_norm_tensor_names(probe)
            if explicit_batch_norm_tensor_names:
                with _seed_batch_norm_tensors_for_onnx_export(
                    model, explicit_batch_norm_tensor_names
                ) as seeded_tensors:
                    _export_onnx_graph(model, temporary_path, example_states)
    finally:
        model.train(was_training)
    exported = onnx.load(temporary_path)
    _restore_exported_batch_norm_tensors(exported, seeded_tensors)
    canonicalize_onnx_refit_names(exported)
    onnx.save(exported, temporary_path)
    onnx.checker.check_model(exported, full_check=True)
    temporary_path.replace(path)
    return exported


def export_qat_onnx(model: nn.Module, path: Path, example_states: Tensor) -> QatOnnxArtifact:
    exported = _export_modelopt_onnx(model, path, example_states)
    quantize_linear_nodes = sum(node.op_type == 'QuantizeLinear' for node in exported.graph.node)
    dequantize_linear_nodes = sum(node.op_type == 'DequantizeLinear' for node in exported.graph.node)
    if quantize_linear_nodes == 0 or dequantize_linear_nodes == 0:
        path.unlink()
        raise ValueError('QAT ONNX export contains no explicit Q/DQ nodes.')
    return QatOnnxArtifact(
        path=path,
        sha256=file_sha256(path),
        quantize_linear_nodes=quantize_linear_nodes,
        dequantize_linear_nodes=dequantize_linear_nodes,
    )


def export_float_qat_onnx(model: Network, path: Path, example_states: Tensor) -> FloatOnnxArtifact:
    with quantizers_disabled(model):
        exported = _export_modelopt_onnx(model, path, example_states)
    if any(node.op_type in ('QuantizeLinear', 'DequantizeLinear') for node in exported.graph.node):
        path.unlink()
        raise ValueError('Floating-point QAT deployment export unexpectedly contains Q/DQ nodes.')
    return FloatOnnxArtifact(path=path, sha256=file_sha256(path))


def _specialize_onnx_batch(source_path: Path, destination_path: Path, batch_size: int) -> onnx.ModelProto:
    if batch_size <= 0:
        raise ValueError('QAT deployment batch size must be positive.')
    exported = onnx.load(source_path)
    if len(exported.graph.input) != 1:
        raise ValueError('QAT ONNX batch specialization requires exactly one graph input.')
    input_shape = exported.graph.input[0].type.tensor_type.shape.dim
    if not input_shape or not input_shape[0].HasField('dim_value'):
        raise ValueError('QAT ONNX batch specialization requires a fixed source batch size.')
    source_batch_size = input_shape[0].dim_value
    if batch_size > source_batch_size:
        raise ValueError('QAT ONNX batch specialization cannot enlarge the exported batch.')

    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in exported.graph.node:
        for input_index, input_name in enumerate(node.input):
            consumers.setdefault(input_name, []).append((node, input_index))

    resized_batch_constants = 0
    resized_reshape_constants = 0
    for node in exported.graph.node:
        if node.op_type != 'Constant' or len(node.output) != 1:
            continue
        value_attribute = next((attribute for attribute in node.attribute if attribute.name == 'value'), None)
        if value_attribute is None:
            continue
        uses = consumers.get(node.output[0], [])
        value = onnx.numpy_helper.to_array(value_attribute.t)
        if uses and all(consumer.op_type == 'Reshape' and input_index == 1 for consumer, input_index in uses):
            if value.ndim == 1 and value.size > 0 and value[0] == source_batch_size:
                specialized_shape = value.copy()
                specialized_shape[0] = batch_size
                value_attribute.t.CopyFrom(onnx.numpy_helper.from_array(specialized_shape))
                resized_reshape_constants += 1
            continue
        if uses and all(consumer.op_type == 'ScatterElements' and input_index == 0 for consumer, input_index in uses):
            if value.ndim > 0 and value.shape[0] == source_batch_size:
                value_attribute.t.CopyFrom(onnx.numpy_helper.from_array(np.ascontiguousarray(value[:batch_size])))
                resized_batch_constants += 1

    if resized_reshape_constants == 0 or resized_batch_constants == 0:
        raise ValueError('QAT ONNX graph does not contain the expected fixed-batch policy-head constants.')
    for graph_value in (*exported.graph.input, *exported.graph.output):
        dimensions = graph_value.type.tensor_type.shape.dim
        if dimensions and dimensions[0].HasField('dim_value') and dimensions[0].dim_value == source_batch_size:
            dimensions[0].dim_value = batch_size

    onnx.checker.check_model(exported, full_check=True)
    write_bytes_atomically(destination_path, exported.SerializeToString())
    return exported


def specialize_qat_onnx_batch(source_path: Path, destination_path: Path, batch_size: int) -> QatOnnxArtifact:
    exported = _specialize_onnx_batch(source_path, destination_path, batch_size)
    quantize_linear_nodes = sum(node.op_type == 'QuantizeLinear' for node in exported.graph.node)
    dequantize_linear_nodes = sum(node.op_type == 'DequantizeLinear' for node in exported.graph.node)
    if quantize_linear_nodes == 0 or dequantize_linear_nodes == 0:
        destination_path.unlink()
        raise ValueError('Specialized QAT ONNX artifact contains no explicit Q/DQ nodes.')
    return QatOnnxArtifact(
        path=destination_path,
        sha256=file_sha256(destination_path),
        quantize_linear_nodes=quantize_linear_nodes,
        dequantize_linear_nodes=dequantize_linear_nodes,
    )


def specialize_float_onnx_batch(source_path: Path, destination_path: Path, batch_size: int) -> FloatOnnxArtifact:
    exported = _specialize_onnx_batch(source_path, destination_path, batch_size)
    if any(node.op_type in ('QuantizeLinear', 'DequantizeLinear') for node in exported.graph.node):
        destination_path.unlink()
        raise ValueError('Specialized floating-point ONNX artifact unexpectedly contains Q/DQ nodes.')
    return FloatOnnxArtifact(path=destination_path, sha256=file_sha256(destination_path))

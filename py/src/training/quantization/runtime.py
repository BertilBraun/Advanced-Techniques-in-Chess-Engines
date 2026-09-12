from __future__ import annotations

import copy
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import modelopt.torch.opt as modelopt
import modelopt.torch.quantization as quantization
import onnx
import torch
from modelopt.torch.quantization.config import QuantizeConfig, QuantizerCfgEntry
from modelopt.torch.quantization.nn import TensorQuantizer
from pydantic import Field
from src.training.network import (
    Network,
    ScaledPostActivationGlobalPoolingResBlock,
    ScaledPostActivationResBlock,
)
from src.training.quantization.configuration import (
    QatCheckpointPhase,
    TensorRtInt8QatConfiguration,
    expected_qat_phase,
)
from src.util.atomic_file import write_bytes_atomically
from src.util.frozen_model import ConfigurationPath, FrozenModel
from src.util.hashing import file_sha256
from torch import Tensor, nn

CalibrationLoop = Callable[[nn.Module], None]


class QatStateIdentity(FrozenModel):
    phase: QatCheckpointPhase
    completed_optimizer_steps: int = Field(ge=0)
    path: ConfigurationPath
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class QatOnnxArtifact(FrozenModel):
    path: ConfigurationPath
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    quantize_linear_nodes: int = Field(gt=0)
    dequantize_linear_nodes: int = Field(gt=0)


@dataclass(frozen=True)
class RestoredQatModel:
    model: Network
    phase: QatCheckpointPhase


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


def fold_scaled_post_activation_batch_norm(model: Network) -> None:
    model.eval()
    for block in model.backbone:
        if not isinstance(block, (ScaledPostActivationResBlock, ScaledPostActivationGlobalPoolingResBlock)):
            raise ValueError('QAT deployment folding requires a scaled post-activation residual trunk.')
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
        fold_scaled_post_activation_batch_norm(restored)
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


def export_qat_onnx(model: Network, path: Path, example_states: Tensor) -> QatOnnxArtifact:
    temporary_path = path.with_name(f'.{path.name}.tmp')
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example_states,),
            str(temporary_path),
            input_names=('states',),
            output_names=('policy_logits', 'wdl_probabilities'),
            opset_version=20,
            do_constant_folding=True,
            dynamo=False,
        )
    model.train(was_training)
    exported = onnx.load(temporary_path)
    onnx.checker.check_model(exported, full_check=True)
    quantize_linear_nodes = sum(node.op_type == 'QuantizeLinear' for node in exported.graph.node)
    dequantize_linear_nodes = sum(node.op_type == 'DequantizeLinear' for node in exported.graph.node)
    if quantize_linear_nodes == 0 or dequantize_linear_nodes == 0:
        temporary_path.unlink()
        raise ValueError('QAT ONNX export contains no explicit Q/DQ nodes.')
    temporary_path.replace(path)
    return QatOnnxArtifact(
        path=path,
        sha256=file_sha256(path),
        quantize_linear_nodes=quantize_linear_nodes,
        dequantize_linear_nodes=dequantize_linear_nodes,
    )

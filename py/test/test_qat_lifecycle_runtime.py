from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from src.games.representation import NetworkDimensions
from src.training.checkpoint.contracts import CheckpointReference, read_checkpoint_manifest
from src.training.checkpoint.paths import checkpoint_manifest_path
from src.training.checkpoint.persistence import create_optimizer
from src.training.configuration import AdamWOptimizerConfiguration
from src.training.network import (
    DensePolicyHeadConfiguration,
    DisabledResidualContext,
    Network,
    NetworkParams,
    ScaledPostActivationResBlock,
    ScaledPostActivationResidualBlockConfiguration,
)
from src.training.quantization.configuration import (
    QatCheckpointPhase,
    QatStateIdentity,
    TensorRtInt8QatConfiguration,
)
from src.util.hashing import file_sha256
from torch import nn

pytest.importorskip('modelopt.torch.quantization')

from src.training.quantization.checkpoint import (  # noqa: E402
    load_qat_model_and_optimizer,
    qat_inference_checkpoint_for_batch,
    save_qat_model_and_optimizer,
)
from src.training.quantization.runtime import (  # noqa: E402
    configure_qat,
    deployment_qat_state,
    export_qat_onnx,
    fixed_batch_example_states,
    fold_scaled_post_activation_batch_norm,
    recalibrate_qat,
    restore_qat_model,
    save_qat_state,
    specialize_qat_onnx_batch,
)


def test_fixed_batch_example_states_repeats_to_exact_deployment_batch() -> None:
    states = torch.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2)

    result = fixed_batch_example_states(states, 5)

    assert result.shape == (5, 2, 2, 2)
    assert torch.equal(result[:3], states)
    assert torch.equal(result[3:], states[:2])


def _fixed_batch_qat_onnx(path: Path, batch_size: int) -> None:
    shape = helper.make_node(
        'Constant', (), ('shape',), value=numpy_helper.from_array(np.array([batch_size, 4], dtype=np.int64))
    )
    reshape = helper.make_node('Reshape', ('states', 'shape'), ('reshaped',))
    scale = helper.make_node('Constant', (), ('scale',), value=numpy_helper.from_array(np.array(0.1, dtype=np.float32)))
    zero = helper.make_node('Constant', (), ('zero',), value=numpy_helper.from_array(np.array(0, dtype=np.int8)))
    quantize = helper.make_node('QuantizeLinear', ('reshaped', 'scale', 'zero'), ('quantized',))
    dequantize = helper.make_node('DequantizeLinear', ('quantized', 'scale', 'zero'), ('updates',))
    scatter_data = helper.make_node(
        'Constant',
        (),
        ('scatter_data',),
        value=numpy_helper.from_array(np.zeros((batch_size, 4), dtype=np.float32)),
    )
    indices = helper.make_node(
        'Constant',
        (),
        ('indices',),
        value=numpy_helper.from_array(np.zeros((batch_size, 4), dtype=np.int64)),
    )
    scatter = helper.make_node('ScatterElements', ('scatter_data', 'indices', 'updates'), ('policy_logits',), axis=1)
    unrelated = helper.make_node(
        'Constant', (), ('unrelated',), value=numpy_helper.from_array(np.array([batch_size], dtype=np.int64))
    )
    graph = helper.make_graph(
        (shape, reshape, scale, zero, quantize, dequantize, scatter_data, indices, scatter, unrelated),
        'fixed-batch-qat',
        (helper.make_tensor_value_info('states', TensorProto.FLOAT, (batch_size, 4)),),
        (helper.make_tensor_value_info('policy_logits', TensorProto.FLOAT, (batch_size, 4)),),
    )
    onnx.save(helper.make_model(graph, opset_imports=(helper.make_opsetid('', 20),)), path)


def test_qat_inference_batch_specialization_uses_only_retained_onnx(tmp_path: Path) -> None:
    source_path = tmp_path / 'model_9.int8.onnx'
    _fixed_batch_qat_onnx(source_path, 320)
    qat_state_path = tmp_path / 'missing-qat-state.pt'
    checkpoint = CheckpointReference(
        generation=9,
        manifest_path=tmp_path / 'checkpoint_9.json',
        model_path=tmp_path / 'missing-model.pt',
        optimizer_path=tmp_path / 'missing-optimizer.pt',
        inference_model_path=source_path,
        inference_model_sha256=file_sha256(source_path),
        qat_state=QatStateIdentity(
            phase=QatCheckpointPhase.DEPLOYMENT,
            completed_optimizer_steps=4500,
            path=qat_state_path,
            sha256='0' * 64,
        ),
    )

    specialized = qat_inference_checkpoint_for_batch(checkpoint, 64)

    assert specialized.inference_model_path.is_file()
    model = onnx.load(specialized.inference_model_path)
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 64
    assert model.graph.output[0].type.tensor_type.shape.dim[0].dim_value == 64
    constants = {
        node.output[0]: numpy_helper.to_array(
            next(attribute.t for attribute in node.attribute if attribute.name == 'value')
        )
        for node in model.graph.node
        if node.op_type == 'Constant'
    }
    assert constants['shape'].tolist() == [64, 4]
    assert constants['scatter_data'].shape == (64, 4)
    assert constants['unrelated'].tolist() == [320]


def test_qat_onnx_batch_specialization_rejects_enlargement(tmp_path: Path) -> None:
    source_path = tmp_path / 'model.int8.onnx'
    _fixed_batch_qat_onnx(source_path, 64)

    with pytest.raises(ValueError, match='cannot enlarge'):
        specialize_qat_onnx_batch(source_path, tmp_path / 'specialized.onnx', 320)


def _network(actions: int = 10) -> Network:
    return Network(
        NetworkParams(
            num_layers=2,
            hidden_size=16,
            residual_context=DisabledResidualContext(),
            residual_block=ScaledPostActivationResidualBlockConfiguration(
                branch_scale=2**-0.5,
                activation_cap=6.0,
            ),
            policy_head=DensePolicyHeadConfiguration(channels=2),
            num_value_channels=2,
            value_fc_size=16,
        ),
        torch.device('cpu'),
        NetworkDimensions(channels=8, rows=3, columns=3, actions=actions),
    )


def _calibrate(model: nn.Module) -> None:
    model(torch.randn((8, 8, 3, 3)))


@pytest.mark.integration
@pytest.mark.parametrize(
    ('phase', 'completed_optimizer_steps', 'expected_batch_norm'),
    (
        (QatCheckpointPhase.PRE_FOLD, 999, nn.BatchNorm2d),
        (QatCheckpointPhase.DEPLOYMENT, 1_000, nn.Identity),
    ),
)
def test_qat_resume_reconstructs_checkpoint_topology(
    tmp_path: Path,
    phase: QatCheckpointPhase,
    completed_optimizer_steps: int,
    expected_batch_norm: type[nn.Module],
) -> None:
    configured = configure_qat(_network(), _calibrate)
    state = save_qat_state(configured, tmp_path / 'modelopt-state.pt', min(completed_optimizer_steps, 999))
    if phase is QatCheckpointPhase.DEPLOYMENT:
        fold_scaled_post_activation_batch_norm(configured)
        state = deployment_qat_state(state, completed_optimizer_steps)
    weights = configured.state_dict()

    restored = restore_qat_model(
        _network(),
        state,
        TensorRtInt8QatConfiguration(deployment_learning_rate=0.02),
    )
    restored.model.load_state_dict(weights)

    assert restored.phase is phase
    assert all(isinstance(block, ScaledPostActivationResBlock) for block in restored.model.backbone)
    assert all(isinstance(block.conv_block1[1], expected_batch_norm) for block in restored.model.backbone)


@pytest.mark.integration
def test_folded_qat_export_contains_explicit_quantization(tmp_path: Path) -> None:
    model = configure_qat(_network(), _calibrate)
    fold_scaled_post_activation_batch_norm(model)
    recalibrate_qat(model, _calibrate)

    artifact = export_qat_onnx(model, tmp_path / 'model.onnx', torch.randn((8, 8, 3, 3)))

    assert artifact.quantize_linear_nodes > 0
    assert artifact.dequantize_linear_nodes > 0


@pytest.mark.integration
def test_deployment_qat_checkpoint_round_trip(tmp_path: Path) -> None:
    model = configure_qat(_network(), _calibrate)
    pre_fold_state = save_qat_state(model, tmp_path / 'modelopt-state.pt', 999)
    fold_scaled_post_activation_batch_norm(model)
    recalibrate_qat(model, _calibrate)
    deployment_state = deployment_qat_state(pre_fold_state, 1_000)
    optimizer_configuration = AdamWOptimizerConfiguration()
    save_qat_model_and_optimizer(
        model,
        create_optimizer(model, optimizer_configuration),
        generation=2,
        completed_optimizer_steps=1_000,
        save_folder=tmp_path,
        qat_state=deployment_state,
        example_states=torch.randn((8, 8, 3, 3)),
    )

    loaded, _, loaded_state = load_qat_model_and_optimizer(
        generation=2,
        network_configuration=model.network_args,
        optimizer_configuration=optimizer_configuration,
        quantization_configuration=TensorRtInt8QatConfiguration(deployment_learning_rate=0.02),
        device=torch.device('cpu'),
        save_folder=tmp_path,
        dimensions=NetworkDimensions(channels=8, rows=3, columns=3, actions=10),
    )

    assert loaded_state.phase is QatCheckpointPhase.DEPLOYMENT
    assert all(isinstance(block.conv_block1[1], nn.Identity) for block in loaded.backbone)


@pytest.mark.integration
def test_qat_checkpoint_load_normalizes_compiled_parameter_keys(tmp_path: Path) -> None:
    model = configure_qat(_network(actions=64), _calibrate)
    pre_fold_state = save_qat_state(model, tmp_path / 'modelopt-state.pt', 0)
    optimizer_configuration = AdamWOptimizerConfiguration()
    reference = save_qat_model_and_optimizer(
        model,
        create_optimizer(model, optimizer_configuration),
        generation=0,
        completed_optimizer_steps=0,
        save_folder=tmp_path,
        qat_state=pre_fold_state,
        example_states=torch.randn((8, 8, 3, 3)),
        bootstrap_probe_states=torch.randn((256, 8, 3, 3)),
    )
    weights = torch.load(reference.model_path, map_location='cpu', weights_only=True)
    torch.save({f'_orig_mod.{key}': value for key, value in weights.items()}, reference.model_path)
    manifest = read_checkpoint_manifest(0, tmp_path).model_copy(
        update={'model_sha256': file_sha256(reference.model_path)}
    )
    checkpoint_manifest_path(0, tmp_path).write_text(manifest.model_dump_json(indent=2) + '\n')

    loaded, _, _ = load_qat_model_and_optimizer(
        generation=0,
        network_configuration=model.network_args,
        optimizer_configuration=optimizer_configuration,
        quantization_configuration=TensorRtInt8QatConfiguration(deployment_learning_rate=0.02),
        device=torch.device('cpu'),
        save_folder=tmp_path,
        dimensions=NetworkDimensions(channels=8, rows=3, columns=3, actions=64),
    )

    assert set(loaded.state_dict()) == set(model.state_dict())


@pytest.mark.integration
def test_generation_zero_qat_checkpoint_calibrates_only_inference_copy(tmp_path: Path) -> None:
    model = configure_qat(_network(actions=64), _calibrate)
    state = save_qat_state(model, tmp_path / 'modelopt-state.pt', 0)
    optimizer_configuration = AdamWOptimizerConfiguration()
    policy_weights_before = model.policy_head[-1].weight.detach().clone()

    reference = save_qat_model_and_optimizer(
        model,
        create_optimizer(model, optimizer_configuration),
        generation=0,
        completed_optimizer_steps=0,
        save_folder=tmp_path,
        qat_state=state,
        example_states=torch.randn((8, 8, 3, 3)),
        bootstrap_probe_states=torch.randn((256, 8, 3, 3)),
    )

    assert torch.equal(model.policy_head[-1].weight, policy_weights_before)
    assert reference.inference_model_path.name.endswith('.jit.pt')
    torch.jit.load(str(reference.inference_model_path))

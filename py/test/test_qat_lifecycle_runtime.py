from __future__ import annotations

import math
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
from src.training.configuration import AdamWOptimizerConfiguration, BootstrapPolicyScaleApplication
from src.training.network import (
    DensePolicyHeadConfiguration,
    DisabledResidualContext,
    Network,
    NetworkParams,
    PostActivationResidualBlockConfiguration,
    ResBlock,
    ScaledPostActivationResBlock,
    ScaledPostActivationResidualBlockConfiguration,
)
from src.training.quantization.configuration import (
    QatCalibrationSource,
    QatCheckpointPhase,
    QatFoldingMode,
    TensorRtInt8QatConfiguration,
)
from src.util.hashing import file_sha256
from torch import nn

pytest.importorskip('modelopt.torch.quantization')

from src.training.policy_prior import scheduled_inference_policy_scale  # noqa: E402
from src.training.quantization.checkpoint import (  # noqa: E402
    load_qat_model_and_optimizer,
    onnx_inference_checkpoint_for_batch,
    save_qat_model_and_optimizer,
)
from src.training.quantization.runtime import (  # noqa: E402
    configure_qat,
    deployment_qat_state,
    export_qat_onnx,
    fixed_batch_example_states,
    fold_post_activation_batch_norm,
    quantizers_disabled,
    recalibrate_qat,
    restore_qat_model,
    save_qat_state,
    specialize_qat_onnx_batch,
)
from src.training.trainer.rank import _fixed_qat_probe_position_count  # noqa: E402


@pytest.mark.parametrize(
    ('generation', 'expected_scale'),
    ((0, 8.0), (5, math.sqrt(8.0)), (10, 1.0), (20, 1.0)),
)
def test_inference_policy_scale_fades_geometrically(
    generation: int,
    expected_scale: float,
) -> None:
    assert scheduled_inference_policy_scale(8.0, generation, 10) == pytest.approx(expected_scale)


def test_fixed_batch_example_states_repeats_to_exact_deployment_batch() -> None:
    states = torch.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2)

    result = fixed_batch_example_states(states, 5)

    assert result.shape == (5, 2, 2, 2)
    assert torch.equal(result[:3], states)
    assert torch.equal(result[3:], states[:2])


@pytest.mark.parametrize(
    ('calibration_source', 'expected_positions'),
    (
        (QatCalibrationSource.EVALUATION_DATASET, 10_000),
        (QatCalibrationSource.REPLAY, 516),
    ),
)
def test_fixed_qat_probe_positions_are_independent_from_replay_calibration_size(
    calibration_source: QatCalibrationSource,
    expected_positions: int,
) -> None:
    configuration = TensorRtInt8QatConfiguration(
        calibration_positions=10_000,
        calibration_source=calibration_source,
        deployment_learning_rate='inherit',
        deployment_warmup_optimizer_steps=0,
    )

    assert _fixed_qat_probe_position_count(configuration, 516) == expected_positions


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


def test_inference_batch_specialization_uses_only_retained_onnx(tmp_path: Path) -> None:
    source_path = tmp_path / 'model_9.int8.onnx'
    _fixed_batch_qat_onnx(source_path, 320)
    checkpoint = CheckpointReference(
        generation=9,
        manifest_path=tmp_path / 'checkpoint_9.json',
        model_path=tmp_path / 'missing-model.pt',
        optimizer_path=tmp_path / 'missing-optimizer.pt',
        inference_model_path=source_path,
        inference_model_sha256=file_sha256(source_path),
        qat_state=None,
    )

    specialized = onnx_inference_checkpoint_for_batch(checkpoint, 64)

    assert specialized.inference_model_path.is_file()
    assert specialized.inference_model_path.name.endswith('.int8.onnx')
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


def _unscaled_network(actions: int = 10) -> Network:
    return Network(
        NetworkParams(
            num_layers=2,
            hidden_size=16,
            residual_context=DisabledResidualContext(),
            residual_block=PostActivationResidualBlockConfiguration(),
            policy_head=DensePolicyHeadConfiguration(channels=2),
            num_value_channels=2,
            value_fc_size=16,
        ),
        torch.device('cpu'),
        NetworkDimensions(channels=8, rows=3, columns=3, actions=actions),
    )


@pytest.mark.integration
def test_unscaled_post_activation_qat_folds_batch_norm_without_changing_outputs() -> None:
    model = configure_qat(_unscaled_network(), _calibrate)
    inputs = torch.randn((8, 8, 3, 3))
    model.eval()
    with quantizers_disabled(model), torch.inference_mode():
        expected = model(inputs)

    fold_post_activation_batch_norm(model)
    model.eval()
    with quantizers_disabled(model), torch.inference_mode():
        actual = model(inputs)

    assert all(isinstance(block, ResBlock) for block in model.backbone)
    assert all(isinstance(block.conv_block1[1], nn.Identity) for block in model.backbone)
    assert torch.allclose(actual[0], expected[0], rtol=1e-4, atol=1e-5)
    assert torch.allclose(actual[1], expected[1], rtol=1e-4, atol=1e-5)


@pytest.mark.integration
def test_scaled_post_activation_qat_fold_preserves_outputs_and_parameter_trainability() -> None:
    model = configure_qat(_network(), _calibrate)
    inputs = torch.randn((8, 8, 3, 3))
    model.eval()
    with quantizers_disabled(model), torch.inference_mode():
        expected = model(inputs)

    fold_post_activation_batch_norm(model)
    model.eval()
    with quantizers_disabled(model), torch.inference_mode():
        actual = model(inputs)

    convolutions = tuple(
        convolution for block in model.backbone for convolution in (block.conv_block1[0], block.conv_block2[0])
    )
    assert all(isinstance(block, ScaledPostActivationResBlock) for block in model.backbone)
    assert all(isinstance(block.conv_block1[1], nn.Identity) for block in model.backbone)
    assert all(isinstance(block.conv_block2[1], nn.Identity) for block in model.backbone)
    assert all(convolution.weight.requires_grad for convolution in convolutions)
    assert all(convolution.bias is not None and convolution.bias.requires_grad for convolution in convolutions)
    assert torch.allclose(actual[0], expected[0], rtol=1e-4, atol=1e-5)
    assert torch.allclose(actual[1], expected[1], rtol=1e-4, atol=1e-5)


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
        fold_post_activation_batch_norm(configured)
        state = deployment_qat_state(state, completed_optimizer_steps)
    weights = configured.state_dict()

    restored = restore_qat_model(
        _network(),
        state,
        TensorRtInt8QatConfiguration(deployment_learning_rate=0.02, deployment_warmup_optimizer_steps=0),
    )
    restored.model.load_state_dict(weights)

    assert restored.phase is phase
    assert all(isinstance(block, ScaledPostActivationResBlock) for block in restored.model.backbone)
    assert all(isinstance(block.conv_block1[1], expected_batch_norm) for block in restored.model.backbone)


@pytest.mark.integration
def test_folded_qat_export_contains_explicit_quantization(tmp_path: Path) -> None:
    model = configure_qat(_network(), _calibrate)
    fold_post_activation_batch_norm(model)
    recalibrate_qat(model, _calibrate)

    artifact = export_qat_onnx(model, tmp_path / 'model.onnx', torch.randn((8, 8, 3, 3)))

    assert artifact.quantize_linear_nodes > 0
    assert artifact.dequantize_linear_nodes > 0


@pytest.mark.integration
def test_qat_export_keeps_zero_and_nonzero_batch_norm_parameters(tmp_path: Path) -> None:
    model = configure_qat(_network(), _calibrate)
    batch_norm = model.backbone[0].conv_block2[1]
    assert isinstance(batch_norm, nn.BatchNorm2d)
    assert batch_norm.bias is not None
    with torch.no_grad():
        batch_norm.bias.zero_()
    zero_path = tmp_path / 'zero.onnx'
    export_qat_onnx(model, zero_path, torch.randn((8, 8, 3, 3)))
    assert torch.count_nonzero(batch_norm.bias) == 0

    with torch.no_grad():
        batch_norm.bias.fill_(0.03125)
    nonzero_path = tmp_path / 'nonzero.onnx'
    export_qat_onnx(model, nonzero_path, torch.randn((8, 8, 3, 3)))

    zero_initializers = {initializer.name: initializer for initializer in onnx.load(zero_path).graph.initializer}
    nonzero_initializers = {initializer.name: initializer for initializer in onnx.load(nonzero_path).graph.initializer}

    assert zero_initializers.keys() == nonzero_initializers.keys()
    assert 'backbone.0.conv_block2.1.bias' in zero_initializers
    assert np.count_nonzero(numpy_helper.to_array(zero_initializers['backbone.0.conv_block2.1.bias'])) == 0
    assert np.allclose(
        numpy_helper.to_array(nonzero_initializers['backbone.0.conv_block2.1.bias']),
        0.03125,
    )


@pytest.mark.integration
def test_deployment_qat_checkpoint_round_trip(tmp_path: Path) -> None:
    model = configure_qat(_network(), _calibrate)
    pre_fold_state = save_qat_state(model, tmp_path / 'modelopt-state.pt', 999)
    fold_post_activation_batch_norm(model)
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
        quantization_configuration=TensorRtInt8QatConfiguration(
            deployment_learning_rate=0.02,
            deployment_warmup_optimizer_steps=0,
        ),
        device=torch.device('cpu'),
        save_folder=tmp_path,
        dimensions=NetworkDimensions(channels=8, rows=3, columns=3, actions=10),
    )

    assert loaded_state.phase is QatCheckpointPhase.DEPLOYMENT
    assert all(isinstance(block.conv_block1[1], nn.Identity) for block in loaded.backbone)


@pytest.mark.integration
def test_deployment_copy_qat_checkpoint_keeps_training_model_unfolded(tmp_path: Path) -> None:
    model = configure_qat(_network(), _calibrate)
    pre_fold_state = save_qat_state(model, tmp_path / 'modelopt-state.pt', 999)
    deployment_state = deployment_qat_state(pre_fold_state, 1_000)
    configuration = TensorRtInt8QatConfiguration(
        fold_after_optimizer_steps=1_000,
        int8_self_play_start_generation=10,
        deployment_learning_rate='inherit',
        deployment_warmup_optimizer_steps=0,
        folding_mode=QatFoldingMode.DEPLOYMENT_COPY,
    )
    original_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    checkpoint = save_qat_model_and_optimizer(
        model,
        create_optimizer(model, AdamWOptimizerConfiguration()),
        generation=2,
        completed_optimizer_steps=1_000,
        save_folder=tmp_path,
        qat_state=deployment_state,
        example_states=torch.randn((8, 8, 3, 3)),
        quantization_configuration=configuration,
    )

    assert all(isinstance(block.conv_block1[1], nn.BatchNorm2d) for block in model.backbone)
    assert all(torch.equal(model.state_dict()[name], tensor) for name, tensor in original_state.items())
    exported = onnx.load(checkpoint.inference_model_path)
    assert all(node.op_type != 'BatchNormalization' for node in exported.graph.node)

    loaded, _, loaded_state = load_qat_model_and_optimizer(
        generation=2,
        network_configuration=model.network_args,
        optimizer_configuration=AdamWOptimizerConfiguration(),
        quantization_configuration=configuration,
        device=torch.device('cpu'),
        save_folder=tmp_path,
        dimensions=NetworkDimensions(channels=8, rows=3, columns=3, actions=10),
    )

    assert loaded_state.phase is QatCheckpointPhase.DEPLOYMENT
    assert all(isinstance(block.conv_block1[1], nn.BatchNorm2d) for block in loaded.backbone)


@pytest.mark.integration
def test_qat_checkpoint_uses_float_export_before_configured_int8_generation(tmp_path: Path) -> None:
    model = configure_qat(_network(), _calibrate)
    state = save_qat_state(model, tmp_path / 'modelopt-state.pt', 500)
    configuration = TensorRtInt8QatConfiguration(
        int8_self_play_start_generation=10,
        deployment_learning_rate=0.02,
        deployment_warmup_optimizer_steps=0,
    )

    reference = save_qat_model_and_optimizer(
        model,
        create_optimizer(model, AdamWOptimizerConfiguration()),
        generation=5,
        completed_optimizer_steps=500,
        save_folder=tmp_path,
        qat_state=state,
        example_states=torch.randn((8, 8, 3, 3)),
        quantization_configuration=configuration,
    )

    assert reference.inference_model_path.name == 'model_5.fp16.onnx'
    exported = onnx.load(reference.inference_model_path)
    assert not any(node.op_type in ('QuantizeLinear', 'DequantizeLinear') for node in exported.graph.node)


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
        quantization_configuration=TensorRtInt8QatConfiguration(
            deployment_learning_rate=0.02,
            deployment_warmup_optimizer_steps=0,
        ),
        device=torch.device('cpu'),
        save_folder=tmp_path,
        dimensions=NetworkDimensions(channels=8, rows=3, columns=3, actions=64),
    )

    assert set(loaded.state_dict()) == set(model.state_dict())


@pytest.mark.integration
def test_generation_zero_qat_checkpoint_calibrates_trainable_and_inference_weights(tmp_path: Path) -> None:
    model = configure_qat(_network(actions=64), _calibrate)
    state = save_qat_state(model, tmp_path / 'modelopt-state.pt', 0)
    optimizer_configuration = AdamWOptimizerConfiguration()
    policy_weights_before = model.policy_head[-1].weight.detach().clone()

    target_top3_mass = 0.70
    reference = save_qat_model_and_optimizer(
        model,
        create_optimizer(model, optimizer_configuration),
        generation=0,
        completed_optimizer_steps=0,
        save_folder=tmp_path,
        qat_state=state,
        example_states=torch.randn((8, 8, 3, 3)),
        bootstrap_with_torchscript=True,
        bootstrap_probe_states=torch.randn((256, 8, 3, 3)),
        bootstrap_policy_prior_target_top3_mass=target_top3_mass,
    )

    calibration = read_checkpoint_manifest(0, tmp_path).policy_prior_calibration
    assert calibration is not None
    assert torch.allclose(model.policy_head[-1].weight, policy_weights_before * calibration.applied_scale)
    saved_weights = torch.load(reference.model_path, map_location='cpu', weights_only=True)
    assert torch.equal(saved_weights['policy_head.4.weight'], model.policy_head[-1].weight)
    assert reference.inference_model_path.name.endswith('.jit.pt')
    assert calibration.target_top3_mass == target_top3_mass
    assert calibration.calibrated_top3_mass == pytest.approx(target_top3_mass, abs=1e-6)
    torch.jit.load(str(reference.inference_model_path))


@pytest.mark.integration
def test_generation_zero_qat_checkpoint_can_scale_only_inference_weights(tmp_path: Path) -> None:
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
        bootstrap_policy_prior_target_top3_mass=0.70,
        quantization_configuration=TensorRtInt8QatConfiguration(
            deployment_learning_rate=0.02,
            deployment_warmup_optimizer_steps=0,
            int8_self_play_start_generation=10,
        ),
        bootstrap_policy_scale_application=BootstrapPolicyScaleApplication.INFERENCE_ONLY,
        bootstrap_policy_scale_fade_generations=10,
    )

    calibration = read_checkpoint_manifest(0, tmp_path).policy_prior_calibration
    assert calibration is not None
    assert torch.equal(model.policy_head[-1].weight, policy_weights_before)
    saved_weights = torch.load(reference.model_path, map_location='cpu', weights_only=True)
    assert torch.equal(saved_weights['policy_head.4.weight'], policy_weights_before)
    assert reference.inference_model_path.name.endswith('.fp16.onnx')
    assert calibration.calibrated_top3_mass == pytest.approx(0.70, abs=1e-6)


@pytest.mark.integration
def test_inference_only_policy_fade_continues_inside_progressive_model_directory(tmp_path: Path) -> None:
    model = configure_qat(_network(actions=64), _calibrate)
    optimizer = create_optimizer(model, AdamWOptimizerConfiguration())
    state = save_qat_state(model, tmp_path / 'modelopt-state.pt', 0)
    probe_states = torch.randn((256, 8, 3, 3))
    configuration = TensorRtInt8QatConfiguration(
        deployment_learning_rate=0.02,
        deployment_warmup_optimizer_steps=0,
        int8_self_play_start_generation=10,
    )
    root_reference = save_qat_model_and_optimizer(
        model,
        optimizer,
        generation=0,
        completed_optimizer_steps=0,
        save_folder=tmp_path,
        qat_state=state,
        example_states=torch.randn((8, 8, 3, 3)),
        bootstrap_probe_states=probe_states,
        bootstrap_policy_prior_target_top3_mass=0.70,
        quantization_configuration=configuration,
        bootstrap_policy_scale_application=BootstrapPolicyScaleApplication.INFERENCE_ONLY,
        bootstrap_policy_scale_fade_generations=10,
    )
    initial_record = read_checkpoint_manifest(0, tmp_path).policy_prior_calibration
    assert initial_record is not None
    progressive_directory = tmp_path / 'models' / 'stage-0'
    progressive_directory.mkdir(parents=True)

    save_qat_model_and_optimizer(
        model,
        optimizer,
        generation=1,
        completed_optimizer_steps=0,
        save_folder=progressive_directory,
        qat_state=state,
        example_states=torch.randn((8, 8, 3, 3)),
        bootstrap_probe_states=probe_states,
        bootstrap_policy_prior_target_top3_mass=0.70,
        quantization_configuration=configuration,
        bootstrap_policy_prior=initial_record,
        bootstrap_policy_scale_application=BootstrapPolicyScaleApplication.INFERENCE_ONLY,
        bootstrap_policy_scale_fade_generations=10,
    )
    generation_one = read_checkpoint_manifest(1, progressive_directory).policy_prior_calibration
    assert generation_one is not None

    generation_two_reference = save_qat_model_and_optimizer(
        model,
        optimizer,
        generation=2,
        completed_optimizer_steps=0,
        save_folder=progressive_directory,
        qat_state=state,
        example_states=torch.randn((8, 8, 3, 3)),
        bootstrap_probe_states=probe_states,
        bootstrap_policy_prior_target_top3_mass=0.70,
        quantization_configuration=configuration,
        bootstrap_policy_scale_application=BootstrapPolicyScaleApplication.INFERENCE_ONLY,
        bootstrap_policy_scale_fade_generations=10,
    )

    generation_two = read_checkpoint_manifest(2, progressive_directory).policy_prior_calibration
    assert generation_two is not None
    assert generation_two_reference.inference_model_path.name.endswith('.fp16.onnx')
    assert generation_one.initial_applied_scale == initial_record.applied_scale
    assert generation_two.initial_applied_scale == initial_record.applied_scale
    assert generation_two.applied_scale <= generation_one.applied_scale
    assert root_reference.manifest_path.parent == tmp_path

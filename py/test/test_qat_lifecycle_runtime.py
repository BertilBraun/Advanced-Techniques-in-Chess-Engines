from __future__ import annotations

from pathlib import Path

import pytest
import torch
from src.games.representation import NetworkDimensions
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
from src.training.quantization.configuration import QatCheckpointPhase, TensorRtInt8QatConfiguration
from torch import nn

pytest.importorskip('modelopt.torch.quantization')

from src.training.quantization.checkpoint import (  # noqa: E402
    load_qat_model_and_optimizer,
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
)


def test_fixed_batch_example_states_repeats_to_exact_deployment_batch() -> None:
    states = torch.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2)

    result = fixed_batch_example_states(states, 5)

    assert result.shape == (5, 2, 2, 2)
    assert torch.equal(result[:3], states)
    assert torch.equal(result[3:], states[:2])


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

    restored = restore_qat_model(_network(), state, TensorRtInt8QatConfiguration())
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
        quantization_configuration=TensorRtInt8QatConfiguration(),
        device=torch.device('cpu'),
        save_folder=tmp_path,
        dimensions=NetworkDimensions(channels=8, rows=3, columns=3, actions=10),
    )

    assert loaded_state.phase is QatCheckpointPhase.DEPLOYMENT
    assert all(isinstance(block.conv_block1[1], nn.Identity) for block in loaded.backbone)


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

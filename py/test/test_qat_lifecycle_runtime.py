from __future__ import annotations

from pathlib import Path

import pytest
import torch
from src.games.representation import NetworkDimensions
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

from src.training.quantization.runtime import (  # noqa: E402
    configure_qat,
    deployment_qat_state,
    fold_scaled_post_activation_batch_norm,
    restore_qat_model,
    save_qat_state,
)


def _network() -> Network:
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
        NetworkDimensions(channels=8, rows=3, columns=3, actions=10),
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

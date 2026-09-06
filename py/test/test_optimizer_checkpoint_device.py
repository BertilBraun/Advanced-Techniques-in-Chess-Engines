from __future__ import annotations

from pathlib import Path

import pytest
import torch
from src.games.representation import NetworkDimensions
from src.training.checkpoint.persistence import (
    create_optimizer,
    load_optimizer,
)
from src.training.network import (
    DisabledResidualContext,
    GoPointPassPolicyHeadConfiguration,
    Network,
    NetworkParams,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable.')
def test_loaded_adamw_step_counters_stay_on_cpu(tmp_path: Path) -> None:
    device = torch.device('cuda', 0)
    parameters = NetworkParams(
        num_layers=1,
        hidden_size=8,
        residual_context=DisabledResidualContext(),
        policy_head=GoPointPassPolicyHeadConfiguration(),
        num_value_channels=2,
        value_fc_size=8,
    )
    dimensions = NetworkDimensions(channels=3, rows=3, columns=3, actions=10)
    model = Network(parameters, device, dimensions)
    optimizer = create_optimizer(model, 'adamw')
    model.training_output(torch.zeros((2, 3, 3, 3), device=device)).policy_logits.sum().backward()
    optimizer.step()
    assert all(state['step'].device.type == 'cpu' for state in optimizer.state.values())

    path = tmp_path / 'optimizer.pt'
    torch.save(optimizer.state_dict(), path)
    loaded = load_optimizer(path, model, 'adamw', device)

    for state in loaded.state.values():
        assert state['step'].device.type == 'cpu'
        assert state['exp_avg'].device == device

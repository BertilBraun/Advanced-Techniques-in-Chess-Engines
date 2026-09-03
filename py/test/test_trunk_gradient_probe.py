from __future__ import annotations

import pytest
import torch
from src.games.representation import NetworkDimensions
from src.training.network import (
    DisabledResidualContext,
    GoPointPassPolicyHeadConfiguration,
    Network,
    NetworkParams,
)
from src.training.trainer.rank import _trunk_parameters


def _network() -> Network:
    parameters = NetworkParams(
        num_layers=2,
        hidden_size=8,
        residual_context=DisabledResidualContext(),
        policy_head=GoPointPassPolicyHeadConfiguration(),
        num_value_channels=2,
        value_fc_size=8,
    )
    dimensions = NetworkDimensions(channels=3, rows=3, columns=3, actions=10)
    return Network(parameters, torch.device('cpu'), dimensions)


def test_trunk_parameters_exclude_the_heads() -> None:
    network = _network()
    selected = {id(parameter) for parameter in _trunk_parameters(network)}
    head_parameters = {
        id(parameter) for module in (network.policy_head, network.value_head) for parameter in module.parameters()
    }
    assert selected
    assert not (selected & head_parameters)


def test_trunk_parameters_cover_every_trunk_module() -> None:
    network = _network()
    selected = {id(parameter) for parameter in _trunk_parameters(network)}
    for module in (network.start_block, network.backbone, network.finish_block):
        for parameter in module.parameters():
            assert id(parameter) in selected


def test_trunk_parameters_reject_a_renamed_trunk() -> None:
    """The probe must fail loudly rather than report zeros if the trunk is renamed."""

    class Renamed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = torch.nn.Linear(4, 4)

    with pytest.raises(RuntimeError, match='no trunk parameters'):
        _trunk_parameters(Renamed())


def test_each_loss_term_produces_a_distinct_nonzero_trunk_gradient() -> None:
    network = _network()
    trunk = _trunk_parameters(network)
    torch.manual_seed(0)
    output = network.training_output(torch.randn((4, 3, 3, 3)))
    norms = []
    for logits in (output.policy_logits, output.wdl_logits):
        gradients = torch.autograd.grad(logits.square().mean(), trunk, retain_graph=True, allow_unused=False)
        norms.append(float(torch.sqrt(sum(gradient.pow(2).sum() for gradient in gradients))))
    assert all(norm > 0.0 for norm in norms)
    assert norms[0] != norms[1]

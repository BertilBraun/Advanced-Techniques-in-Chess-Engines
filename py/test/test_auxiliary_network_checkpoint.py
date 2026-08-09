from pathlib import Path

import torch

from src.games.representation import NetworkDimensions
from src.training.network import Network, NetworkParams, SEPlacement
from src.training.checkpoint.persistence import create_optimizer, save_model_and_optimizer


def test_training_model_keeps_auxiliary_heads_but_jit_inference_model_trims_them(tmp_path: Path) -> None:
    parameters = NetworkParams(
        num_layers=1,
        hidden_size=8,
        se_placement=SEPlacement.DISABLED,
        num_policy_channels=2,
        num_value_channels=2,
        value_fc_size=8,
    )
    dimensions = NetworkDimensions(channels=3, rows=3, columns=3, actions=10)
    model = Network(parameters, torch.device('cpu'), dimensions, auxiliary_action_sizes=(10,))
    output = model.training_output(torch.zeros((2, 3, 3, 3)))

    assert output.policy_logits.shape == (2, 10)
    assert output.wdl_logits.shape == (2, 3)
    assert output.auxiliary_logits[0].shape == (2, 10)

    save_model_and_optimizer(model, create_optimizer(model, 'adamw'), 1, tmp_path)
    training_state = torch.load(tmp_path / 'model_1.pt', map_location='cpu', weights_only=True)
    inference_model = torch.jit.load(str(tmp_path / 'model_1.jit.pt'), map_location='cpu')

    assert any(name.startswith('auxiliaryHeads.') for name in training_state)
    assert all(not name.startswith('auxiliaryHeads.') for name, _ in inference_model.named_parameters())

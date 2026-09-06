from __future__ import annotations

from pathlib import Path

import pytest
import torch
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.games.representation import NetworkDimensions
from src.training.checkpoint.contracts import load_checkpoint_manifest
from src.training.checkpoint.persistence import (
    create_optimizer,
    load_model,
    load_model_and_optimizer,
    save_model_and_optimizer,
)
from src.training.network import (
    CHESS_POLICY_PLANE_COUNT,
    AttentionNetworkParams,
    Chess76PlaneDirectPolicyHeadConfiguration,
    DensePolicyHeadConfiguration,
    DisabledResidualContext,
    GlobalPoolingResidualContext,
    GoPointPassPolicyHeadConfiguration,
    Network,
    NetworkConfiguration,
    NetworkParams,
    ResidualContextPlacement,
)
from src.training.targets import (
    LegalMovesHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
)

CHESS_PLANE_ACTION_SIZE = CHESS_POLICY_PLANE_COUNT * 64
CHESS_PLANE_NETWORK_DIMENSIONS = NetworkDimensions(
    channels=CHESS_NETWORK_DIMENSIONS.channels,
    rows=CHESS_NETWORK_DIMENSIONS.rows,
    columns=CHESS_NETWORK_DIMENSIONS.columns,
    actions=CHESS_PLANE_ACTION_SIZE,
)


@pytest.mark.parametrize(
    'parameters',
    (
        NetworkParams(
            num_layers=1,
            hidden_size=8,
            residual_context=DisabledResidualContext(),
            policy_head=GoPointPassPolicyHeadConfiguration(),
            num_value_channels=2,
            value_fc_size=8,
        ),
        NetworkParams(
            num_layers=1,
            hidden_size=8,
            residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_BLOCK),
            policy_head=GoPointPassPolicyHeadConfiguration(),
            num_value_channels=2,
            value_fc_size=8,
        ),
        AttentionNetworkParams(
            num_layers=1,
            embedding_size=8,
            num_heads=2,
            feedforward_size=16,
            policy_head=GoPointPassPolicyHeadConfiguration(),
            num_value_channels=2,
            value_fc_size=8,
        ),
    ),
)
def test_training_model_keeps_auxiliary_heads_but_jit_inference_model_trims_them(
    tmp_path: Path,
    parameters: NetworkConfiguration,
) -> None:
    dimensions = NetworkDimensions(channels=3, rows=3, columns=3, actions=10)
    auxiliary_heads = (
        NextPolicyHeadLayout(kind='next_policy', action_size=10, ply_offset=1),
        RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=100.0),
    )
    model = Network(parameters, torch.device('cpu'), dimensions, auxiliary_heads=auxiliary_heads)
    output = model.training_output(torch.zeros((2, 3, 3, 3)))

    assert output.policy_logits.shape == (2, 10)
    assert output.wdl_logits.shape == (2, 3)
    assert output.auxiliary_logits[0].shape == (2, 10)
    assert output.auxiliary_logits[1].shape == (2, 1)

    save_model_and_optimizer(model, create_optimizer(model, 'adamw'), 1, tmp_path)
    manifest = load_checkpoint_manifest(1, tmp_path)
    training_state = torch.load(tmp_path / 'model_1.pt', map_location='cpu', weights_only=True)
    extra_files = {'network.json': ''}
    inference_model = torch.jit.load(
        str(tmp_path / 'model_1.jit.pt'),
        map_location='cpu',
        _extra_files=extra_files,
    )
    loaded_model = load_model(
        tmp_path / 'model_1.pt',
        parameters,
        torch.device('cpu'),
        dimensions,
        auxiliary_heads=auxiliary_heads,
    )
    loaded_model.eval()
    loaded_output = loaded_model.training_output(torch.zeros((2, 3, 3, 3)))
    inference_policy, inference_wdl = inference_model(torch.zeros((2, 3, 3, 3)))
    training_policy_logits, training_wdl_logits = loaded_model.logit_forward(torch.zeros((2, 3, 3, 3)))

    assert any(name.startswith('auxiliary_head_modules.') for name in training_state)
    assert manifest.network == model.checkpoint_definition()
    assert all(not name.startswith('auxiliary_head_modules.') for name, _ in inference_model.named_parameters())
    assert tuple(output.shape for output in loaded_output.auxiliary_logits) == ((2, 10), (2, 1))
    assert inference_policy.shape == (2, 10)
    assert inference_wdl.shape == (2, 3)
    torch.testing.assert_close(inference_policy, training_policy_logits)
    torch.testing.assert_close(inference_wdl, torch.softmax(training_wdl_logits, dim=1))
    assert b'"kind":"' + parameters.kind.encode() + b'"' in extra_files['network.json']

    with pytest.raises(ValueError, match='network definition'):
        load_model_and_optimizer(
            1,
            parameters.model_copy(update={'num_layers': 2}),
            torch.device('cpu'),
            tmp_path,
            'adamw',
            dimensions,
            auxiliary_heads=auxiliary_heads,
        )


def test_spatial_policy_checkpoint_and_trimmed_jit_preserve_inference_abi(tmp_path: Path) -> None:
    parameters = NetworkParams(
        num_layers=1,
        hidden_size=16,
        residual_context=DisabledResidualContext(),
        policy_head=Chess76PlaneDirectPolicyHeadConfiguration(),
        num_value_channels=2,
        value_fc_size=8,
    )
    auxiliary_heads = (
        NextPolicyHeadLayout(kind='next_policy', action_size=CHESS_PLANE_ACTION_SIZE, ply_offset=1),
        RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=100.0),
    )
    model = Network(parameters, torch.device('cpu'), CHESS_PLANE_NETWORK_DIMENSIONS, auxiliary_heads)
    model.eval()
    inputs = torch.randn((2, 29, 8, 8))
    expected_training_output = model.training_output(inputs)
    expected_policy, expected_wdl = model(inputs)

    save_model_and_optimizer(model, create_optimizer(model, 'adamw'), 1, tmp_path)
    loaded = load_model(
        tmp_path / 'model_1.pt',
        parameters,
        torch.device('cpu'),
        CHESS_PLANE_NETWORK_DIMENSIONS,
        auxiliary_heads,
    )
    loaded.eval()
    loaded_training_output = loaded.training_output(inputs)
    inference_model = torch.jit.load(str(tmp_path / 'model_1.jit.pt'), map_location='cpu')
    inference_policy, inference_wdl = inference_model(inputs)

    assert expected_training_output.policy_logits.shape == (2, CHESS_PLANE_ACTION_SIZE)
    assert tuple(output.shape for output in expected_training_output.auxiliary_logits) == (
        (2, CHESS_PLANE_ACTION_SIZE),
        (2, 1),
    )
    assert all('action_plane_indices' not in name for name, _ in loaded.named_buffers())
    assert all(not name.startswith('auxiliary_head_modules.') for name, _ in inference_model.named_buffers())
    torch.testing.assert_close(loaded_training_output.policy_logits, expected_training_output.policy_logits)
    torch.testing.assert_close(loaded_training_output.auxiliary_logits[0], expected_training_output.auxiliary_logits[0])
    torch.testing.assert_close(inference_policy, expected_policy)
    torch.testing.assert_close(inference_wdl, expected_wdl)


@pytest.mark.parametrize(
    'parameters',
    (
        NetworkParams(
            num_layers=2,
            hidden_size=64,
            residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK),
            policy_head=Chess76PlaneDirectPolicyHeadConfiguration(),
            num_value_channels=2,
            value_fc_size=16,
        ),
        AttentionNetworkParams(
            num_layers=2,
            embedding_size=64,
            num_heads=2,
            feedforward_size=128,
            policy_head=Chess76PlaneDirectPolicyHeadConfiguration(),
            num_value_channels=2,
            value_fc_size=16,
        ),
        NetworkParams(
            num_layers=1,
            hidden_size=32,
            residual_context=DisabledResidualContext(),
            policy_head=DensePolicyHeadConfiguration(channels=8),
            num_value_channels=2,
            value_fc_size=16,
        ),
    ),
)
def test_jit_export_matches_training_forward_for_chess_heads(
    tmp_path: Path,
    parameters: NetworkConfiguration,
) -> None:
    torch.manual_seed(17)
    auxiliary_heads = (
        NextPolicyHeadLayout(kind='next_policy', action_size=CHESS_PLANE_ACTION_SIZE, ply_offset=1),
        LegalMovesHeadLayout(kind='legal_moves', action_size=CHESS_PLANE_ACTION_SIZE),
        RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=100.0),
    )
    model = Network(parameters, torch.device('cpu'), CHESS_PLANE_NETWORK_DIMENSIONS, auxiliary_heads)
    model.eval()
    inputs = torch.rand((4, 29, 8, 8))
    expected_policy_logits, expected_value = model(inputs)

    save_model_and_optimizer(model, create_optimizer(model, 'adamw'), 1, tmp_path)
    inference_model = torch.jit.load(str(tmp_path / 'model_1.jit.pt'), map_location='cpu')
    inference_model.eval()
    inference_policy, inference_wdl = inference_model(inputs)

    assert inference_policy.shape == (4, CHESS_PLANE_ACTION_SIZE)
    torch.testing.assert_close(inference_policy, expected_policy_logits, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(inference_wdl, expected_value, rtol=0.0, atol=1e-5)

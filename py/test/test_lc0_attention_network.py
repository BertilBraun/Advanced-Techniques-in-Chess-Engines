from __future__ import annotations

import pytest
import torch
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.checkpoint.persistence import create_model
from src.training.network import (
    ChessFromToAttentionPolicyHeadConfiguration,
    InferenceNetwork,
    Lc0AttentionInput,
    Lc0AttentionNetworkParams,
    Lc0EncoderBlock,
    NetworkDefinition,
    SmolgenAttentionBiasConfiguration,
)

DIMENSIONS = CHESS_NETWORK_DIMENSIONS


def small_lc0_network(num_layers: int = 2) -> Lc0AttentionNetworkParams:
    return Lc0AttentionNetworkParams(
        num_layers=num_layers,
        embedding_size=32,
        num_heads=4,
        feedforward_size=64,
        smolgen=SmolgenAttentionBiasConfiguration(compressed_size=4, hidden_size=16, generated_size=16),
        policy_head=ChessFromToAttentionPolicyHeadConfiguration(key_size=32),
        num_value_channels=8,
        value_fc_size=16,
    )


def random_states(count: int = 3) -> torch.Tensor:
    shape = (count, DIMENSIONS.channels, DIMENSIONS.rows, DIMENSIONS.columns)
    return torch.randint(0, 2, shape, generator=torch.Generator().manual_seed(4)).float()


def test_lc0_network_emits_one_logit_per_action_and_three_outcomes() -> None:
    model = create_model(small_lc0_network(), torch.device('cpu'), DIMENSIONS)

    output = model.training_output(random_states())

    assert (output.policy_logits.shape, output.wdl_logits.shape) == ((3, DIMENSIONS.actions), (3, 3))


def test_scripted_lc0_inference_network_matches_the_eager_model() -> None:
    torch.manual_seed(2)
    model = create_model(small_lc0_network(), torch.device('cpu'), DIMENSIONS)
    model.eval()
    inference = InferenceNetwork(model)
    inference.eval()
    inference.fuse_model()
    scripted = torch.jit.script(inference)
    states = random_states()

    with torch.no_grad():
        eager_policy, eager_value = inference(states)
        scripted_policy, scripted_value = scripted(states)

    assert torch.allclose(eager_policy, scripted_policy, atol=1e-5) and torch.allclose(
        eager_value, scripted_value, atol=1e-6
    )


@pytest.mark.parametrize('num_layers', (1, 4, 10))
def test_lc0_residual_branches_are_scaled_by_the_inverse_fourth_root_of_twice_the_depth(num_layers: int) -> None:
    model = create_model(small_lc0_network(num_layers), torch.device('cpu'), DIMENSIONS)

    scales = [block.residual_scale for block in model.backbone if isinstance(block, Lc0EncoderBlock)]

    assert scales == pytest.approx([(2 * num_layers) ** -0.25] * num_layers)


def test_lc0_input_gates_start_neutral() -> None:
    model = create_model(small_lc0_network(), torch.device('cpu'), DIMENSIONS)
    start_block = model.start_block
    assert isinstance(start_block, Lc0AttentionInput)

    assert torch.all(start_block.multiplicative_gate == 1) and torch.all(start_block.additive_gate == 0)


def test_lc0_network_definition_round_trips_through_json() -> None:
    model = create_model(small_lc0_network(), torch.device('cpu'), DIMENSIONS)
    definition = model.checkpoint_definition()

    restored = NetworkDefinition.model_validate_json(definition.model_dump_json())

    assert restored.architecture == small_lc0_network()


def test_lc0_input_distinguishes_squares_with_identical_planes() -> None:
    model = create_model(small_lc0_network(), torch.device('cpu'), DIMENSIONS)
    start_block = model.start_block
    assert isinstance(start_block, Lc0AttentionInput)

    with torch.no_grad():
        tokens = start_block(torch.zeros(1, DIMENSIONS.channels, DIMENSIONS.rows, DIMENSIONS.columns))

    assert not torch.allclose(tokens[0, 0], tokens[0, 63])

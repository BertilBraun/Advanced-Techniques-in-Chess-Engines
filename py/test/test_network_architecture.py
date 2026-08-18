import pytest
import torch
from pydantic import ValidationError

from torch import Tensor, nn

from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.network import (
    DensePolicyHeadConfiguration,
    Chess76PlanePolicyHead,
    Chess76PlanePolicyHeadConfiguration,
    DisabledResidualContext,
    GlobalPoolingBias,
    GlobalPoolingResBlock,
    GlobalPoolingResidualContext,
    Network,
    NetworkParams,
    ResBlock,
    ResidualContextPlacement,
    SqueezeExcitation,
    SqueezeExcitationResidualContext,
)
from src.training.targets import NextPolicyHeadLayout, RemainingGameLengthHeadLayout


DENSE_POLICY_HEAD = DensePolicyHeadConfiguration(channels=4)


class Multiply(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, values: Tensor) -> Tensor:
        return values * self.factor


def squeeze_excitation_blocks(network: Network) -> tuple[bool, ...]:
    return tuple(
        isinstance(block.squeeze_excitation, SqueezeExcitation)
        for block in network.backBone
        if isinstance(block, ResBlock)
    )


def global_pooling_blocks(network: Network) -> tuple[bool, ...]:
    return tuple(isinstance(block, GlobalPoolingResBlock) for block in network.backBone)


def test_squeeze_excitation_is_inside_residual_branch_before_skip_addition() -> None:
    block = ResBlock(16, use_squeeze_excitation=True)
    block.conv_block1 = nn.Identity()
    block.conv_block2 = nn.Identity()
    block.squeeze_excitation = Multiply(2.0)
    block.relu2 = nn.Identity()
    inputs = torch.ones((1, 16, 2, 2))

    assert torch.equal(block(inputs), inputs * 3)


def test_squeeze_excitation_uses_reduction_sixteen() -> None:
    squeeze_excitation = SqueezeExcitation(32)
    first_projection = squeeze_excitation.excite[0]

    assert isinstance(first_projection, nn.Conv2d)
    assert first_projection.out_channels == 2


def test_squeeze_excitation_placement_modes() -> None:
    disabled = Network(
        NetworkParams(
            num_layers=4,
            hidden_size=16,
            residual_context=DisabledResidualContext(),
            policy_head=DENSE_POLICY_HEAD,
        ),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
    )
    every_block = Network(
        NetworkParams(
            num_layers=4,
            hidden_size=16,
            residual_context=SqueezeExcitationResidualContext(
                placement=ResidualContextPlacement.EVERY_BLOCK,
            ),
            policy_head=DENSE_POLICY_HEAD,
        ),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
    )
    every_second_block = Network(
        NetworkParams(
            num_layers=4,
            hidden_size=16,
            residual_context=SqueezeExcitationResidualContext(
                placement=ResidualContextPlacement.EVERY_SECOND_BLOCK,
            ),
            policy_head=DENSE_POLICY_HEAD,
        ),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
    )

    assert len(disabled.backBone) == 4
    assert squeeze_excitation_blocks(disabled) == (False, False, False, False)
    assert squeeze_excitation_blocks(every_block) == (True, True, True, True)
    assert squeeze_excitation_blocks(every_second_block) == (False, True, False, True)


def test_global_pooling_uses_channel_means_and_maxima_as_additive_biases() -> None:
    pooling = GlobalPoolingBias(global_channels=2, local_channels=2)
    with torch.no_grad():
        pooling.projection.weight.copy_(
            torch.tensor(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                )
            )
        )
        pooling.projection.bias.zero_()
    local_features = torch.zeros((1, 2, 2, 2))
    global_features = torch.tensor(((((1.0, 3.0), (5.0, 7.0)), ((2.0, 4.0), (6.0, 8.0))),))

    output = pooling(local_features, global_features)

    assert torch.equal(output[0, 0], torch.full((2, 2), 4.0))
    assert torch.equal(output[0, 1], torch.full((2, 2), 7.0))


def test_global_pooling_placement_uses_one_quarter_of_channels() -> None:
    network = Network(
        NetworkParams(
            num_layers=4,
            hidden_size=16,
            residual_context=GlobalPoolingResidualContext(
                placement=ResidualContextPlacement.EVERY_SECOND_BLOCK,
            ),
            policy_head=DENSE_POLICY_HEAD,
        ),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
    )

    assert global_pooling_blocks(network) == (False, True, False, True)
    blocks = tuple(block for block in network.backBone if isinstance(block, GlobalPoolingResBlock))
    assert tuple(block.global_channels for block in blocks) == (4, 4)
    assert tuple(block.conv_block2[0].in_channels for block in blocks) == (12, 12)


def test_global_pooling_requires_distinct_global_and_local_channels() -> None:
    with pytest.raises(ValidationError):
        NetworkParams(
            num_layers=1,
            hidden_size=1,
            residual_context=GlobalPoolingResidualContext(
                placement=ResidualContextPlacement.EVERY_BLOCK,
            ),
            policy_head=DENSE_POLICY_HEAD,
        )


def test_chess_76_plane_policy_heads_preserve_action_outputs_and_backpropagate() -> None:
    auxiliary_heads = (
        NextPolicyHeadLayout(kind='next_policy', action_size=1880, ply_offset=1),
        RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=100.0),
    )
    network = Network(
        NetworkParams(
            num_layers=1,
            hidden_size=16,
            residual_context=DisabledResidualContext(),
            policy_head=Chess76PlanePolicyHeadConfiguration(hidden_channels=32),
        ),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
        auxiliary_heads,
    )

    output = network.training_output(torch.randn((2, 29, 8, 8)))
    loss = output.policy_logits.sum() + sum(logits.sum() for logits in output.auxiliary_logits)
    loss.backward()

    assert output.policy_logits.shape == (2, 1880)
    assert tuple(logits.shape for logits in output.auxiliary_logits) == ((2, 1880), (2, 1))
    assert isinstance(network.policyHead, Chess76PlanePolicyHead)
    assert torch.unique(network.policyHead.action_plane_indices).numel() == 1880
    learned_head_parameters = sum(
        parameter.numel()
        for name, parameter in network.named_parameters()
        if name.startswith(('policyHead.', 'valueHead.', 'auxiliaryHeads.'))
    )
    assert learned_head_parameters < 100_000
    assert network.policyHead.projection[0].weight.grad is not None
    assert network.auxiliaryHeads[0].projection[0].weight.grad is not None

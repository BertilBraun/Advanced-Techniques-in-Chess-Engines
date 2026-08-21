import pytest
import torch
from pydantic import ValidationError

from torch import Tensor, nn

from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.network import (
    ATTENTION_LINEAR_INITIALIZATION_STD,
    AttentionEncoderBlock,
    AttentionNetworkParams,
    Chess76PlaneDirectPolicyHeadConfiguration,
    DensePolicyHeadConfiguration,
    DisabledResidualContext,
    GlobalPoolingBias,
    GlobalPoolingResBlock,
    GlobalPoolingResidualContext,
    Network,
    NetworkParams,
    PolicyPlaneHead,
    ResBlock,
    ResidualContextPlacement,
    SqueezeExcitation,
    SqueezeExcitationResidualContext,
)
from src.training.targets import (
    LegalMovesHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
)


CHESS_POLICY_HEAD = Chess76PlaneDirectPolicyHeadConfiguration()


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
            policy_head=CHESS_POLICY_HEAD,
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
            policy_head=CHESS_POLICY_HEAD,
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
            policy_head=CHESS_POLICY_HEAD,
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
            policy_head=CHESS_POLICY_HEAD,
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
            policy_head=CHESS_POLICY_HEAD,
        )


def test_chess_76_plane_policy_heads_preserve_action_outputs_and_backpropagate() -> None:
    auxiliary_heads = (
        NextPolicyHeadLayout(kind='next_policy', action_size=4864, ply_offset=1),
        RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=100.0),
    )
    network = Network(
        NetworkParams(
            num_layers=1,
            hidden_size=16,
            residual_context=DisabledResidualContext(),
            policy_head=CHESS_POLICY_HEAD,
        ),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
        auxiliary_heads,
    )

    output = network.training_output(torch.randn((2, 29, 8, 8)))
    loss = output.policy_logits.sum() + sum(logits.sum() for logits in output.auxiliary_logits)
    loss.backward()

    assert output.policy_logits.shape == (2, 4864)
    assert tuple(logits.shape for logits in output.auxiliary_logits) == ((2, 4864), (2, 1))
    assert isinstance(network.policyHead, PolicyPlaneHead)
    assert isinstance(network.auxiliaryHeads[0], PolicyPlaneHead)
    learned_head_parameters = sum(
        parameter.numel()
        for name, parameter in network.named_parameters()
        if name.startswith(('policyHead.', 'valueHead.', 'auxiliaryHeads.'))
    )
    assert learned_head_parameters < 100_000
    assert network.policyHead.input_block[0].weight.grad is not None
    assert network.auxiliaryHeads[0].input_block[0].weight.grad is not None


def test_policy_plane_head_flattens_logits_plane_major() -> None:
    head = PolicyPlaneHead(input_channels=4, hidden_channels=8, plane_count=3)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.output_projection.bias.copy_(torch.tensor((1.0, 2.0, 3.0)))
    head.eval()

    logits = head(torch.zeros((1, 4, 8, 8)))

    assert logits.shape == (1, 3 * 64)
    assert torch.equal(logits[0, :64], torch.full((64,), 1.0))
    assert torch.equal(logits[0, 64:128], torch.full((64,), 2.0))
    assert torch.equal(logits[0, 128:], torch.full((64,), 3.0))


def test_chess_plane_heads_use_hidden_64_primary_and_32_auxiliary() -> None:
    network = Network(
        NetworkParams(
            num_layers=1,
            hidden_size=112,
            residual_context=DisabledResidualContext(),
            policy_head=CHESS_POLICY_HEAD,
        ),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
        (
            NextPolicyHeadLayout(kind='next_policy', action_size=4864, ply_offset=1),
            LegalMovesHeadLayout(kind='legal_moves', action_size=4864),
        ),
    )

    assert isinstance(network.policyHead, PolicyPlaneHead)
    assert network.policyHead.input_block[0].out_channels == 64
    assert network.policyHead.spatial_block[0].kernel_size == (3, 3)
    assert network.policyHead.output_projection.out_channels == 76
    primary_parameters = sum(parameter.numel() for parameter in network.policyHead.parameters())
    assert 45_000 < primary_parameters < 55_000
    for auxiliary_head in network.auxiliaryHeads:
        assert isinstance(auxiliary_head, PolicyPlaneHead)
        assert auxiliary_head.input_block[0].out_channels == 32
        auxiliary_parameters = sum(parameter.numel() for parameter in auxiliary_head.parameters())
        assert 13_000 < auxiliary_parameters < 18_000


def test_head_output_layers_use_small_initialization() -> None:
    torch.manual_seed(3)
    network = Network(
        NetworkParams(
            num_layers=1,
            hidden_size=64,
            residual_context=DisabledResidualContext(),
            policy_head=CHESS_POLICY_HEAD,
        ),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
        (
            NextPolicyHeadLayout(kind='next_policy', action_size=4864, ply_offset=1),
            RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=100.0),
        ),
    )

    assert float(network.policyHead.output_projection.weight.std()) == pytest.approx(0.01, rel=0.25)
    assert torch.equal(network.policyHead.output_projection.bias, torch.zeros(76))
    assert float(network.auxiliaryHeads[0].output_projection.weight.std()) == pytest.approx(0.01, rel=0.25)
    value_output = network.valueHead[-1]
    assert isinstance(value_output, nn.Linear)
    assert float(value_output.weight.std()) == pytest.approx(0.01, rel=0.5)
    assert torch.equal(value_output.bias, torch.zeros(3))
    length_output = network.auxiliaryHeads[1][-1]
    assert isinstance(length_output, nn.Linear)
    assert float(length_output.weight.std()) == pytest.approx(0.01, rel=0.5)


def test_attention_trunk_normalizes_tokens_before_spatial_output() -> None:
    network = Network(
        AttentionNetworkParams(
            num_layers=2,
            embedding_size=16,
            num_heads=2,
            feedforward_size=32,
            policy_head=CHESS_POLICY_HEAD,
        ),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
    )

    assert isinstance(network.finishBlock, nn.Sequential)
    assert isinstance(network.finishBlock[0], nn.LayerNorm)
    assert network.finishBlock[0].normalized_shape == (16,)


def test_attention_initialization_scales_residual_output_projections() -> None:
    torch.manual_seed(5)
    num_layers = 8
    network = Network(
        AttentionNetworkParams(
            num_layers=num_layers,
            embedding_size=128,
            num_heads=4,
            feedforward_size=256,
            policy_head=CHESS_POLICY_HEAD,
        ),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
    )
    expected_residual_std = ATTENTION_LINEAR_INITIALIZATION_STD / (2 * num_layers) ** 0.5

    for block in network.backBone:
        assert isinstance(block, AttentionEncoderBlock)
        assert float(block.query_key_value_projection.weight.std()) == pytest.approx(0.02, rel=0.1)
        assert float(block.feedforward[0].weight.std()) == pytest.approx(0.02, rel=0.1)
        assert float(block.attention_output_projection.weight.std()) == pytest.approx(expected_residual_std, rel=0.1)
        assert float(block.feedforward[3].weight.std()) == pytest.approx(expected_residual_std, rel=0.1)
    assert float(network.startBlock.projection.weight.std()) == pytest.approx(0.02, rel=0.1)
    assert float(network.startBlock.row_embeddings.std()) == pytest.approx(128**-0.5, rel=0.3)


def test_dense_policy_head_remains_selectable_with_replica_initialization() -> None:
    torch.manual_seed(7)
    network = Network(
        NetworkParams(
            num_layers=1,
            hidden_size=16,
            residual_context=DisabledResidualContext(),
            policy_head=DensePolicyHeadConfiguration(channels=8),
        ),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
    )
    network.eval()

    logits, _ = network.logit_forward(torch.randn((2, 29, 8, 8)))
    dense_output = network.policyHead[-1]

    assert logits.shape == (2, 4864)
    assert isinstance(network.policyHead, nn.Sequential)
    assert isinstance(dense_output, nn.Linear)
    assert float(dense_output.weight.std()) == pytest.approx((2.0 / 512) ** 0.5, rel=0.1)

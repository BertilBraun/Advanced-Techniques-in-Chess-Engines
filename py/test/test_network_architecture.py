import torch

from torch import Tensor, nn

from src.neural_network import Network, ResBlock, SqueezeExcitation
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.configuration import NetworkParams, SEPlacement


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
        NetworkParams(num_layers=4, hidden_size=16, se_placement=SEPlacement.DISABLED),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
    )
    every_block = Network(
        NetworkParams(num_layers=4, hidden_size=16, se_placement=SEPlacement.EVERY_BLOCK),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
    )
    every_second_block = Network(
        NetworkParams(num_layers=4, hidden_size=16, se_placement=SEPlacement.EVERY_SECOND_BLOCK),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
    )

    assert len(disabled.backBone) == 4
    assert squeeze_excitation_blocks(disabled) == (False, False, False, False)
    assert squeeze_excitation_blocks(every_block) == (True, True, True, True)
    assert squeeze_excitation_blocks(every_second_block) == (False, True, False, True)

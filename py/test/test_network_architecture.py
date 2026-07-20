import torch

from torch import Tensor, nn

from src.Network import Network, ResBlock, SqueezeExcitation
from src.train.TrainingArgs import NetworkParams, SEPlacement


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
    disabled = Network(NetworkParams(4, 16, SEPlacement.DISABLED), torch.device('cpu'))
    every_block = Network(NetworkParams(4, 16, SEPlacement.EVERY_BLOCK), torch.device('cpu'))
    every_second_block = Network(
        NetworkParams(4, 16, SEPlacement.EVERY_SECOND_BLOCK),
        torch.device('cpu'),
    )

    assert len(disabled.backBone) == 4
    assert squeeze_excitation_blocks(disabled) == (False, False, False, False)
    assert squeeze_excitation_blocks(every_block) == (True, True, True, True)
    assert squeeze_excitation_blocks(every_second_block) == (False, True, False, True)

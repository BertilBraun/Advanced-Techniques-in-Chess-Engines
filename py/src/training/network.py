from enum import Enum

import torch
from pydantic import Field

from torch import nn, Tensor

from src.games.representation import NetworkDimensions
from src.training.batch import TrainingModelOutput
from src.util.frozen_model import FrozenModel
from src.util.log import log


class SEPlacement(str, Enum):
    DISABLED = 'disabled'
    EVERY_BLOCK = 'every_block'
    EVERY_SECOND_BLOCK = 'every_second_block'

    def applies_to(self, block_index: int) -> bool:
        if self is SEPlacement.DISABLED:
            return False
        if self is SEPlacement.EVERY_BLOCK:
            return True
        return block_index % 2 == 1


class NetworkParams(FrozenModel):
    num_layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    se_placement: SEPlacement = SEPlacement.DISABLED
    num_policy_channels: int = Field(default=4, gt=0)
    num_value_channels: int = Field(default=2, gt=0)
    value_fc_size: int = Field(default=48, gt=0)


class Network(nn.Module):
    """
    The neural network model for the AlphaZero bot.

    The architecture is based on the AlphaZero paper, but with less layers.

    We use a residual neural network with NUM_RES_BLOCKS residual blocks.
    The input to the network is a ENCODING_CHANNELSxrow_countxcolumn_count tensor representing the board state.
    The output of the network is a policy over all possible moves and a value for the current board state.
    """

    def __init__(
        self,
        args: NetworkParams,
        device: torch.device,
        dimensions: NetworkDimensions,
        auxiliary_output_sizes: tuple[int, ...] = (),
    ) -> None:
        super().__init__()

        self.device = device
        self.network_args = args
        self.dimensions = dimensions
        self.auxiliary_output_sizes = auxiliary_output_sizes

        encoding_channels = dimensions.channels
        row_count = dimensions.rows
        column_count = dimensions.columns
        action_size = dimensions.actions

        self.startBlock = nn.Sequential(
            nn.Conv2d(encoding_channels, args.hidden_size, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(args.hidden_size),
            nn.ReLU(inplace=True),
        )

        self.backBone = nn.ModuleList(
            [
                ResBlock(
                    args.hidden_size,
                    use_squeeze_excitation=args.se_placement.applies_to(block_index),
                )
                for block_index in range(args.num_layers)
            ]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(args.hidden_size, args.num_policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(args.num_policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(args.num_policy_channels * row_count * column_count, action_size),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(args.hidden_size, args.num_value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(args.num_value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(args.num_value_channels * row_count * column_count, args.value_fc_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.value_fc_size, dimensions.outcomes),
        )
        self.auxiliaryHeads = nn.ModuleList(
            (
                nn.Sequential(
                    nn.Conv2d(args.hidden_size, args.num_policy_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(args.num_policy_channels),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                    nn.Linear(args.num_policy_channels * row_count * column_count, auxiliary_output_size),
                )
                for auxiliary_output_size in auxiliary_output_sizes
            )
        )

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        self.to(device=self.device, dtype=torch.float32)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._features(x)
        policy_logits = self.policyHead(x)
        value_logits = self.valueHead(x)

        policy = torch.softmax(policy_logits, dim=1)
        value = torch.softmax(value_logits, dim=1)
        return policy, value

    def _features(self, x: Tensor) -> Tensor:
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        return x

    def logit_forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._features(x)
        policy_logits = self.policyHead(x)
        value_logits = self.valueHead(x)

        return policy_logits, value_logits

    def training_output(self, x: Tensor) -> TrainingModelOutput:
        features = self._features(x)
        return TrainingModelOutput(
            policy_logits=self.policyHead(features),
            wdl_logits=self.valueHead(features),
            auxiliary_logits=tuple(head(features) for head in self.auxiliaryHeads),
        )

    def fuse_model(self):
        for m in self.modules():
            if (
                type(m) is nn.Sequential
                and len(m) >= 2
                and isinstance(m[0], nn.Conv2d)
                and isinstance(m[1], nn.BatchNorm2d)
            ):
                modules_to_fuse = [str(i) for i in range(min(3, len(m)))]  # Conv2d, BatchNorm2d, ReLU
                torch.ao.quantization.fuse_modules(m, modules_to_fuse, inplace=True)

    def disable_auto_grad(self):
        for p in self.parameters():
            p.requires_grad = False

    def print_params(self):
        for name, param in self.named_parameters():
            log(name, list(param.shape))
        sum_of_params = sum(p.numel() for p in self.parameters())
        log(f'Total number of parameters: {sum_of_params}')
        sum_of_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log(
            f'Total number of trainable parameters: {sum_of_trainable_params} ({sum_of_trainable_params / sum_of_params * 100:.2f}%)'
        )


class ResBlock(nn.Module):
    def __init__(
        self,
        num_hidden: int,
        use_squeeze_excitation: bool = False,
        squeeze_excitation_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(num_hidden),
        )
        self.squeeze_excitation: nn.Module = (
            SqueezeExcitation(num_hidden, squeeze_excitation_reduction) if use_squeeze_excitation else nn.Identity()
        )
        self.relu2 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.squeeze_excitation(x)
        x = x + residual
        x = self.relu2(x)
        return x


class SqueezeExcitation(nn.Module):
    """Channel attention for a residual branch, using a default reduction of 16."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # H×W → 1×1
        self.excite = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.excite(self.squeeze(x))  # shape: (N, C, 1, 1)
        return x * w  # channel‑wise re‑weight

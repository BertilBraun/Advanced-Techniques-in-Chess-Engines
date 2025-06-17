import torch

from torch import nn, Tensor

from src.settings import CurrentGame
from src.util.log import log


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
        num_res_blocks: int,
        hidden_size: int,
        device: torch.device,
        # TODO make these parameters configurable?
        se_positions: tuple[int, ...] = (1, 3),  # 0‑based indices of ResBlocks to upgrade
    ) -> None:
        super().__init__()

        self.device = device

        # TODO make these parameters configurable?
        num_policy_channels = 4
        num_value_channels = 2
        value_fc_size = 48

        encoding_channels, row_count, column_count = CurrentGame.representation_shape
        action_size = CurrentGame.action_size

        self.startBlock = nn.Sequential(
            nn.Conv2d(encoding_channels, hidden_size, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )

        self.backBone = nn.ModuleList()
        for i in range(num_res_blocks):
            self.backBone.append(ResBlock(hidden_size))
            if i in se_positions:
                self.backBone.append(SE1x1(hidden_size))  # Add SE block after ResBlock

        self.policyHead = nn.Sequential(
            nn.Conv2d(hidden_size, num_policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(num_policy_channels * row_count * column_count, action_size),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(hidden_size, num_value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(num_value_channels * row_count * column_count, value_fc_size),
            nn.ReLU(inplace=True),
            nn.Linear(value_fc_size, 1),
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
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        policy_logits = self.policyHead(x)
        value_logits = self.valueHead(x)

        policy = torch.softmax(policy_logits, dim=1)
        value = torch.tanh(value_logits)
        return policy, value

    def logit_forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        policy_logits = self.policyHead(x)
        value_logits = self.valueHead(x)

        return policy_logits, value_logits

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
    def __init__(self, num_hidden: int) -> None:
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
        self.relu2 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x += residual
        x = self.relu2(x)
        return x


class SE1x1(nn.Module):
    """Light‑weight squeeze‑and‑excitation (channel attention) block.

    Args:
        channels (int): number of feature channels in the input tensor.
        reduction (int): bottleneck ratio *r* (Hu et al., 2017).  Default 8.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # H×W → 1×1
        self.excite = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        w = self.excite(self.squeeze(x))  # shape: (N, C, 1, 1)
        return x * w  # channel‑wise re‑weight

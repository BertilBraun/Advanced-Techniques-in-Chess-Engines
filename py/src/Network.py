import torch

from torch import nn, Tensor

from src.settings import CurrentGame, TORCH_DTYPE
from src.util.log import log


class Network(nn.Module):
    """
    The neural network model for the AlphaZero bot.

    The architecture is based on the AlphaZero paper, but with less layers.

    We use a residual neural network with NUM_RES_BLOCKS residual blocks.
    The input to the network is a ENCODING_CHANNELSxrow_countxcolumn_count tensor representing the board state.
    The output of the network is a policy over all possible moves and a value for the current board state.
    """

    def __init__(self, num_res_blocks: int, hidden_size: int, device: torch.device) -> None:
        super().__init__()

        self.device = device

        encoding_channels, row_count, column_count = CurrentGame.representation_shape
        action_size = CurrentGame.action_size

        self.startBlock = nn.Sequential(
            nn.Conv2d(encoding_channels, hidden_size, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList([ResBlock(hidden_size) for _ in range(num_res_blocks)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(hidden_size, 16, kernel_size=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * (row_count - 2) * (column_count - 2), action_size),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(hidden_size, 16, kernel_size=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * (row_count - 2) * (column_count - 2), 1),
            nn.Tanh(),
        )

        self.to(device=self.device, dtype=TORCH_DTYPE)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        return self.policyHead(x), self.valueHead(x)

    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
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

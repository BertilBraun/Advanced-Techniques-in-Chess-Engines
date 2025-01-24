import torch

from torch import nn, Tensor

from src.settings import CurrentGame, TORCH_DTYPE


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
            nn.Conv2d(encoding_channels, hidden_size, kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList([ResBlock(hidden_size) for _ in range(num_res_blocks)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(hidden_size, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * (row_count - 2) * (column_count - 2), action_size),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(hidden_size, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * (row_count - 2) * (column_count - 2), 1),
            nn.Tanh(),
        )

        self.to(device=self.device, dtype=TORCH_DTYPE)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                torch.ao.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)  # Conv2d, BatchNorm2d, ReLU
            if type(m) == ResBlock:
                m.fuse_model()

    def disable_auto_grad(self):
        for p in self.parameters():
            p.requires_grad = False

    def print_params(self):
        for name, param in self.named_parameters():
            print(name, list(param.shape))
        sum_of_params = sum(p.numel() for p in self.parameters())
        print(f'Total number of parameters: {sum_of_params}')
        sum_of_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f'Total number of trainable parameters: {sum_of_trainable_params} ({sum_of_trainable_params / sum_of_params * 100:.2f}%)'
        )


class ResBlock(nn.Module):
    def __init__(self, num_hidden: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.relu2 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.relu2(x)
        return x

    def fuse_model(self):
        torch.ao.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
        torch.ao.quantization.fuse_modules(self, ['conv2', 'bn2'], inplace=True)

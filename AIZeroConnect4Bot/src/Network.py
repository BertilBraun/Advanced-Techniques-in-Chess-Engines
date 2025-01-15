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
            nn.Conv2d(hidden_size, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * row_count * column_count, action_size),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(hidden_size, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * row_count * column_count, 1),
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


# TODO compare inference speed with and without fusing on both cpu as well as gpu compiled as well as not compiled
if __name__ == '__main__':
    import time
    from itertools import product

    sample_shape = CurrentGame.representation_shape

    for device, dtype, fused, compiled, batch_size in product(
        ['cpu', 'cuda'],
        [torch.float32, torch.float16, torch.bfloat16],
        [True, False],
        ['none', 'jit', 'compile'],
        [1, 32, 128],
    ):
        model = Network(num_res_blocks=10, hidden_size=256, device=torch.device(device))
        model.to(device=device, dtype=dtype)
        model.eval()

        if fused:
            model.fuse_model()

        if compiled == 'jit':
            model = torch.jit.script(model)
        elif compiled == 'compile':
            model = torch.compile(model)

        num_iterations = 128 * 4
        iterations = num_iterations // batch_size

        inputs = [torch.randn((batch_size, *sample_shape), device=device, dtype=dtype) for _ in range(iterations)]

        start = time.time()
        for i in range(iterations):
            model(inputs[i])
        total_time = time.time() - start
        print(f'{device=} {dtype=} {fused=} {compiled=} {batch_size=} {iterations=} {total_time=:.2f}')

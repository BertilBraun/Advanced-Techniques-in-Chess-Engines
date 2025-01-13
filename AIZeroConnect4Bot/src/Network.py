import torch

from torch import nn, Tensor
import torch.ao.quantization

from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.settings import TRAINING_ARGS, CurrentGame, TORCH_DTYPE


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
            nn.Conv2d(encoding_channels, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList([ResBlock(hidden_size) for _ in range(num_res_blocks)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(hidden_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * row_count * column_count, action_size),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(hidden_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * row_count * column_count, 1),
            nn.Tanh(),
        )

        self.to(device=self.device, dtype=TORCH_DTYPE)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.quant(x)
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        policy = self.dequant(policy)
        value = self.dequant(value)
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
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.relu2 = nn.ReLU()

        self.add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.add.add(x, residual)
        x = self.relu2(x)
        return x

    def fuse_model(self):
        torch.ao.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
        torch.ao.quantization.fuse_modules(self, ['conv2', 'bn2'], inplace=True)


def int8_quantize_model(model: Network, dataset: SelfPlayDataset) -> Network:
    print('Quantizing model to int8...')
    model = model.eval()
    print('Model in eval mode')
    model.fuse_model()
    print('Model fused')
    model.qconfig = torch.ao.quantization.default_qconfig
    # get_default_qconfig('x86')
    print('Model qconfig set')

    torch.quantization.prepare(model, inplace=True)
    print('Model prepared')

    for sample, _, _ in dataset:
        print('Forwarding sample')
        model(sample)
    print('Forwarded all samples')

    print('Converting model')
    torch.quantization.convert(model, inplace=True)
    print('Model converted')

    return model


if __name__ == '__main__':
    from src.util.save_paths import load_model, model_save_path
    import os

    os.environ['WANDB_WATCH'] = 'false'

    model = load_model(model_save_path(1, TRAINING_ARGS.save_path), TRAINING_ARGS.network, torch.device('cpu'))
    dataset = SelfPlayDataset.load_iteration(TRAINING_ARGS.save_path, 1)
    model = int8_quantize_model(model, dataset)

    print(model)
    print(model(dataset[0][0]))

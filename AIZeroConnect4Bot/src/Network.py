import torch
import numpy as np
import torch.nn.functional as F

from torch import nn, Tensor, softmax

from AIZeroConnect4Bot.src.util.log import log, ratio
from AIZeroConnect4Bot.src.settings import CURRENT_GAME, TORCH_DTYPE


NN_CACHE: dict[int, tuple[Tensor, Tensor]] = {}

TOTAL_EVALS = 0
TOTAL_HITS = 0


def cached_network_forward(network: nn.Module, x: Tensor) -> tuple[Tensor, Tensor]:
    # Cache the results and deduplicate positions (only run calculation once and return the result in multiple places)
    # use hash_board on each board state in x and check if it is in the cache or twice in x
    hashes = CURRENT_GAME.hash_boards(x)
    to_process = []
    to_process_hashes = []
    for i, h in enumerate(hashes):
        if h not in NN_CACHE or h in hashes[:i]:
            to_process.append(x[i])
            to_process_hashes.append(h)

    if to_process:
        policy, value = network(torch.stack(to_process))
        for hash, p, v in zip(to_process_hashes, policy, value):
            NN_CACHE[hash] = (p, v)

    global TOTAL_EVALS, TOTAL_HITS
    TOTAL_EVALS += len(x)
    TOTAL_HITS += len(x) - len(to_process)

    policies = torch.stack([NN_CACHE[hash][0] for hash in hashes])
    values = torch.stack([NN_CACHE[hash][1] for hash in hashes])
    return policies, values


def cached_network_inference(network: nn.Module, x: Tensor) -> tuple[np.ndarray, np.ndarray]:
    result: tuple[Tensor, Tensor] = cached_network_forward(network, x)
    policy, value = result
    policy = softmax(policy, dim=1).to(dtype=torch.float32, device='cpu').numpy()
    value = value.squeeze(1).to(dtype=torch.float32, device='cpu').numpy()
    return policy, value


def clear_cache() -> None:
    if TOTAL_EVALS != 0:
        log('Cache hit rate:', ratio(TOTAL_HITS, TOTAL_EVALS), 'on cache size', len(NN_CACHE))
    NN_CACHE.clear()


class Network(nn.Module):
    """
    The neural network model for the AlphaZero bot.

    The architecture is based on the AlphaZero paper, but with less layers.

    We use a residual neural network with NUM_RES_BLOCKS residual blocks.
    The input to the network is a ENCODING_CHANNELSxrow_countxcolumn_count tensor representing the board state.
    The output of the network is a policy over all possible moves and a value for the current board state.
    """

    def __init__(self) -> None:
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        encoding_channels, row_count, column_count = CURRENT_GAME.representation_shape
        num_res_blocks, num_hidden = CURRENT_GAME.network_properties
        action_size = CURRENT_GAME.action_size

        self.startBlock = nn.Sequential(
            nn.Conv2d(encoding_channels, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_res_blocks)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * row_count * column_count, action_size),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
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


class ResBlock(nn.Module):
    def __init__(self, num_hidden: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

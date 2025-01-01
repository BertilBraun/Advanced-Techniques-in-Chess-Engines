import torch
import numpy as np
import tensorflow as tf
import torch.nn.functional as F

from torch import nn, Tensor, softmax

from src.util.log import log, ratio
from src.settings import CurrentGame, TORCH_DTYPE

_NETWORK_ID = int
_NN_CACHE: dict[_NETWORK_ID, dict[int, tuple[Tensor, Tensor]]] = {}

_TOTAL_EVALS = 0
_TOTAL_HITS = 0


@torch.no_grad()
def cached_network_forward(network: nn.Module, x: Tensor) -> tuple[Tensor, Tensor]:
    # Cache the results and deduplicate positions (only run calculation once and return the result in multiple places)
    # use hash_board on each board state in x and check if it is in the cache or twice in x
    current_network_cache = _NN_CACHE.setdefault(id(network), {})

    hashes = CurrentGame.hash_boards(x)
    to_process = []
    to_process_hashes = []
    for i, h in enumerate(hashes):
        if h not in current_network_cache or h in hashes[:i]:
            to_process.append(x[i])
            to_process_hashes.append(h)

    if to_process:
        policy, value = network(torch.stack(to_process))
        for hash, p, v in zip(to_process_hashes, policy, value):
            current_network_cache[hash] = (p, v)

    global _TOTAL_EVALS, _TOTAL_HITS
    _TOTAL_EVALS += len(x)
    _TOTAL_HITS += len(x) - len(to_process)

    policies = torch.stack([current_network_cache[hash][0] for hash in hashes])
    values = torch.stack([current_network_cache[hash][1] for hash in hashes])
    return policies, values


def cached_network_inference(network: nn.Module, x: Tensor) -> tuple[np.ndarray, np.ndarray]:
    result: tuple[Tensor, Tensor] = cached_network_forward(network, x)
    policy, values = result
    policy = softmax(policy, dim=1).to(dtype=torch.float32, device='cpu').numpy()
    values = values.to(dtype=torch.float32, device='cpu').numpy()
    # TODO value mean should not be required anymore - remainder of the multiple value output model
    value = np.mean(values, axis=1)
    return policy, value


def clear_model_inference_cache(iteration: int) -> None:
    cache_size = sum(len(cache) for cache in _NN_CACHE.values())
    if _TOTAL_EVALS != 0:
        tf.summary.scalar('cache_hit_rate', _TOTAL_HITS / _TOTAL_EVALS, step=iteration)
        tf.summary.scalar('unique_positions_in_cache', cache_size, step=iteration)
        tf.summary.histogram(
            'nn_output_value_distribution',
            [round(v.item(), 1) for network in _NN_CACHE.values() for _, v in network.values()],
            step=iteration,
        )
        log('Cache hit rate:', ratio(_TOTAL_HITS, _TOTAL_EVALS), 'on cache size', cache_size)
    _NN_CACHE.clear()


class Network(nn.Module):
    """
    The neural network model for the AlphaZero bot.

    The architecture is based on the AlphaZero paper, but with less layers.

    We use a residual neural network with NUM_RES_BLOCKS residual blocks.
    The input to the network is a ENCODING_CHANNELSxrow_countxcolumn_count tensor representing the board state.
    The output of the network is a policy over all possible moves and a value for the current board state.
    """

    def __init__(self, num_res_blocks: int, hidden_size: int) -> None:
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

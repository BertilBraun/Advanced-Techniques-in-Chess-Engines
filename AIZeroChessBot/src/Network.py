import torch
import numpy as np
import torch.nn.functional as F

from numpy.typing import NDArray
from torch import nn, Tensor, softmax

from AIZeroChessBot.src.MoveEncoding import ACTION_SIZE
from AIZeroChessBot.src.BoardEncoding import ENCODING_CHANNELS


ROW_COUNT = 8
COLUMN_COUNT = 8
NUM_RES_BLOCKS = 8
NUM_HIDDEN = 256


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.startBlock = nn.Sequential(
            nn.Conv2d(ENCODING_CHANNELS, NUM_HIDDEN, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_HIDDEN),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList([ResBlock(NUM_HIDDEN) for _ in range(NUM_RES_BLOCKS)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(NUM_HIDDEN, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * ROW_COUNT * COLUMN_COUNT, ACTION_SIZE),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(NUM_HIDDEN, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * ROW_COUNT * COLUMN_COUNT, 1),
            nn.Tanh(),
        )

        self.to(self.device)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

    def inference(self, x: Tensor) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        result: tuple[Tensor, Tensor] = self(x)
        policy, value = result
        policy = softmax(policy, dim=1).cpu().numpy()
        value = value.squeeze(1).cpu().numpy()
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

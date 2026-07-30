from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.az.games.go.configuration import GoGameConfiguration, ResidualGoModelConfiguration


@dataclass(frozen=True)
class GoModelOutput:
    policy_logits: Tensor
    value: Tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convolution_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.normalization_1 = nn.BatchNorm2d(channels)
        self.convolution_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.normalization_2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = torch.relu(self.normalization_1(self.convolution_1(inputs)))
        residual = self.normalization_2(self.convolution_2(residual))
        return torch.relu(inputs + residual)


class ResidualGoModel(nn.Module):
    def __init__(
        self,
        game_configuration: GoGameConfiguration,
        model_configuration: ResidualGoModelConfiguration,
    ) -> None:
        super().__init__()
        self._board_size = game_configuration.board_size
        self._action_count = game_configuration.action_count
        self._input_plane_count = game_configuration.input_plane_count
        channels = model_configuration.channels
        self.input_convolution = nn.Conv2d(
            game_configuration.input_plane_count,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.input_normalization = nn.BatchNorm2d(channels)
        self.residual_tower = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(model_configuration.residual_blocks))
        )
        self.policy_convolution = nn.Conv2d(channels, model_configuration.policy_channels, kernel_size=1, bias=False)
        self.policy_normalization = nn.BatchNorm2d(model_configuration.policy_channels)
        self.policy_projection = nn.Linear(
            model_configuration.policy_channels * self._board_size**2,
            self._action_count,
        )
        self.value_convolution = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_normalization = nn.BatchNorm2d(1)
        self.value_hidden = nn.Linear(self._board_size**2, model_configuration.value_hidden_size)
        self.value_projection = nn.Linear(model_configuration.value_hidden_size, 1)

    def forward(self, inputs: Tensor) -> GoModelOutput:
        if (
            inputs.ndim != 4
            or inputs.shape[1] != self._input_plane_count
            or inputs.shape[2:] != (self._board_size, self._board_size)
        ):
            raise ValueError('Go model inputs must have shape B x planes x N x N.')
        features = torch.relu(self.input_normalization(self.input_convolution(inputs)))
        features = self.residual_tower(features)
        policy = torch.relu(self.policy_normalization(self.policy_convolution(features)))
        policy_logits = self.policy_projection(torch.flatten(policy, start_dim=1))
        value = torch.relu(self.value_normalization(self.value_convolution(features)))
        value = torch.relu(self.value_hidden(torch.flatten(value, start_dim=1)))
        return GoModelOutput(policy_logits=policy_logits, value=torch.tanh(self.value_projection(value)).squeeze(1))

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, TypeAlias

import torch
from pydantic import BeforeValidator, Field, JsonValue, model_validator

from torch import nn, Tensor
from torch.nn import functional

from src.games.representation import NetworkDimensions
from src.training.batch import TrainingModelOutput
from src.training.targets import AuxiliaryHeadLayout, NextPolicyHeadLayout, RemainingGameLengthHeadLayout
from src.util.frozen_model import FrozenModel
from src.util.log import log


class ResidualContextPlacement(str, Enum):
    EVERY_BLOCK = 'every_block'
    EVERY_SECOND_BLOCK = 'every_second_block'

    def applies_to(self, block_index: int) -> bool:
        if self is ResidualContextPlacement.EVERY_BLOCK:
            return True
        return block_index % 2 == 1


class DisabledResidualContext(FrozenModel):
    kind: Literal['disabled'] = 'disabled'


class SqueezeExcitationResidualContext(FrozenModel):
    kind: Literal['squeeze_excitation'] = 'squeeze_excitation'
    placement: ResidualContextPlacement


class GlobalPoolingResidualContext(FrozenModel):
    kind: Literal['global_pooling'] = 'global_pooling'
    placement: ResidualContextPlacement


ResidualContextConfiguration: TypeAlias = Annotated[
    DisabledResidualContext | SqueezeExcitationResidualContext | GlobalPoolingResidualContext,
    Field(discriminator='kind'),
]


class Chess76PlaneDirectPolicyHeadConfiguration(FrozenModel):
    kind: Literal['chess_76_plane_direct_v2'] = 'chess_76_plane_direct_v2'


class GoPointPassPolicyHeadConfiguration(FrozenModel):
    kind: Literal['go_point_pass_v1'] = 'go_point_pass_v1'


PolicyHeadConfiguration: TypeAlias = Annotated[
    Chess76PlaneDirectPolicyHeadConfiguration | GoPointPassPolicyHeadConfiguration,
    Field(discriminator='kind'),
]


class NetworkHeadParams(FrozenModel):
    policy_head: PolicyHeadConfiguration
    num_value_channels: int = Field(default=2, gt=0)
    value_fc_size: int = Field(default=48, gt=0)


class NetworkParams(NetworkHeadParams):
    kind: Literal['convolutional'] = 'convolutional'
    num_layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    residual_context: ResidualContextConfiguration = DisabledResidualContext()

    @model_validator(mode='after')
    def validate_global_pooling_width(self) -> NetworkParams:
        match self.residual_context:
            case GlobalPoolingResidualContext() if self.hidden_size < 2:
                raise ValueError('Global-pooling residual blocks require at least two hidden channels.')
        return self


class AttentionNetworkParams(NetworkHeadParams):
    kind: Literal['attention'] = 'attention'
    num_layers: int = Field(gt=0)
    embedding_size: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    feedforward_size: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode='after')
    def validate_attention_dimensions(self) -> AttentionNetworkParams:
        if self.embedding_size % self.num_heads:
            raise ValueError('Attention embedding size must be divisible by the number of heads.')
        return self


NetworkConfigurationInput: TypeAlias = NetworkParams | AttentionNetworkParams | dict[str, JsonValue]


def _normalize_network_discriminator(configuration: NetworkConfigurationInput) -> NetworkConfigurationInput:
    match configuration:
        case dict() if 'kind' not in configuration:
            return {**configuration, 'kind': 'convolutional'}
        case NetworkParams() | AttentionNetworkParams() | dict():
            return configuration


NetworkConfiguration: TypeAlias = Annotated[
    NetworkParams | AttentionNetworkParams,
    Field(discriminator='kind'),
    BeforeValidator(_normalize_network_discriminator),
]


class NetworkDefinition(FrozenModel):
    architecture: NetworkConfiguration
    dimensions: NetworkDimensions
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...]


class Network(nn.Module):
    """AlphaZero training model with a configured convolutional or attention backbone."""

    def __init__(
        self,
        args: NetworkConfiguration,
        device: torch.device,
        dimensions: NetworkDimensions,
        auxiliary_heads: tuple[AuxiliaryHeadLayout, ...] = (),
    ) -> None:
        super().__init__()

        self.device = device
        self.network_args = args
        self.dimensions = dimensions
        self.auxiliary_heads = auxiliary_heads

        encoding_channels = dimensions.channels
        row_count = dimensions.rows
        column_count = dimensions.columns
        action_size = dimensions.actions

        match args:
            case NetworkParams():
                hidden_size = args.hidden_size
                self.startBlock = nn.Sequential(
                    nn.Conv2d(encoding_channels, hidden_size, kernel_size=3, padding='same', bias=False),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU(inplace=True),
                )
                self.backBone = nn.ModuleList(
                    [
                        _build_residual_block(hidden_size, args.residual_context, block_index)
                        for block_index in range(args.num_layers)
                    ]
                )
                self.finishBlock = nn.Identity()
            case AttentionNetworkParams():
                hidden_size = args.embedding_size
                self.startBlock = AttentionInput(
                    encoding_channels,
                    row_count,
                    column_count,
                    hidden_size,
                )
                self.backBone = nn.ModuleList(
                    [
                        AttentionEncoderBlock(
                            hidden_size,
                            args.num_heads,
                            args.feedforward_size,
                            args.dropout,
                        )
                        for _ in range(args.num_layers)
                    ]
                )
                self.finishBlock = AttentionOutput(row_count, column_count)

        self.policyHead = _build_policy_head(
            hidden_size,
            row_count,
            column_count,
            action_size,
            args.policy_head,
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(hidden_size, args.num_value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(args.num_value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(args.num_value_channels * row_count * column_count, args.value_fc_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.value_fc_size, dimensions.outcomes),
        )
        self.auxiliaryHeads = nn.ModuleList(
            _build_auxiliary_head(
                hidden_size,
                row_count,
                column_count,
                action_size,
                args.policy_head,
                auxiliary_head,
            )
            for auxiliary_head in auxiliary_heads
        )

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        self.to(device=self.device, dtype=torch.float32)

    @torch.jit.unused
    def checkpoint_definition(self) -> NetworkDefinition:
        return NetworkDefinition(
            architecture=self.network_args,
            dimensions=self.dimensions,
            auxiliary_heads=self.auxiliary_heads,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._features(x)
        policy_logits = self.policyHead(x)
        value_logits = self.valueHead(x)

        value = torch.softmax(value_logits, dim=1)
        return policy_logits, value

    def _features(self, x: Tensor) -> Tensor:
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        return self.finishBlock(x)

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

    def fuse_model(self) -> None:
        for m in self.modules():
            if (
                type(m) is nn.Sequential
                and len(m) >= 2
                and isinstance(m[0], nn.Conv2d)
                and isinstance(m[1], nn.BatchNorm2d)
            ):
                modules_to_fuse = [str(i) for i in range(min(3, len(m)))]  # Conv2d, BatchNorm2d, ReLU
                torch.ao.quantization.fuse_modules(m, modules_to_fuse, inplace=True)

    def disable_auto_grad(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def print_params(self) -> None:
        for name, param in self.named_parameters():
            log(name, list(param.shape))
        sum_of_params = sum(p.numel() for p in self.parameters())
        log(f'Total number of parameters: {sum_of_params}')
        sum_of_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log(
            f'Total number of trainable parameters: {sum_of_trainable_params} ({sum_of_trainable_params / sum_of_params * 100:.2f}%)'
        )


class GoPointPassPolicyHead(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.point_projection = nn.Conv2d(input_channels, 1, kernel_size=1, bias=True)
        self.pass_projection = nn.Linear(input_channels, 1)

    def forward(self, features: Tensor) -> Tensor:
        point_logits = self.point_projection(features).flatten(start_dim=1)
        pooled_features = torch.mean(features, dim=(2, 3))
        pass_logit = self.pass_projection(pooled_features)
        return torch.cat((point_logits, pass_logit), dim=1)


def _build_policy_head(
    input_channels: int,
    row_count: int,
    column_count: int,
    action_size: int,
    configuration: PolicyHeadConfiguration,
) -> nn.Module:
    match configuration:
        case Chess76PlaneDirectPolicyHeadConfiguration():
            if row_count != 8 or column_count != 8 or action_size != 76 * 64:
                raise ValueError('Chess direct policy heads require an 8x8 board and 4,864 actions.')
            return nn.Sequential(
                nn.Conv2d(input_channels, 76, kernel_size=1, bias=True),
                nn.Flatten(),
            )
        case GoPointPassPolicyHeadConfiguration():
            if action_size != row_count * column_count + 1:
                raise ValueError('Go point-pass policy heads require one action per point plus pass.')
            return GoPointPassPolicyHead(input_channels)


def _build_auxiliary_head(
    input_channels: int,
    row_count: int,
    column_count: int,
    action_size: int,
    policy_configuration: PolicyHeadConfiguration,
    layout: AuxiliaryHeadLayout,
) -> nn.Module:
    match layout:
        case NextPolicyHeadLayout(action_size=next_policy_action_size):
            if next_policy_action_size != action_size:
                raise ValueError('Next-policy action space must match the primary policy action space.')
            return _build_policy_head(
                input_channels,
                row_count,
                column_count,
                action_size,
                policy_configuration,
            )
        case RemainingGameLengthHeadLayout(output_size=output_size):
            return nn.Sequential(
                nn.Conv2d(input_channels, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(row_count * column_count, output_size),
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


class AttentionInput(nn.Module):
    def __init__(
        self,
        input_channels: int,
        rows: int,
        columns: int,
        embedding_size: int,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.columns = columns
        self.projection = nn.Linear(input_channels, embedding_size)
        self.row_embeddings = nn.Parameter(torch.empty(rows, embedding_size))
        self.column_embeddings = nn.Parameter(torch.empty(columns, embedding_size))
        self.normalization = nn.LayerNorm(embedding_size)
        nn.init.normal_(self.row_embeddings, mean=0.0, std=embedding_size**-0.5)
        nn.init.normal_(self.column_embeddings, mean=0.0, std=embedding_size**-0.5)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.shape[0]
        tokens = inputs.permute(0, 2, 3, 1).reshape(batch_size, self.rows * self.columns, -1)
        positions = (self.row_embeddings[:, None, :] + self.column_embeddings[None, :, :]).reshape(
            self.rows * self.columns,
            -1,
        )
        return self.normalization(self.projection(tokens) + positions[None, :, :])


class AttentionEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        feedforward_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        self.attention_dropout_probability = dropout
        self.attention_normalization = nn.LayerNorm(embedding_size)
        self.query_key_value_projection = nn.Linear(embedding_size, embedding_size * 3)
        self.attention_output_projection = nn.Linear(embedding_size, embedding_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_normalization = nn.LayerNorm(embedding_size)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, feedforward_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, embedding_size),
        )
        self.feedforward_dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.attention_normalization(inputs)
        batch_size, square_count, _ = normalized.shape
        query_key_value = self.query_key_value_projection(normalized).reshape(
            batch_size,
            square_count,
            3,
            self.num_heads,
            self.head_size,
        )
        query, key, value = query_key_value.permute(2, 0, 3, 1, 4).unbind(0)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout_probability if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, square_count, self.embedding_size)
        attended = self.attention_output_projection(attended)
        residual = inputs + self.attention_dropout(attended)
        feedforward = self.feedforward(self.feedforward_normalization(residual))
        return residual + self.feedforward_dropout(feedforward)


class AttentionOutput(nn.Module):
    def __init__(self, rows: int, columns: int) -> None:
        super().__init__()
        self.rows = rows
        self.columns = columns

    def forward(self, tokens: Tensor) -> Tensor:
        batch_size = tokens.shape[0]
        return tokens.reshape(batch_size, self.rows, self.columns, -1).permute(0, 3, 1, 2).contiguous()


class GlobalPoolingBias(nn.Module):
    def __init__(self, global_channels: int, local_channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(global_channels * 2, local_channels)

    def forward(self, local_features: Tensor, global_features: Tensor) -> Tensor:
        # A board-size-scaled copy of the mean is redundant for fixed-size models.
        means = torch.mean(global_features, dim=(2, 3))
        maxima = torch.amax(global_features, dim=(2, 3))
        biases = self.projection(torch.cat((means, maxima), dim=1))
        return local_features + biases[:, :, None, None]


class GlobalPoolingResBlock(nn.Module):
    def __init__(self, num_hidden: int) -> None:
        super().__init__()
        self.global_channels = max(1, num_hidden // 4)
        local_channels = num_hidden - self.global_channels
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )
        self.global_pooling_bias = GlobalPoolingBias(self.global_channels, local_channels)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(local_channels, num_hidden, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(num_hidden),
        )
        self.relu2 = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.conv_block1(inputs)
        global_features = features[:, : self.global_channels]
        local_features = features[:, self.global_channels :]
        biased_features = self.global_pooling_bias(local_features, global_features)
        return self.relu2(self.conv_block2(biased_features) + inputs)


def _build_residual_block(
    hidden_channels: int,
    residual_context: ResidualContextConfiguration,
    block_index: int,
) -> nn.Module:
    match residual_context:
        case DisabledResidualContext():
            return ResBlock(hidden_channels)
        case SqueezeExcitationResidualContext(placement=placement):
            return ResBlock(hidden_channels, use_squeeze_excitation=placement.applies_to(block_index))
        case GlobalPoolingResidualContext(placement=placement):
            return (
                GlobalPoolingResBlock(hidden_channels)
                if placement.applies_to(block_index)
                else ResBlock(hidden_channels)
            )


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

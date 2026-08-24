from __future__ import annotations

import copy
import math
from enum import Enum
from typing import Annotated, Literal, TypeAlias

import torch
from pydantic import BeforeValidator, Field, JsonValue, model_validator
from src.games.representation import NetworkDimensions
from src.training.batch import TrainingModelOutput
from src.training.targets import (
    AuxiliaryHeadLayout,
    FutureSearchValueHeadLayout,
    IrreversibleProgressHeadLayout,
    LegalMovesHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    SearchCorrectionHeadLayout,
)
from src.util.frozen_model import FrozenModel
from torch import Tensor, nn
from torch.nn import functional


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


MAXIMUM_DENSE_SPATIAL_REDUCTIONS = 2


class Chess76PlaneDirectPolicyHeadConfiguration(FrozenModel):
    kind: Literal['chess_76_plane_direct_v2'] = 'chess_76_plane_direct_v2'


class DensePolicyHeadConfiguration(FrozenModel):
    kind: Literal['dense'] = 'dense'
    channels: int = Field(gt=0)
    spatial_reductions: int = Field(default=0, ge=0, le=MAXIMUM_DENSE_SPATIAL_REDUCTIONS)
    bottleneck_rank: int | None = Field(default=None, gt=0)


class GoPointPassPolicyHeadConfiguration(FrozenModel):
    kind: Literal['go_point_pass_v1'] = 'go_point_pass_v1'


PolicyHeadConfiguration: TypeAlias = Annotated[
    Chess76PlaneDirectPolicyHeadConfiguration | DensePolicyHeadConfiguration | GoPointPassPolicyHeadConfiguration,
    Field(discriminator='kind'),
]

POLICY_PLANE_PRIMARY_HIDDEN_CHANNELS = 64
POLICY_PLANE_AUXILIARY_HIDDEN_CHANNELS = 32
CHESS_POLICY_PLANE_COUNT = 76
SMALL_OUTPUT_INITIALIZATION_STD = 0.01
ATTENTION_LINEAR_INITIALIZATION_STD = 0.02


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
                self.start_block = nn.Sequential(
                    nn.Conv2d(encoding_channels, hidden_size, kernel_size=3, padding='same', bias=False),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU(inplace=True),
                )
                self.backbone = nn.ModuleList(
                    [
                        _build_residual_block(hidden_size, args.residual_context, block_index)
                        for block_index in range(args.num_layers)
                    ]
                )
                self.finish_block = nn.Identity()
            case AttentionNetworkParams():
                hidden_size = args.embedding_size
                self.start_block = AttentionInput(
                    encoding_channels,
                    row_count,
                    column_count,
                    hidden_size,
                )
                self.backbone = nn.ModuleList(
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
                self.finish_block = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    AttentionOutput(row_count, column_count),
                )

        self.policy_head = _build_policy_head(
            hidden_size,
            row_count,
            column_count,
            action_size,
            args.policy_head,
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_size, args.num_value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(args.num_value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(args.num_value_channels * row_count * column_count, args.value_fc_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.value_fc_size, dimensions.outcomes),
        )
        self.auxiliary_head_modules = nn.ModuleList(
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

        self._initialize_parameters(args)

        self.to(device=self.device, dtype=torch.float32)

    @torch.jit.unused
    def _initialize_parameters(self, args: NetworkConfiguration) -> None:
        for module in self.modules():
            match module:
                case nn.Conv2d() | nn.Linear():
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        match args:
            case AttentionNetworkParams(num_layers=num_layers):
                _initialize_attention_trunk(self.start_block, self.backbone, num_layers)
            case NetworkParams():
                pass
        _initialize_small_policy_output(self.policy_head, args.policy_head)
        _initialize_small_linear_output(self.value_head[-1])
        for auxiliary_module, auxiliary_layout in zip(self.auxiliary_head_modules, self.auxiliary_heads):
            match auxiliary_layout:
                case NextPolicyHeadLayout() | LegalMovesHeadLayout():
                    _initialize_small_policy_output(auxiliary_module, args.policy_head)
                case _:
                    _initialize_small_linear_output(auxiliary_module[-1])

    @torch.jit.unused
    def checkpoint_definition(self) -> NetworkDefinition:
        return NetworkDefinition(
            architecture=self.network_args,
            dimensions=self.dimensions,
            auxiliary_heads=self.auxiliary_heads,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        policy_logits, value_logits = self.logit_forward(x)
        return policy_logits, torch.softmax(value_logits, dim=1)

    def _features(self, x: Tensor) -> Tensor:
        x = self.start_block(x)
        for block in self.backbone:
            x = block(x)
        return self.finish_block(x)

    def logit_forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._features(x)
        policy_logits = self.policy_head(x)
        value_logits = self.value_head(x)

        return policy_logits, value_logits

    def training_output(self, x: Tensor) -> TrainingModelOutput:
        features = self._features(x)
        return TrainingModelOutput(
            policy_logits=self.policy_head(features),
            wdl_logits=self.value_head(features),
            auxiliary_logits=tuple(head(features) for head in self.auxiliary_head_modules),
        )

    def fuse_model(self) -> None:
        fuse_conv_batchnorm(self)


class ZeroSearchCorrectionHead(nn.Module):
    def forward(self, features: Tensor) -> Tensor:
        return torch.zeros((features.shape[0], 1), dtype=features.dtype, device=features.device)


class InferenceNetwork(nn.Module):
    def __init__(self, training_model: Network) -> None:
        super().__init__()
        self.start_block = copy.deepcopy(training_model.start_block)
        self.backbone = copy.deepcopy(training_model.backbone)
        self.finish_block = copy.deepcopy(training_model.finish_block)
        self.policy_head = copy.deepcopy(training_model.policy_head)
        self.value_head = copy.deepcopy(training_model.value_head)
        self.searchCorrectionHead = copy.deepcopy(_search_correction_head(training_model))
        self.network_definition = NetworkDefinition(
            architecture=training_model.network_args,
            dimensions=training_model.dimensions,
            auxiliary_heads=tuple(head for head in training_model.auxiliary_heads if head.kind == 'search_correction'),
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.start_block(inputs)
        for block in self.backbone:
            features = block(features)
        features = self.finish_block(features)
        return (
            self.policy_head(features),
            torch.softmax(self.value_head(features), dim=1),
            torch.sigmoid(self.searchCorrectionHead(features)),
        )

    @torch.jit.unused
    def checkpoint_definition(self) -> NetworkDefinition:
        return self.network_definition

    def fuse_model(self) -> None:
        fuse_conv_batchnorm(self)


# Deep BatchNorm CNNs get an accidental sharp random behaviour prior at generation 0 (eval-mode
# running-statistics mismatch, measured logit std 7-52), which the r3 recipe depends on to break
# search symmetry and bootstrap decisive self-play. LayerNorm trunks have no such artifact (measured
# 0.04-0.07) and stall in uniform search; this scale recreates the sharp prior deliberately, on the
# inference export only, so the training model still starts near-uniform for healthy optimization.
BOOTSTRAP_POLICY_PRIOR_LOGIT_SCALE = 75.0


def apply_bootstrap_policy_prior(inference_model: InferenceNetwork, args: NetworkConfiguration) -> None:
    match args:
        case AttentionNetworkParams():
            match inference_model.policy_head:
                case PolicyPlaneHead(output_projection=output_projection):
                    with torch.no_grad():
                        output_projection.weight.mul_(BOOTSTRAP_POLICY_PRIOR_LOGIT_SCALE)
                        output_projection.bias.mul_(BOOTSTRAP_POLICY_PRIOR_LOGIT_SCALE)


def fuse_conv_batchnorm(root: nn.Module) -> None:
    for module in root.modules():
        if type(module) is not nn.Sequential:
            continue
        groups = _conv_batchnorm_groups(module)
        if groups:
            torch.ao.quantization.fuse_modules(module, groups, inplace=True)


def _conv_batchnorm_groups(module: nn.Sequential) -> list[list[str]]:
    # Fuses every Conv2d, BatchNorm2d and optional trailing ReLU run, not only a leading one.
    groups: list[list[str]] = []
    index = 0
    while index + 1 < len(module):
        if isinstance(module[index], nn.Conv2d) and isinstance(module[index + 1], nn.BatchNorm2d):
            group = [str(index), str(index + 1)]
            if index + 2 < len(module) and isinstance(module[index + 2], nn.ReLU):
                group.append(str(index + 2))
            groups.append(group)
            index += len(group)
        else:
            index += 1
    return groups


def _search_correction_head(training_model: Network) -> nn.Module:
    for index, head in enumerate(training_model.auxiliary_heads):
        match head:
            case SearchCorrectionHeadLayout():
                return training_model.auxiliary_head_modules[index]
    return ZeroSearchCorrectionHead()


class PolicyPlaneHead(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, plane_count: int) -> None:
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_block = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.output_projection = nn.Conv2d(hidden_channels, plane_count, kernel_size=1, bias=True)

    def forward(self, features: Tensor) -> Tensor:
        return self.output_projection(self.spatial_block(self.input_block(features))).flatten(start_dim=1)


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
    plane_hidden_channels: int = POLICY_PLANE_PRIMARY_HIDDEN_CHANNELS,
) -> nn.Module:
    match configuration:
        case Chess76PlaneDirectPolicyHeadConfiguration():
            if row_count != 8 or column_count != 8 or action_size != CHESS_POLICY_PLANE_COUNT * 64:
                raise ValueError(
                    f'Chess direct policy heads require an 8x8 board and '
                    f'{CHESS_POLICY_PLANE_COUNT * 64} actions, not {action_size}; '
                    f'the reduced chess action encoding needs a dense policy head.'
                )
            return PolicyPlaneHead(input_channels, plane_hidden_channels, CHESS_POLICY_PLANE_COUNT)
        case DensePolicyHeadConfiguration():
            return _build_dense_policy_head(input_channels, row_count, column_count, action_size, configuration)
        case GoPointPassPolicyHeadConfiguration():
            if action_size != row_count * column_count + 1:
                raise ValueError('Go point-pass policy heads require one action per point plus pass.')
            return GoPointPassPolicyHead(input_channels)


def _build_dense_policy_head(
    input_channels: int,
    row_count: int,
    column_count: int,
    action_size: int,
    configuration: DensePolicyHeadConfiguration,
) -> nn.Sequential:
    reduced_rows = row_count - 2 * configuration.spatial_reductions
    reduced_columns = column_count - 2 * configuration.spatial_reductions
    if reduced_rows < 1 or reduced_columns < 1:
        raise ValueError(
            f'{configuration.spatial_reductions} unpadded 3x3 reductions do not fit a {row_count}x{column_count} board.'
        )
    layers: list[nn.Module] = []
    for _ in range(configuration.spatial_reductions):
        layers.append(nn.Conv2d(input_channels, input_channels, kernel_size=3, bias=False))
        layers.append(nn.BatchNorm2d(input_channels))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(input_channels, configuration.channels, kernel_size=1, bias=False))
    layers.append(nn.BatchNorm2d(configuration.channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Flatten())
    flattened_features = configuration.channels * reduced_rows * reduced_columns
    match configuration.bottleneck_rank:
        case None:
            layers.append(nn.Linear(flattened_features, action_size))
        case bottleneck_rank:
            layers.append(nn.Linear(flattened_features, bottleneck_rank))
            layers.append(nn.Linear(bottleneck_rank, action_size))
    return nn.Sequential(*layers)


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
                plane_hidden_channels=POLICY_PLANE_AUXILIARY_HIDDEN_CHANNELS,
            )
        case RemainingGameLengthHeadLayout(output_size=output_size):
            return _build_scalar_auxiliary_head(input_channels, row_count, column_count, output_size)
        case FutureSearchValueHeadLayout(output_size=output_size):
            return _build_scalar_auxiliary_head(input_channels, row_count, column_count, output_size)
        case IrreversibleProgressHeadLayout(output_size=output_size):
            return _build_scalar_auxiliary_head(input_channels, row_count, column_count, output_size)
        case LegalMovesHeadLayout(action_size=legal_action_size):
            if legal_action_size != action_size:
                raise ValueError('Legal-moves action space must match the primary policy action space.')
            return _build_policy_head(
                input_channels,
                row_count,
                column_count,
                action_size,
                policy_configuration,
                plane_hidden_channels=POLICY_PLANE_AUXILIARY_HIDDEN_CHANNELS,
            )
        case SearchCorrectionHeadLayout(output_size=output_size):
            return _build_scalar_auxiliary_head(input_channels, row_count, column_count, output_size)


def _initialize_attention_trunk(
    start_block: nn.Module,
    back_bone: nn.ModuleList,
    num_layers: int,
) -> None:
    residual_output_std = ATTENTION_LINEAR_INITIALIZATION_STD / math.sqrt(2 * num_layers)
    match start_block:
        case AttentionInput(projection=projection):
            nn.init.normal_(projection.weight, std=ATTENTION_LINEAR_INITIALIZATION_STD)
            nn.init.zeros_(projection.bias)
    for block in back_bone:
        match block:
            case AttentionEncoderBlock():
                for linear in (block.query_key_value_projection, block.feedforward[0]):
                    nn.init.normal_(linear.weight, std=ATTENTION_LINEAR_INITIALIZATION_STD)
                    nn.init.zeros_(linear.bias)
                for residual_projection in (block.attention_output_projection, block.feedforward[3]):
                    nn.init.normal_(residual_projection.weight, std=residual_output_std)
                    nn.init.zeros_(residual_projection.bias)


def _initialize_small_policy_output(module: nn.Module, configuration: PolicyHeadConfiguration) -> None:
    match configuration:
        case Chess76PlaneDirectPolicyHeadConfiguration():
            assert isinstance(module, PolicyPlaneHead), 'The 76-plane policy configuration must build a plane head.'
            nn.init.normal_(module.output_projection.weight, std=SMALL_OUTPUT_INITIALIZATION_STD)
            nn.init.zeros_(module.output_projection.bias)
        case DensePolicyHeadConfiguration(bottleneck_rank=bottleneck_rank):
            assert isinstance(module, nn.Sequential), 'The dense policy configuration must build a sequential head.'
            _initialize_small_linear_output(module[-1])
            if bottleneck_rank is not None:
                # Kaiming on the first factor stacks with the BN eval-mode blowup and over-sharpens the
                # generation-0 behaviour prior far beyond the validated logit-std band (measured 256 vs
                # the proven 7-52); halving the first factor's std lands the export near band-centre (~32).
                first_factor = module[-2]
                assert isinstance(first_factor, nn.Linear)
                nn.init.normal_(first_factor.weight, std=SMALL_OUTPUT_INITIALIZATION_STD / 2)
                nn.init.zeros_(first_factor.bias)
        case GoPointPassPolicyHeadConfiguration():
            # The Go point-pass head keeps its historical Kaiming initialization.
            pass


def _initialize_small_linear_output(module: nn.Module) -> None:
    assert isinstance(module, nn.Linear), 'Small-output initialization expects the final head layer to be linear.'
    nn.init.normal_(module.weight, std=SMALL_OUTPUT_INITIALIZATION_STD)
    nn.init.zeros_(module.bias)


def _build_scalar_auxiliary_head(
    input_channels: int,
    row_count: int,
    column_count: int,
    output_size: int,
) -> nn.Module:
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

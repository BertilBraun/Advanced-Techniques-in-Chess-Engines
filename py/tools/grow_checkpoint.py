"""Grow a trained checkpoint into the next progressive stage without changing what it plays.

Every new unit is wired random-in and zero-out: it computes a nonzero activation, but the weights
that read it are zero, so the network's output is unchanged while the gradient that reaches those
reading weights is not. Zeroing both sides instead would preserve the function and leave the new
capacity permanently dead.

Two details of this architecture do not survive a plain channel copy. The residual branch scale is
1/sqrt(depth), so a block copied into a deeper trunk contributes less than it did; the compensation
goes on the affine parameters of the batch norm that ends the branch, because scaling the
convolution in front of it is undone by the normalisation. And a global-pooling block splits its
first convolution's output positionally at hidden // 4, so widening moves that boundary and the
copy has to permute around it or channels change role.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.training.checkpoint.contracts import read_checkpoint_manifest
from src.training.network import (
    GlobalPoolingResidualContext,
    Network,
    NetworkParams,
    ScaledPostActivationResidualBlockConfiguration,
)
from src.training.progressive import ProgressiveModelDefinition, ProgressiveModelSizingConfiguration
from torch import Tensor

NEW_UNIT_INITIAL_SCALE = 0.02


@dataclass(frozen=True)
class ChannelMap:
    """Where each source channel lands in the grown tensor, and which targets are new."""

    source_to_target: dict[int, int]
    target_width: int

    @property
    def new_targets(self) -> tuple[int, ...]:
        taken = set(self.source_to_target.values())
        return tuple(index for index in range(self.target_width) if index not in taken)


def _trunk_map(source_width: int, target_width: int) -> ChannelMap:
    return ChannelMap({index: index for index in range(source_width)}, target_width)


def _pooling_map(source_width: int, target_width: int) -> ChannelMap:
    source_global = max(1, source_width // 4)
    target_global = max(1, target_width // 4)
    mapping = {index: index for index in range(source_global)}
    for index in range(source_global, source_width):
        mapping[index] = index - source_global + target_global
    return ChannelMap(mapping, target_width)


def _local_map(source_width: int, target_width: int) -> ChannelMap:
    source_local = source_width - max(1, source_width // 4)
    target_local = target_width - max(1, target_width // 4)
    return ChannelMap({index: index for index in range(source_local)}, target_local)


def _pooling_projection_input_map(source_width: int, target_width: int) -> ChannelMap:
    # The projection reads mean and maximum of every global channel, concatenated in that order,
    # so widening moves the maxima block as well as extending each half.
    source_global = max(1, source_width // 4)
    target_global = max(1, target_width // 4)
    mapping = {index: index for index in range(source_global)}
    for index in range(source_global):
        mapping[source_global + index] = target_global + index
    return ChannelMap(mapping, target_global * 2)


def _grow_tensor(
    source: Tensor,
    target: Tensor,
    output_map: ChannelMap | None,
    input_map: ChannelMap | None,
    output_scale: float,
    generator: torch.Generator,
) -> Tensor:
    grown = target.clone()
    output_indices = (
        torch.tensor(sorted(output_map.source_to_target), dtype=torch.long) if output_map is not None else None
    )
    if output_map is not None:
        assert output_indices is not None
        target_rows = torch.tensor(
            [output_map.source_to_target[int(index)] for index in output_indices], dtype=torch.long
        )
        selected = source.index_select(0, output_indices)
        if input_map is not None:
            input_indices = torch.tensor(sorted(input_map.source_to_target), dtype=torch.long)
            target_columns = torch.tensor(
                [input_map.source_to_target[int(index)] for index in input_indices], dtype=torch.long
            )
            selected = selected.index_select(1, input_indices)
            # Columns the source never had read new units, and stay zero so the output is unchanged.
            grown.index_fill_(1, torch.tensor(input_map.new_targets, dtype=torch.long), 0.0)
            block = grown.index_select(0, target_rows)
            block.index_copy_(1, target_columns, selected * output_scale)
            grown.index_copy_(0, target_rows, block)
        else:
            grown.index_copy_(0, target_rows, selected * output_scale)
        # Rows the source never had produce new units, and keep a small random value so they carry
        # a nonzero activation for the zeroed readers above to earn a gradient from.
        new_rows = torch.tensor(output_map.new_targets, dtype=torch.long)
        if new_rows.numel():
            noise = torch.empty((new_rows.numel(), *grown.shape[1:]), dtype=grown.dtype).normal_(
                0.0, NEW_UNIT_INITIAL_SCALE, generator=generator
            )
            grown.index_copy_(0, new_rows, noise)
        return grown
    assert input_map is not None
    input_indices = torch.tensor(sorted(input_map.source_to_target), dtype=torch.long)
    target_columns = torch.tensor([input_map.source_to_target[int(index)] for index in input_indices], dtype=torch.long)
    grown.index_fill_(1, torch.tensor(input_map.new_targets, dtype=torch.long), 0.0)
    grown.index_copy_(1, target_columns, source.index_select(1, input_indices) * output_scale)
    return grown


def _grow_vector(source: Tensor, target: Tensor, channel_map: ChannelMap, scale: float, fill: float) -> Tensor:
    grown = target.clone()
    indices = torch.tensor(sorted(channel_map.source_to_target), dtype=torch.long)
    rows = torch.tensor([channel_map.source_to_target[int(index)] for index in indices], dtype=torch.long)
    grown.index_copy_(0, rows, source.index_select(0, indices) * scale)
    new_rows = torch.tensor(channel_map.new_targets, dtype=torch.long)
    if new_rows.numel():
        grown.index_fill_(0, new_rows, fill)
    return grown


def _normalization_fill(key: str) -> float:
    if key.endswith('running_var') or key.endswith('.weight'):
        return 1.0
    return 0.0


def grow_state_dict(
    source: dict[str, Tensor],
    target: dict[str, Tensor],
    source_width: int,
    target_width: int,
    source_depth: int,
    branch_scale_ratio: float,
    seed: int,
) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    grown = {key: value.clone() for key, value in target.items()}
    pooling_blocks = {
        int(key.split('.')[1]) for key in target if key.startswith('backbone.') and 'global_pooling_bias' in key
    }
    for key, target_value in target.items():
        if key.endswith('num_batches_tracked'):
            continue
        parts = key.split('.')
        is_backbone = parts[0] == 'backbone'
        block_index = int(parts[1]) if is_backbone else -1
        if is_backbone and block_index >= source_depth:
            # An appended block: its branch ends zeroed so the trunk passes through untouched, and
            # the gradient at that zero is the activation in front of it, which is not zero.
            if parts[2] == 'conv_block2' and parts[3] == '0':
                grown[key] = torch.zeros_like(target_value)
            elif parts[2] == 'conv_block2' and key.endswith('.bias'):
                grown[key] = torch.zeros_like(target_value)
            continue
        if key not in source:
            continue
        source_value = source[key]
        if source_value.shape == target_value.shape:
            grown[key] = source_value.clone()
            continue
        pooling = block_index in pooling_blocks
        in_branch_output = is_backbone and parts[2] == 'conv_block2'
        # Only the affine parameters carry the branch-scale compensation: the running statistics
        # describe the distribution being normalised, and scaling them would change the
        # normalisation rather than the output.
        scale = (
            branch_scale_ratio
            if in_branch_output and source_value.dim() == 1 and key.endswith(('.weight', '.bias'))
            else 1.0
        )
        if source_value.dim() == 1:
            channel_map = (
                _pooling_map(source_width, target_width)
                if pooling and is_backbone and parts[2] == 'conv_block1'
                else _local_map(source_width, target_width)
                if pooling and is_backbone and parts[2] == 'global_pooling_bias'
                else _trunk_map(source_width, target_width)
            )
            grown[key] = _grow_vector(source_value, target_value, channel_map, scale, _normalization_fill(key))
            continue
        output_map: ChannelMap | None = None
        input_map: ChannelMap | None = None
        if source_value.shape[0] != target_value.shape[0]:
            if pooling and is_backbone and parts[2] == 'conv_block1':
                output_map = _pooling_map(source_width, target_width)
            elif pooling and is_backbone and parts[2] == 'global_pooling_bias':
                output_map = _local_map(source_width, target_width)
            else:
                output_map = _trunk_map(source_width, target_width)
        if source_value.shape[1] != target_value.shape[1]:
            if pooling and is_backbone and parts[2] == 'global_pooling_bias':
                input_map = _pooling_projection_input_map(source_width, target_width)
            elif pooling and in_branch_output:
                input_map = _local_map(source_width, target_width)
            else:
                input_map = _trunk_map(source_width, target_width)
        grown[key] = _grow_tensor(source_value, target_value, output_map, input_map, scale, generator)
    return grown


def _model(definition: ProgressiveModelDefinition, game: ChessImplementation) -> Network:
    return Network(
        definition.network,
        torch.device('cpu'),
        game.network_dimensions,
        game.target_layout.auxiliary_heads,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-config', required=True, type=Path)
    parser.add_argument('--source-run-state', required=True, type=Path)
    parser.add_argument('--source-generation', required=True, type=int)
    parser.add_argument('--source-model-id', required=True)
    parser.add_argument('--target-model-id', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', default=20260922, type=int)
    arguments = parser.parse_args()

    experiment = load_experiment_configuration(arguments.experiment_config)
    if not isinstance(experiment, ChessExperimentConfiguration):
        raise ValueError('Growing a checkpoint requires a chess experiment configuration.')
    sizing = experiment.training.progressive_model_sizing
    if not isinstance(sizing, ProgressiveModelSizingConfiguration):
        raise ValueError('Growing a checkpoint requires a progressive model-sizing configuration.')
    source_definition = sizing.model(arguments.source_model_id)
    target_definition = sizing.model(arguments.target_model_id)
    if not isinstance(source_definition.network, NetworkParams) or not isinstance(
        target_definition.network, NetworkParams
    ):
        raise ValueError('Growing a checkpoint is only defined for convolutional stages.')
    if not isinstance(source_definition.network.residual_context, GlobalPoolingResidualContext):
        raise ValueError('Growing a checkpoint expects the global-pooling residual context.')
    for network in (source_definition.network, target_definition.network):
        if not isinstance(network.residual_block, ScaledPostActivationResidualBlockConfiguration):
            raise ValueError('Growing a checkpoint expects scaled post-activation residual blocks.')

    game = ChessImplementation(experiment)
    manifest = read_checkpoint_manifest(arguments.source_generation, arguments.source_run_state)
    del manifest
    source_state = torch.load(
        arguments.source_run_state / f'model_{arguments.source_generation}.pt',
        map_location='cpu',
        weights_only=True,
    )
    source_state = {key.removeprefix('_orig_mod.'): value for key, value in source_state.items()}
    target_state = _model(target_definition, game).state_dict()

    source_scale = source_definition.network.residual_block.branch_scale
    target_scale = target_definition.network.residual_block.branch_scale
    grown = grow_state_dict(
        source_state,
        target_state,
        source_definition.network.hidden_size,
        target_definition.network.hidden_size,
        source_definition.network.num_layers,
        source_scale / target_scale,
        arguments.seed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grown, arguments.output)
    print(
        f'source {source_definition.network.num_layers}x{source_definition.network.hidden_size} '
        f'branch_scale {source_scale:.6f} (1/sqrt = {1 / math.sqrt(source_definition.network.num_layers):.6f})'
    )
    print(
        f'target {target_definition.network.num_layers}x{target_definition.network.hidden_size} '
        f'branch_scale {target_scale:.6f}'
    )
    print(f'branch compensation {source_scale / target_scale:.6f}')
    print(f'wrote {arguments.output}')


if __name__ == '__main__':
    main()

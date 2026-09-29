"""Grow a distilled student checkpoint into a wider and deeper trunk that plays identically at the start."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from src.distillation.teacher import read_network_definition
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.checkpoint.persistence import create_model
from src.training.network import (
    GlobalPoolingResidualContext,
    Network,
    NetworkParams,
    ScaledPostActivationResidualBlockConfiguration,
)
from src.util.log import log
from tools.distill_train_student import (
    OptimizerKind,
    create_student_optimizer,
    load_initial_weights,
    save_student_checkpoint,
)
from tools.grow_checkpoint import grow_state_dict

FIDELITY_TOLERANCE = 1e-3


def grown_architecture(source: NetworkParams, layers: int, hidden_size: int) -> NetworkParams:
    match source.residual_block:
        case ScaledPostActivationResidualBlockConfiguration(activation_cap=activation_cap):
            block = ScaledPostActivationResidualBlockConfiguration(
                branch_scale=1.0 / math.sqrt(layers), activation_cap=activation_cap
            )
        case _:
            raise ValueError('Growing a student expects scaled post-activation residual blocks.')
    if not isinstance(source.residual_context, GlobalPoolingResidualContext):
        raise ValueError('Growing a student expects the global-pooling residual context.')
    if layers < source.num_layers or hidden_size < source.hidden_size:
        raise ValueError('A grown student must be at least as deep and as wide as its source.')
    return source.model_copy(update={'num_layers': layers, 'hidden_size': hidden_size, 'residual_block': block})


def grow_student(source_model: Network, source: NetworkParams, target: NetworkParams, seed: int) -> Network:
    target_model = create_model(target, torch.device('cpu'), CHESS_NETWORK_DIMENSIONS)
    assert isinstance(source.residual_block, ScaledPostActivationResidualBlockConfiguration)
    assert isinstance(target.residual_block, ScaledPostActivationResidualBlockConfiguration)
    grown = grow_state_dict(
        source_model.state_dict(),
        target_model.state_dict(),
        source.hidden_size,
        target.hidden_size,
        source.num_layers,
        source.residual_block.branch_scale / target.residual_block.branch_scale,
        seed,
    )
    target_model.load_state_dict(grown, strict=True)
    return target_model


def largest_output_difference(first: Network, second: Network, states: torch.Tensor) -> float:
    first.eval()
    second.eval()
    with torch.no_grad():
        first_output = first.training_output(states)
        second_output = second.training_output(states)
    return max(
        (first_output.policy_logits - second_output.policy_logits).abs().max().item(),
        (first_output.wdl_logits - second_output.wdl_logits).abs().max().item(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', required=True, type=Path, help='checkpoint_N.json of the student.')
    parser.add_argument('--layers', required=True, type=int)
    parser.add_argument('--hidden-size', required=True, type=int)
    parser.add_argument('--output-run-state', required=True, type=Path)
    parser.add_argument('--generation', required=True, type=int)
    parser.add_argument('--seed', default=20260927, type=int)
    arguments = parser.parse_args()

    definition = read_network_definition(arguments.source_checkpoint)
    if definition is None or not isinstance(definition.architecture, NetworkParams):
        raise ValueError(f'{arguments.source_checkpoint} carries no convolutional network definition.')
    source = definition.architecture
    target = grown_architecture(source, arguments.layers, arguments.hidden_size)
    source_model = create_model(source, torch.device('cpu'), CHESS_NETWORK_DIMENSIONS)
    load_initial_weights(source_model, arguments.source_checkpoint, torch.device('cpu'))
    target_model = grow_student(source_model, source, target, arguments.seed)

    dimensions = CHESS_NETWORK_DIMENSIONS
    states = torch.randint(
        0, 2, (64, dimensions.channels, dimensions.rows, dimensions.columns), generator=torch.Generator().manual_seed(1)
    )
    difference = largest_output_difference(source_model, target_model, states.float())
    log(
        f'grew {source.num_layers}x{source.hidden_size} into {target.num_layers}x{target.hidden_size}: '
        f'{sum(parameter.numel() for parameter in source_model.parameters()):,} -> '
        f'{sum(parameter.numel() for parameter in target_model.parameters()):,} parameters, '
        f'largest output difference {difference:.2e}'
    )
    if difference > FIDELITY_TOLERANCE:
        raise SystemExit(f'The grown student does not reproduce its source: difference {difference:.2e}.')
    arguments.output_run_state.mkdir(parents=True, exist_ok=True)
    optimizer = create_student_optimizer(target_model, OptimizerKind.ADAMW, 1e-4)
    save_student_checkpoint(target_model, optimizer, arguments.generation, arguments.output_run_state)
    log(f'wrote the grown generation {arguments.generation} student to {arguments.output_run_state}')


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from src.training.architecture_catalog import load_architecture_catalog
from src.training.network import Network


@dataclass(frozen=True)
class ModelPreparationArguments:
    catalog_path: Path
    model_id: str
    output_path: Path
    random_seed: int


def _parse_arguments() -> ModelPreparationArguments:
    parser = argparse.ArgumentParser(description='Export a catalog architecture for native inference benchmarking.')
    parser.add_argument(
        '--catalog',
        type=Path,
        default=Path('configs/architectures/chess-cnn-attention-v1.yaml'),
    )
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--random-seed', type=int, default=20260818)
    parsed = parser.parse_args()
    if not parsed.output.name.endswith('.jit.pt'):
        raise ValueError('Architecture benchmark model output must end in .jit.pt.')
    return ModelPreparationArguments(
        catalog_path=parsed.catalog,
        model_id=parsed.model_id,
        output_path=parsed.output,
        random_seed=parsed.random_seed,
    )


def prepare_architecture_benchmark_model(arguments: ModelPreparationArguments) -> int:
    catalog = load_architecture_catalog(arguments.catalog_path)
    matches = tuple(entry for entry in catalog.models if entry.model_id == arguments.model_id)
    if len(matches) != 1:
        raise ValueError(f'Architecture catalog must contain exactly one model named {arguments.model_id}.')
    entry = matches[0]
    definition = entry.definition
    torch.manual_seed(arguments.random_seed)
    network = Network(
        definition.architecture,
        torch.device('cpu'),
        definition.dimensions,
        definition.auxiliary_output_sizes,
    )
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    if parameter_count != entry.expected_training_parameters:
        raise ValueError('Constructed parameter count disagrees with the architecture catalog.')
    network.auxiliaryHeads = nn.ModuleList()
    network.auxiliary_output_sizes = ()
    network.eval()
    network.fuse_model()
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(torch.jit.script(network), str(arguments.output_path))
    return parameter_count


def main() -> None:
    arguments = _parse_arguments()
    parameter_count = prepare_architecture_benchmark_model(arguments)
    print(f'parameters={parameter_count}')
    print(f'bytes={arguments.output_path.stat().st_size}')


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.experiment.configuration import load_experiment_configuration
from src.games.composition import create_game_implementation
from src.training.network import InferenceNetwork, Network


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--zero-parameters', action='store_true')
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if not arguments.output.name.endswith('.jit.pt'):
        raise ValueError('Benchmark model output must end in .jit.pt.')
    torch.manual_seed(arguments.seed)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    experiment = load_experiment_configuration(arguments.run_config)
    game = create_game_implementation(experiment)
    network = Network(
        experiment.training.network,
        torch.device('cpu'),
        game.network_dimensions,
        game.target_layout.auxiliary_heads,
    )
    if arguments.zero_parameters:
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
    inference_network = InferenceNetwork(network)
    inference_network.eval()
    inference_network.fuse_model()
    torch.jit.script(inference_network).save(str(arguments.output))
    print(f'parameters={sum(parameter.numel() for parameter in network.parameters())}')
    print(f'bytes={arguments.output.stat().st_size}')


if __name__ == '__main__':
    main()

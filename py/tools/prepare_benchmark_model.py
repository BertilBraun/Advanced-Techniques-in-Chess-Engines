from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.Network import Network
from src.train.TrainingArgs import NetworkParams


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--layers', required=True, type=int)
    parser.add_argument('--hidden-size', required=True, type=int)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)
    network = Network(
        NetworkParams(
            num_layers=arguments.layers,
            hidden_size=arguments.hidden_size,
            se_positions=(1, 3),
        ),
        torch.device('cpu'),
    )
    network.eval()
    network.fuse_model()
    torch.jit.script(network).save(str(arguments.output))
    print(f'parameters={sum(parameter.numel() for parameter in network.parameters())}')
    print(f'bytes={arguments.output.stat().st_size}')


if __name__ == '__main__':
    main()

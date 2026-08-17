from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.training.architecture_catalog import load_architecture_catalog


@dataclass(frozen=True)
class ReplayGenerationArguments:
    catalog_path: Path
    output_path: Path
    sample_count: int
    random_seed: int


def _parse_arguments() -> ReplayGenerationArguments:
    parser = argparse.ArgumentParser(description='Create a deterministic synthetic chess architecture replay.')
    parser.add_argument(
        '--catalog',
        type=Path,
        default=Path('configs/architectures/chess-cnn-attention-v1.yaml'),
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=8192)
    parser.add_argument('--random-seed', type=int, default=20260817)
    parsed = parser.parse_args()
    if parsed.sample_count <= 0:
        raise ValueError('Synthetic replay sample count must be positive.')
    return ReplayGenerationArguments(
        catalog_path=parsed.catalog,
        output_path=parsed.output,
        sample_count=parsed.sample_count,
        random_seed=parsed.random_seed,
    )


def create_synthetic_architecture_replay(arguments: ReplayGenerationArguments) -> None:
    catalog = load_architecture_catalog(arguments.catalog_path)
    dimensions = {entry.definition.dimensions for entry in catalog.models}
    auxiliary_output_sizes = {entry.definition.auxiliary_output_sizes for entry in catalog.models}
    if len(dimensions) != 1 or len(auxiliary_output_sizes) != 1:
        raise ValueError('Architecture catalog models must share one output contract.')
    network_dimensions = dimensions.pop()
    auxiliary_sizes = auxiliary_output_sizes.pop()
    if auxiliary_sizes != (network_dimensions.actions, 1):
        raise ValueError('Synthetic replay requires next-policy and remaining-length auxiliary heads.')

    generator = np.random.default_rng(arguments.random_seed)
    row_indices = np.arange(arguments.sample_count)
    states = generator.integers(
        -1,
        2,
        size=(arguments.sample_count, network_dimensions.channels, network_dimensions.rows, network_dimensions.columns),
        dtype=np.int8,
    )
    policy_targets = np.zeros((arguments.sample_count, network_dimensions.actions), dtype=np.float16)
    policy_targets[row_indices, generator.integers(0, network_dimensions.actions, size=arguments.sample_count)] = 1
    wdl_targets = np.zeros((arguments.sample_count, network_dimensions.outcomes), dtype=np.float16)
    wdl_targets[row_indices, generator.integers(0, network_dimensions.outcomes, size=arguments.sample_count)] = 1
    next_policy_targets = np.zeros((arguments.sample_count, network_dimensions.actions), dtype=np.float16)
    next_policy_targets[
        row_indices,
        generator.integers(0, network_dimensions.actions, size=arguments.sample_count),
    ] = 1
    remaining_length_targets = generator.uniform(0, 400, size=(arguments.sample_count, 1)).astype(np.float32)

    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.output_path,
        states=states,
        policy_targets=policy_targets,
        wdl_targets=wdl_targets,
        next_policy_targets=next_policy_targets,
        remaining_length_targets=remaining_length_targets,
    )


def main() -> None:
    create_synthetic_architecture_replay(_parse_arguments())


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from tools.v34_orchestration import (
    ChildCommand,
    parse_devices,
    run_child_commands,
    validate_input_paths,
    validate_new_output_root,
)


@dataclass(frozen=True)
class Arguments:
    experiment: Path
    run_directory: Path
    checkpoint_generation: int
    opening_manifest: Path
    stockfish_executable: Path
    output_root: Path
    ten_thousand_devices: tuple[int, ...]
    eighty_thousand_devices: tuple[int, ...]
    opening_selection_seed: int
    match_random_seed: int
    dry_run: bool


def _ladder_command(
    arguments: Arguments,
    model_searches: int,
    parallel_searches: int,
    devices: tuple[int, ...],
    stockfish_nodes: tuple[int, ...],
    output_directory: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        '-m',
        'tools.run_stockfish_ladder',
        '--experiment',
        str(arguments.experiment),
        '--run-directory',
        str(arguments.run_directory),
        '--checkpoint-generation',
        str(arguments.checkpoint_generation),
        '--opening-manifest',
        str(arguments.opening_manifest),
        '--stockfish-executable',
        str(arguments.stockfish_executable),
        '--stockfish-node-ladder',
        *(str(nodes) for nodes in stockfish_nodes),
        '--probe-games',
        '10',
        '--opening-selection-seed',
        str(arguments.opening_selection_seed),
        '--match-random-seed',
        str(arguments.match_random_seed),
        '--devices',
        *(str(device) for device in devices),
        '--model-searches',
        str(model_searches),
        '--parallel-searches',
        str(parallel_searches),
        '--inference-workers',
        '1',
        '--inference-batch-size',
        '64',
        '--outstanding-batches',
        '1',
        '--exploration-constant',
        '1.0',
        '--output-directory',
        str(output_directory),
    )


def child_commands(arguments: Arguments) -> tuple[ChildCommand, ...]:
    return (
        ChildCommand(
            name='ladder-10k',
            command=_ladder_command(
                arguments,
                model_searches=10_000,
                parallel_searches=4,
                devices=arguments.ten_thousand_devices,
                stockfish_nodes=(50_000, 100_000),
                output_directory=arguments.output_root / 'ladder-10k',
            ),
            log_path=arguments.output_root / 'ladder-10k.log',
        ),
        ChildCommand(
            name='ladder-80k',
            command=_ladder_command(
                arguments,
                model_searches=80_000,
                parallel_searches=8,
                devices=arguments.eighty_thousand_devices,
                stockfish_nodes=(50_000, 100_000, 200_000),
                output_directory=arguments.output_root / 'ladder-80k',
            ),
            log_path=arguments.output_root / 'ladder-80k.log',
        ),
    )


def run_ladders(arguments: Arguments) -> int:
    outcomes = run_child_commands(child_commands(arguments), arguments.dry_run)
    return int(any(outcome.return_code != 0 for outcome in outcomes))


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Run the two v34 terminal-checkpoint Stockfish ladders concurrently.')
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--run-directory', required=True, type=Path)
    parser.add_argument('--checkpoint-generation', required=True, type=int)
    parser.add_argument('--opening-manifest', required=True, type=Path)
    parser.add_argument('--stockfish-executable', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--ten-thousand-devices', nargs='+', default=(0, 1, 2), type=int)
    parser.add_argument('--eighty-thousand-devices', nargs='+', default=(3, 4, 5, 6, 7), type=int)
    parser.add_argument('--opening-selection-seed', default=20260815, type=int)
    parser.add_argument('--match-random-seed', default=20260816, type=int)
    parser.add_argument('--dry-run', action='store_true')
    namespace = parser.parse_args()
    arguments = Arguments(
        experiment=namespace.experiment,
        run_directory=namespace.run_directory,
        checkpoint_generation=namespace.checkpoint_generation,
        opening_manifest=namespace.opening_manifest,
        stockfish_executable=namespace.stockfish_executable,
        output_root=namespace.output_root,
        ten_thousand_devices=parse_devices(tuple(namespace.ten_thousand_devices), '10k ladder'),
        eighty_thousand_devices=parse_devices(tuple(namespace.eighty_thousand_devices), '80k ladder'),
        opening_selection_seed=namespace.opening_selection_seed,
        match_random_seed=namespace.match_random_seed,
        dry_run=namespace.dry_run,
    )
    validate_input_paths(
        (arguments.experiment, arguments.run_directory, arguments.opening_manifest, arguments.stockfish_executable)
    )
    validate_new_output_root(arguments.output_root)
    if set(arguments.ten_thousand_devices) & set(arguments.eighty_thousand_devices):
        raise ValueError('The 10k and 80k ladders require disjoint GPU sets.')
    if min(arguments.checkpoint_generation, arguments.opening_selection_seed, arguments.match_random_seed) < 0:
        raise ValueError('Checkpoint generation and random seeds must be nonnegative.')
    return arguments


def main() -> None:
    raise SystemExit(run_ladders(parse_arguments()))


if __name__ == '__main__':
    main()

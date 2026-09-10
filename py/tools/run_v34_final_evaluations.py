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
    devices: tuple[int, ...]
    policy_only_stockfish_nodes: int
    shallow_stockfish_nodes: int
    deep_stockfish_nodes: int
    very_deep_stockfish_nodes: int
    match_random_seed: int
    dry_run: bool


@dataclass(frozen=True)
class EvaluationMode:
    name: str
    stockfish_nodes: int
    budget_arguments: tuple[str, ...]


def _modes(arguments: Arguments) -> tuple[EvaluationMode, ...]:
    return (
        EvaluationMode('policy-only', arguments.policy_only_stockfish_nodes, ('--model-policy-only',)),
        EvaluationMode(
            'search-64', arguments.shallow_stockfish_nodes, ('--model-searches', '64', '--parallel-searches', '1')
        ),
        EvaluationMode(
            'search-10000', arguments.deep_stockfish_nodes, ('--model-searches', '10000', '--parallel-searches', '4')
        ),
        EvaluationMode(
            'search-80000',
            arguments.very_deep_stockfish_nodes,
            ('--model-searches', '80000', '--parallel-searches', '8'),
        ),
    )


def _command(arguments: Arguments, mode: EvaluationMode) -> tuple[str, ...]:
    return (
        sys.executable,
        '-m',
        'tools.run_stockfish_gauntlet',
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
        '--stockfish-nodes',
        str(mode.stockfish_nodes),
        '--all-opening-pairs',
        '--opening-selection',
        'prefix',
        '--match-random-seed',
        str(arguments.match_random_seed),
        '--devices',
        *(str(device) for device in arguments.devices),
        *mode.budget_arguments,
        '--inference-workers',
        '1',
        '--inference-batch-size',
        '64',
        '--outstanding-batches',
        '1',
        '--exploration-constant',
        '1.0',
        '--output-directory',
        str(arguments.output_root / mode.name),
    )


def child_commands(arguments: Arguments) -> tuple[ChildCommand, ...]:
    return tuple(
        ChildCommand(mode.name, _command(arguments, mode), arguments.output_root / f'{mode.name}.log')
        for mode in _modes(arguments)
    )


def run_evaluations(arguments: Arguments) -> int:
    outcomes = run_child_commands(child_commands(arguments), arguments.dry_run)
    for outcome in outcomes:
        if outcome.return_code == 0:
            print(f'{outcome.name} result: {arguments.output_root / outcome.name / "result.json"}', flush=True)
    return int(any(outcome.return_code != 0 for outcome in outcomes))


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Run all four v34 final Stockfish evaluations concurrently.')
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--run-directory', required=True, type=Path)
    parser.add_argument('--checkpoint-generation', required=True, type=int)
    parser.add_argument('--opening-manifest', required=True, type=Path)
    parser.add_argument('--stockfish-executable', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--devices', nargs='+', default=tuple(range(8)), type=int)
    parser.add_argument('--policy-only-stockfish-nodes', required=True, type=int)
    parser.add_argument('--shallow-stockfish-nodes', required=True, type=int)
    parser.add_argument('--deep-stockfish-nodes', required=True, type=int)
    parser.add_argument('--very-deep-stockfish-nodes', required=True, type=int)
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
        devices=parse_devices(tuple(namespace.devices), 'Final evaluation'),
        policy_only_stockfish_nodes=namespace.policy_only_stockfish_nodes,
        shallow_stockfish_nodes=namespace.shallow_stockfish_nodes,
        deep_stockfish_nodes=namespace.deep_stockfish_nodes,
        very_deep_stockfish_nodes=namespace.very_deep_stockfish_nodes,
        match_random_seed=namespace.match_random_seed,
        dry_run=namespace.dry_run,
    )
    validate_input_paths(
        (arguments.experiment, arguments.run_directory, arguments.opening_manifest, arguments.stockfish_executable)
    )
    validate_new_output_root(arguments.output_root)
    node_counts = (
        arguments.policy_only_stockfish_nodes,
        arguments.shallow_stockfish_nodes,
        arguments.deep_stockfish_nodes,
        arguments.very_deep_stockfish_nodes,
    )
    if any(nodes <= 0 for nodes in node_counts):
        raise ValueError('Every selected Stockfish node count must be positive.')
    if arguments.checkpoint_generation < 0 or arguments.match_random_seed < 0:
        raise ValueError('Checkpoint generation and match seed must be nonnegative.')
    return arguments


def main() -> None:
    raise SystemExit(run_evaluations(parse_arguments()))


if __name__ == '__main__':
    main()

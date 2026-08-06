from __future__ import annotations

import argparse
from pathlib import Path

from src.train.ChessReplay import inspect_chess_archives, rebuild_chess_replay


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect chess completed-game archives or rebuild RAM replay.')
    subcommands = parser.add_subparsers(dest='command', required=True)
    inspect_parser = subcommands.add_parser('inspect', help='Validate and summarize model-generation archives.')
    inspect_parser.add_argument('run_path', type=Path)
    rebuild_parser = subcommands.add_parser('rebuild', help='Validate archives and deterministically rebuild replay.')
    rebuild_parser.add_argument('run_path', type=Path)
    rebuild_parser.add_argument('--capacity', type=int, required=True)
    rebuild_parser.add_argument('--sampler-seed', type=int, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == 'inspect':
        inspections = inspect_chess_archives(arguments.run_path)
        for inspection in inspections:
            print(
                f'model_generation={inspection.model_generation} games={inspection.game_count} '
                f'eligible_samples={inspection.eligible_sample_count} '
                f'completed_searches={inspection.completed_searches} bytes={inspection.byte_count} '
                f'path={inspection.path}'
            )
        return
    snapshot = rebuild_chess_replay(
        arguments.run_path,
        capacity=arguments.capacity,
        sampler_seed=arguments.sampler_seed,
    )
    print(
        f'credited_samples={snapshot.credited_samples} '
        f'credited_completed_searches={snapshot.credited_completed_searches} '
        f'live_samples={len(snapshot.samples)} sampler_seed={snapshot.sampler_seed}'
    )


if __name__ == '__main__':
    main()

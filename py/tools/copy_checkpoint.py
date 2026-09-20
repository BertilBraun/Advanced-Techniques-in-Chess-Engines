from __future__ import annotations

import argparse
from pathlib import Path

from src.training.checkpoint.persistence import import_checkpoint


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Copy and verify one complete training checkpoint.')
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    checkpoint = import_checkpoint(
        arguments.source_manifest.resolve(),
        arguments.generation,
        arguments.destination.resolve(),
    )
    print(checkpoint.manifest_path)


if __name__ == '__main__':
    main()

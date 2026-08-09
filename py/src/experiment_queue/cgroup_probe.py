from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cgroup-processes', required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.cgroup_processes.write_text(f'{os.getpid()}\n', encoding='ascii')


if __name__ == '__main__':
    main()

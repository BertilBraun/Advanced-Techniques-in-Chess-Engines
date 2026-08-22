from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run one queue experiment with its revision-matched approval.')
    parser.add_argument('--approval-directory', required=True, type=Path)
    parser.add_argument('--run-config', required=True, type=Path)
    return parser.parse_args()


def repository_revision(repository_root: Path) -> str:
    result = subprocess.run(
        ('git', 'rev-parse', 'HEAD'),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def approval_path(approval_directory: Path, run_configuration_path: Path) -> Path:
    return approval_directory.resolve() / f'{run_configuration_path.stem}.json'


def main() -> None:
    arguments = parse_arguments()
    repository_root = Path(__file__).resolve().parent.parent
    run_configuration_path = arguments.run_config.resolve()
    command = (
        sys.executable,
        str(repository_root / 'py' / 'train.py'),
        '--expected-source-revision',
        repository_revision(repository_root),
        '--approval-file',
        str(approval_path(arguments.approval_directory, run_configuration_path)),
        '--run-config',
        str(run_configuration_path),
    )
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()

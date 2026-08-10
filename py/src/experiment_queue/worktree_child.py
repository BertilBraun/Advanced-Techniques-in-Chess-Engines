from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-worktree', required=True, type=Path)
    parser.add_argument('--runtime-directory', required=True, type=Path)
    parser.add_argument('--setup-commands', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    setup_commands = tuple(tuple(command) for command in json.loads(arguments.setup_commands))
    command = tuple(arguments.command[1:] if arguments.command[:1] == ['--'] else arguments.command)
    if not command:
        raise ValueError('Runner command must not be empty.')
    for setup_command in setup_commands:
        subprocess.run(setup_command, cwd=arguments.source_worktree, check=True)
    completed = subprocess.run(command, cwd=arguments.runtime_directory, check=False)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()

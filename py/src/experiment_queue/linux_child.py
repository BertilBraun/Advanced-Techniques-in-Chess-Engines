from __future__ import annotations

import argparse
import os


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-affinity', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if not arguments.command:
        parser.error('a runner command is required')
    return arguments


def main() -> None:
    arguments = parse_arguments()
    cpu_affinity = {int(cpu_index) for cpu_index in arguments.cpu_affinity.split(',')}
    os.sched_setaffinity(0, cpu_affinity)
    os.execvpe(arguments.command[0], arguments.command, os.environ)


if __name__ == '__main__':
    main()

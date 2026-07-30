from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

from src.az.experiment.lifecycle import ExperimentRunRepository
from src.az.experiment.calibration import calibrate_run, load_calibration_request


def _control_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Control an authenticated Go experiment lifecycle.')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('stop', 'status', 'calibrate'):
        command = commands.add_parser(name)
        command.add_argument('--run-directory', type=Path, required=True)
        if name == 'calibrate':
            command.add_argument('--request', type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    selected = tuple(sys.argv[1:] if arguments is None else arguments)
    if selected and selected[0] in ('stop', 'status', 'calibrate'):
        parsed = _control_parser().parse_args(selected)
        repository = ExperimentRunRepository(parsed.run_directory.resolve())
        if parsed.command == 'stop':
            print(repository.request_stop().model_dump_json(indent=2))
        elif parsed.command == 'status':
            print(repository.load().model_dump_json(indent=2))
        else:
            print(
                calibrate_run(
                    repository,
                    load_calibration_request(parsed.request.resolve()),
                ).model_dump_json(indent=2)
            )
        return 0
    os.execv(
        sys.executable,
        (sys.executable, '-m', 'src.az.experiment.phase_cli', *selected),
    )
    raise AssertionError('Process replacement unexpectedly returned.')


if __name__ == '__main__':
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment_queue.configuration import load_queue_configuration
from src.experiment_queue.runner import ExperimentQueueRunner
from src.experiment_queue.state import FailedExperimentStatus, load_queue_summary
from src.experiment_queue.validation import validate_queue_for_launch


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run or inspect the resource-aware experiment queue.')
    subparsers = parser.add_subparsers(dest='action', required=True)

    run_parser = subparsers.add_parser('run', help='Validate and run a queue.')
    run_parser.add_argument('--queue-config', required=True, type=Path)

    status_parser = subparsers.add_parser('status', help='Display an existing queue summary.')
    status_parser.add_argument('--summary', required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.action == 'status':
        print(load_queue_summary(arguments.summary.resolve()).model_dump_json(indent=2))
        return 0

    configuration = load_queue_configuration(arguments.queue_config.resolve())
    validated_queue = validate_queue_for_launch(configuration)
    summary = ExperimentQueueRunner(validated_queue).run()
    print(summary.model_dump_json(indent=2))
    return 1 if any(isinstance(status, FailedExperimentStatus) for status in summary.experiments) else 0


if __name__ == '__main__':
    raise SystemExit(main())

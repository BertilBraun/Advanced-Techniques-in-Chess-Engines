from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment_queue.result_export import (
    ExplicitExperimentConfiguration,
    ExportRequest,
    export_experiment_results,
)


def _explicit_experiment(value: str) -> ExplicitExperimentConfiguration:
    experiment_id, separator, experiment_path = value.partition('=')
    if not separator or not experiment_id or not experiment_path:
        raise argparse.ArgumentTypeError('Explicit experiments must use EXPERIMENT_ID=CONFIGURATION_PATH.')
    return ExplicitExperimentConfiguration(experiment_id=experiment_id, experiment_file=Path(experiment_path))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export terminal experiment-queue evidence to one verified ZIP.')
    parser.add_argument('--queue-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--queue-summary', type=Path)
    parser.add_argument('--tensorboard-log-root', type=Path)
    parser.add_argument('--experiment-id', action='append', default=[])
    parser.add_argument('--experiment-config', action='append', default=[], type=_explicit_experiment)
    parser.add_argument('--queue-stdout-log', type=Path)
    parser.add_argument('--queue-stderr-log', type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    request = ExportRequest(
        queue_configuration_path=arguments.queue_config,
        output_path=arguments.output,
        queue_summary_path=arguments.queue_summary,
        tensorboard_log_root=arguments.tensorboard_log_root,
        experiment_ids=tuple(arguments.experiment_id),
        explicit_experiments=tuple(arguments.experiment_config),
        queue_stdout_log=arguments.queue_stdout_log,
        queue_stderr_log=arguments.queue_stderr_log,
    )
    manifest = export_experiment_results(request)
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

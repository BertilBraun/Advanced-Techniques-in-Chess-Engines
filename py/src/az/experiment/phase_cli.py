from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
from uuid import UUID

from src.az.config.dependency_lock import parse_pinned_dependency_lock
from src.az.config.manifest import (
    DependencyDeclaration,
    build_manifest,
    current_python_build,
    file_sha256,
)
from src.az.config.serialization import (
    load_resolved_configuration,
    model_sha256,
    resolve_file,
    write_resolved_configuration,
)
from src.az.experiment.lifecycle import ExperimentRunRepository
from src.az.experiment.calibration import calibrate_run, load_calibration_request
from src.az.experiment.environment import inspect_hardware
from src.az.experiment.runner import (
    run_evaluation,
    run_remaining,
    run_reporting,
    run_training_window,
)
from src.az.experiment.smoke import local_cpu_smoke_configuration


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run an authenticated Go experiment lifecycle.')
    commands = parser.add_subparsers(dest='command', required=True)

    validate = commands.add_parser('validate', help='Validate and resolve an authoring configuration in memory.')
    validate.add_argument('configuration', type=Path)

    resolve = commands.add_parser('resolve', help='Write a fully resolved immutable configuration.')
    resolve.add_argument('configuration', type=Path)
    resolve.add_argument('--output', type=Path, required=True)

    smoke = commands.add_parser('write-smoke-config', help='Write the typed local CPU readiness profile.')
    smoke.add_argument('--output', type=Path, required=True)

    freeze = commands.add_parser('freeze', help='Freeze a resolved configuration into a new run directory.')
    freeze.add_argument('configuration', type=Path)
    freeze.add_argument('--run-directory', type=Path, required=True)
    freeze.add_argument('--artifact-root', type=Path, required=True)
    freeze.add_argument('--run-id', type=UUID, required=True)
    freeze.add_argument('--repository-root', type=Path, required=True)
    freeze.add_argument('--dependency-lock', type=Path, required=True)
    freeze.add_argument('--build-id', default='development')
    freeze.add_argument('--reference-artifact-root', type=Path)

    for name in ('run', 'training-run', 'evaluate', 'report', 'stop', 'resume', 'status'):
        command = commands.add_parser(name)
        command.add_argument('--run-directory', type=Path, required=True)
        if name == 'resume':
            command.add_argument('--recover-crash', action='store_true')
    calibrate = commands.add_parser(
        'calibrate',
        help='Publish an authenticated adaptive-search calibration from committed traces.',
    )
    calibrate.add_argument('--run-directory', type=Path, required=True)
    calibrate.add_argument('--request', type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = create_parser().parse_args(arguments)
    if parsed.command == 'validate':
        print(model_sha256(resolve_file(parsed.configuration.resolve())))
        return 0
    if parsed.command == 'resolve':
        configuration = resolve_file(parsed.configuration.resolve())
        write_resolved_configuration(parsed.output.resolve(), configuration)
        print(model_sha256(configuration))
        return 0
    if parsed.command == 'write-smoke-config':
        write_resolved_configuration(parsed.output.resolve(), local_cpu_smoke_configuration())
        return 0
    repository = ExperimentRunRepository(parsed.run_directory.resolve())
    try:
        match parsed.command:
            case 'freeze':
                configuration = load_resolved_configuration(parsed.configuration.resolve())
                lock_path = parsed.dependency_lock.resolve()
                run_directory = parsed.run_directory.resolve()
                expected_run_directory = parsed.artifact_root.resolve().joinpath(
                    *configuration.experiment.output_directory.parts
                )
                if run_directory != expected_run_directory:
                    raise ValueError('Run directory must equal artifact root plus the resolved output directory.')
                manifest = build_manifest(
                    configuration=configuration,
                    repository_root=parsed.repository_root.resolve(),
                    build=current_python_build(parsed.build_id),
                    dependencies=DependencyDeclaration(
                        lock_file=lock_path,
                        lock_file_sha256=file_sha256(lock_path),
                        packages=(
                            parse_pinned_dependency_lock(lock_path)
                            if configuration.experiment.manifest_policy.record_dependency_versions
                            else ()
                        ),
                    ),
                    hardware=inspect_hardware(parsed.artifact_root.resolve()),
                )
                state = repository.freeze(
                    parsed.configuration.resolve(),
                    parsed.run_id,
                    manifest,
                    parsed.repository_root.resolve(),
                    (None if parsed.reference_artifact_root is None else parsed.reference_artifact_root.resolve()),
                )
            case 'run':
                state = run_remaining(repository)
            case 'training-run':
                state = run_training_window(repository)
            case 'evaluate':
                state = run_evaluation(repository)
            case 'report':
                state = run_reporting(repository)
            case 'stop':
                request = repository.request_stop()
                print(request.model_dump_json(indent=2))
                return 0
            case 'resume':
                repository.resume(parsed.recover_crash)
                state = run_remaining(repository)
            case 'status':
                state = repository.load()
            case 'calibrate':
                reference = calibrate_run(
                    repository,
                    load_calibration_request(parsed.request.resolve()),
                )
                print(reference.model_dump_json(indent=2))
                return 0
    except Exception as error:
        if parsed.command in ('run', 'training-run', 'evaluate', 'report', 'resume'):
            repository.record_failure(f'{type(error).__name__}: {error}')
        raise
    print(state.model_dump_json(indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

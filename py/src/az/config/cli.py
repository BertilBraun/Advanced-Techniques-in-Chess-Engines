from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.az.config.dependency_lock import parse_pinned_dependency_lock
from src.az.config.manifest import (
    DependencyDeclaration,
    HardwareDeclaration,
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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Validate and resolve Go-first experiment configurations.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    validate_parser = subparsers.add_parser('validate', help='Validate an authoring configuration.')
    validate_parser.add_argument('configuration', type=Path)

    resolve_parser = subparsers.add_parser('resolve', help='Write a fully materialized configuration.')
    resolve_parser.add_argument('configuration', type=Path)
    resolve_parser.add_argument('--output', type=Path, required=True)

    print_parser = subparsers.add_parser('print-config', help='Print a resolved configuration.')
    print_parser.add_argument('configuration', type=Path)
    print_parser.add_argument('--resolved-input', action='store_true')

    manifest_parser = subparsers.add_parser('manifest', help='Print a manifest without launching a run.')
    manifest_parser.add_argument('configuration', type=Path)
    manifest_parser.add_argument('--repository-root', type=Path, required=True)
    manifest_parser.add_argument('--dependency-lock', type=Path, required=True)
    manifest_parser.add_argument('--gpu-model', required=True)
    manifest_parser.add_argument('--gpu-count', type=int, required=True)
    manifest_parser.add_argument('--logical-cpu-count', type=int, required=True)
    manifest_parser.add_argument('--ram-gib', type=float, required=True)
    manifest_parser.add_argument('--free-disk-gib', type=float, required=True)
    manifest_parser.add_argument('--build-id', default='development')
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = create_parser().parse_args(arguments)
    match parsed.command:
        case 'validate':
            configuration = resolve_file(parsed.configuration)
            print(model_sha256(configuration))
        case 'resolve':
            configuration = resolve_file(parsed.configuration)
            write_resolved_configuration(parsed.output, configuration)
            print(model_sha256(configuration))
        case 'print-config':
            configuration = (
                load_resolved_configuration(parsed.configuration)
                if parsed.resolved_input
                else resolve_file(parsed.configuration)
            )
            print(configuration.model_dump_json(indent=2))
        case 'manifest':
            configuration = load_resolved_configuration(parsed.configuration)
            lock_path = parsed.dependency_lock.resolve()
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
                hardware=HardwareDeclaration(
                    gpu_model=parsed.gpu_model,
                    gpu_count=parsed.gpu_count,
                    logical_cpu_count=parsed.logical_cpu_count,
                    ram_gib=parsed.ram_gib,
                    free_disk_gib=parsed.free_disk_gib,
                ),
            )
            print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
